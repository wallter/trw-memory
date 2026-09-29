"""PRD-CORE-253 FR05 curate verbs + FR01 moved-checkout detection.

Every test drives a real ``SQLiteBackend`` on disk: a bulk re-key is exactly
where rows go missing, so the assertions are on what the store actually holds
afterwards rather than on what the function reported.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import make_entry
from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.curate import (
    NamespaceStores,
    detect_moved_checkout,
    merge_namespace,
    rename_namespace,
    store_census,
)
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.namespace_admin import (
    memory_namespace_diagnose_impl,
    memory_namespace_merge_impl,
    memory_namespace_rename_impl,
)

OLD = "project:trw-framework-11111111"
NEW = "project:trw-framework-22222222"
OTHER = "project:something-else-33333333"


@pytest.fixture
def backend(tmp_path: Path) -> StorageBackend:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


def _seed(backend: StorageBackend, namespace: str, ids: list[str]) -> None:
    for entry_id in ids:
        backend.store(make_entry(entry_id=entry_id, namespace=namespace, content=f"{entry_id} in {namespace}"))


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------


def test_rename_relabels_every_row_and_is_idempotent(backend: StorageBackend) -> None:
    """FR05: the repair path for a moved checkout carries every row forward."""
    _seed(backend, OLD, ["M-1", "M-2", "M-3"])

    result = rename_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.status, result.source_rows, result.moved, result.skipped) == ("renamed", 3, 3, 0)
    assert backend.count(namespace=OLD) == 0
    assert backend.count(namespace=NEW) == 3
    assert backend.get("M-2", namespace=NEW) is not None
    assert backend.get("M-2", namespace=OLD) is None

    # Idempotent: a re-run finds an empty source and reports zero moved.
    again = rename_namespace(NamespaceStores.shared(backend), OLD, NEW)
    assert (again.status, again.moved) == ("noop", 0)
    assert backend.count(namespace=NEW) == 3


def test_rename_moves_rows_byte_identical(backend: StorageBackend) -> None:
    """``_move_rows`` is exempted from the SEC-001 store-gate totality guard as
    ``rewrite_of_persisted_entry`` (already-gated content, re-labelled only) --
    this pins the half of that exemption a static scan cannot prove: the moved
    row's content is byte-identical to the source, and only ``namespace``
    (plus the storage-managed ``updated_at``) changed.
    """
    entry = make_entry(
        entry_id="M-1",
        namespace=OLD,
        content="use absolute paths",
        detail="a gotcha worth remembering",
        tags=["gotcha", "paths"],
        importance=0.73,
    )
    backend.store(entry)
    before = backend.get("M-1", namespace=OLD)
    assert before is not None
    # ``namespace``/``updated_at`` change by design; ``sync_seq`` is the
    # backend's own write-bookkeeping counter (bumped by every ``store()``,
    # including this one) -- storage plumbing, not row CONTENT.
    _NON_CONTENT_FIELDS = {"namespace", "updated_at", "sync_seq"}
    before_hash = hashlib.sha256(before.model_dump_json(exclude=_NON_CONTENT_FIELDS).encode("utf-8")).hexdigest()

    rename_namespace(NamespaceStores.shared(backend), OLD, NEW)

    after = backend.get("M-1", namespace=NEW)
    assert after is not None
    after_hash = hashlib.sha256(after.model_dump_json(exclude=_NON_CONTENT_FIELDS).encode("utf-8")).hexdigest()

    assert after_hash == before_hash, "a namespace move must not alter row content"
    assert after.namespace == NEW
    assert after.content == before.content == "use absolute paths"
    assert after.detail == before.detail
    assert after.tags == before.tags
    assert after.importance == before.importance


def test_rename_refuses_a_populated_destination(backend: StorageBackend) -> None:
    """That case is a merge, and making the caller say so blocks a silent union."""
    _seed(backend, OLD, ["M-1"])
    _seed(backend, NEW, ["M-9"])

    with pytest.raises(ConfigError, match="Use merge"):
        rename_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert backend.count(namespace=OLD) == 1, "the refusal must not have moved anything"
    assert backend.count(namespace=NEW) == 1


@pytest.mark.parametrize("two_stores", [False, True])
def test_a_rename_never_overwrites_a_row_another_connection_committed(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, two_stores: bool
) -> None:
    """rc5 C12: a store landing between the empty-destination check and the move must survive it."""
    destination = SQLiteBackend(tmp_path / "destination.db") if two_stores else backend
    stores = NamespaceStores(source=backend, destination=destination)
    _seed(backend, OLD, ["M-1"])
    other = SQLiteBackend(Path(destination._db_path))  # another connection, open before the rename
    real_count = destination.count
    threads: list[threading.Thread] = []

    def count_then_race(*, namespace: str | None = None) -> int:
        seen = real_count(namespace=namespace)
        if namespace == NEW:  # another connection commits into the destination right after the check
            writer = threading.Thread(
                target=other.store,
                args=(make_entry(entry_id="M-1", namespace=NEW, content="committed by another writer"),),
            )
            writer.start()
            threads.append(writer)
            # Unguarded, the commit lands well inside this window; a check that holds the
            # write lock keeps the writer waiting until the move has committed.
            writer.join(timeout=5.0)
        return seen

    monkeypatch.setattr(destination, "count", count_then_race)
    rename_namespace(stores, OLD, NEW)
    for writer in threads:
        writer.join()
    other.close()

    stored = destination.get("M-1", namespace=NEW)
    if two_stores:
        destination.close()
    assert stored is not None
    assert stored.content == "committed by another writer", "the rename overwrote a committed row"


def test_rename_refuses_a_non_transactional_destination_rather_than_race_it(tmp_path: Path) -> None:
    """B71-61: YAML's store()/delete() take no lock at all, and update()'s lock is
    per-entry-file rather than per-namespace, so there is no shared lock the
    rename's empty-destination check could hold to make the check-and-move atomic
    the way 8d1ad71cc made it atomic on SQLite. Rather than let a row another
    process writes into the destination between the check and the move be
    silently overwritten, the rename refuses outright.
    """
    from trw_memory.storage.yaml_backend import YAMLBackend

    source = SQLiteBackend(tmp_path / "source.db")
    destination = YAMLBackend(tmp_path / "dest-entries")
    stores = NamespaceStores(source=source, destination=destination)
    _seed(source, OLD, ["M-1"])

    with pytest.raises(ConfigError, match="does not support transactions"):
        rename_namespace(stores, OLD, NEW)

    assert source.count(namespace=OLD) == 1, "the refusal must not have moved anything out of source"
    assert destination.count(namespace=NEW) == 0, "the refusal must not have written anything to the destination"
    source.close()


def test_rename_refuses_a_self_rename(backend: StorageBackend) -> None:
    _seed(backend, OLD, ["M-1"])

    with pytest.raises(ConfigError, match="same namespace"):
        rename_namespace(NamespaceStores.shared(backend), OLD, OLD)


def test_rename_rejects_an_ungrammatical_namespace(backend: StorageBackend) -> None:
    with pytest.raises(ConfigError):
        rename_namespace(NamespaceStores.shared(backend), OLD, "not a namespace!")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_keeps_the_destination_row_and_reports_the_skip(backend: StorageBackend) -> None:
    """FR05: a conflict is reported, never resolved by overwriting."""
    _seed(backend, OLD, ["M-1", "M-shared"])
    backend.store(make_entry(entry_id="M-shared", namespace=NEW, content="the destination's version"))

    result = merge_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.status, result.source_rows, result.moved, result.skipped) == ("merged", 2, 1, 1)
    kept = backend.get("M-shared", namespace=NEW)
    assert kept is not None and kept.content == "the destination's version"
    assert backend.get("M-1", namespace=NEW) is not None
    # The skipped row stays put: a merge never deletes a row it did not copy.
    assert backend.get("M-shared", namespace=OLD) is not None
    assert backend.count(namespace=OLD) == 1


def test_merge_of_an_empty_source_is_a_noop(backend: StorageBackend) -> None:
    _seed(backend, NEW, ["M-9"])

    result = merge_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.status, result.moved, result.skipped) == ("noop", 0, 0)
    assert backend.count(namespace=NEW) == 1


def test_merge_terminates_when_every_row_conflicts(backend: StorageBackend) -> None:
    """The skipped rows stay in the source, so the loop must not re-read forever."""
    _seed(backend, OLD, ["M-1", "M-2"])
    _seed(backend, NEW, ["M-1", "M-2"])

    result = merge_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.moved, result.skipped) == (0, 2)
    assert backend.count(namespace=OLD) == 2


# ---------------------------------------------------------------------------
# moved-checkout detection (FR01)
# ---------------------------------------------------------------------------
#
# These drive ``store_census`` -- the census the PRODUCTION callers use
# (``memory_namespace_diagnose`` and trw-mcp's ``_moved_checkout_readback``).
# They used to call a single-backend ``namespace_census`` that nothing but this
# module called; asserting FR01 through a function no caller reaches proved the
# detector worked against an input production never builds. Seeding through
# ``create_backend_from_config`` also exercises the split layout store_census
# has to span, which the single open backend could not represent at all.


def _census_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "census"))


def _seed_via_config(config: MemoryConfig, namespace: str, ids: list[str]) -> None:
    with create_backend_from_config(config, namespace) as store:
        _seed(store, namespace, ids)


def test_moved_checkout_is_detected_and_never_repaired(tmp_path: Path) -> None:
    """FR01: same slug, different digest, empty current namespace -> report only."""
    config = _census_config(tmp_path)
    _seed_via_config(config, OLD, ["M-1", "M-2"])

    observation = detect_moved_checkout(NEW, store_census(config))

    assert observation is not None
    assert observation.current_namespace == NEW
    assert observation.current_rows == 0
    assert [candidate.namespace for candidate in observation.candidates] == [OLD]
    assert observation.candidates[0].rows == 2
    assert observation.repair_command == f"trw-memory namespace rename {OLD} {NEW}"
    # Detection is a read: nothing moved.
    assert store_census(config) == {OLD: 2}


def test_a_populated_current_namespace_reports_nothing(tmp_path: Path) -> None:
    """The signal requires an EMPTY current namespace, or every project trips it."""
    config = _census_config(tmp_path)
    _seed_via_config(config, OLD, ["M-1"])
    _seed_via_config(config, NEW, ["M-2"])

    assert detect_moved_checkout(NEW, store_census(config)) is None


def test_a_different_slug_reports_nothing(tmp_path: Path) -> None:
    """A fresh clone of a differently named project produces none of the signal."""
    config = _census_config(tmp_path)
    _seed_via_config(config, OTHER, ["M-1"])

    assert detect_moved_checkout(NEW, store_census(config)) is None


def test_a_non_project_namespace_reports_nothing(tmp_path: Path) -> None:
    """``user:local`` and ``global`` are not checkouts and cannot be moved."""
    config = _census_config(tmp_path)
    _seed_via_config(config, OLD, ["M-1"])

    assert detect_moved_checkout("user:local", store_census(config)) is None
    assert detect_moved_checkout("global", store_census(config)) is None


# ---------------------------------------------------------------------------
# The tool wrappers: RBAC before any row is touched
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, **overrides: object) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "store"), **overrides)


def test_curate_tools_check_write_permission_on_both_namespaces(tmp_path: Path) -> None:
    """A late permission check on a bulk re-key is a half-completed move."""
    config = _config(
        tmp_path,
        rbac_enabled=True,
        default_role="none",
        namespace_roles={OLD: "writer"},  # writer on the source only
    )

    refused = memory_namespace_rename_impl(OLD, NEW, config=config)

    assert refused["status"] == "forbidden"
    assert NEW in str(refused["error"])


def test_curate_tools_round_trip_through_the_registered_impls(tmp_path: Path) -> None:
    """The tool surface, not just the engine, moves rows and reports counts."""
    config = _config(tmp_path)
    with create_backend_from_config(config, OLD) as backend:
        _seed(backend, OLD, ["M-1", "M-2"])

    renamed = memory_namespace_rename_impl(OLD, NEW, config=config)
    assert (renamed["status"], renamed["moved"]) == ("renamed", 2)

    merged = memory_namespace_merge_impl(NEW, OTHER, config=config)
    assert (merged["status"], merged["moved"]) == ("merged", 2)


def test_diagnose_tool_resolves_the_callers_namespace_and_writes_nothing(tmp_path: Path) -> None:
    """An empty ``namespace`` argument resolves the caller's FR01 identity."""
    config = _config(tmp_path)

    reported = memory_namespace_diagnose_impl(NEW, config=config)

    assert reported["status"] == "ok"
    assert reported["namespace"] == NEW
    assert reported["moved_checkout"] is None

    resolved = memory_namespace_diagnose_impl(config=config)
    assert str(resolved["namespace"]).startswith("project:")


def test_a_yaml_backend_crosses_two_stores_rather_than_sharing_one(tmp_path: Path) -> None:
    """The shared-vs-crossed predicate was wrong for YAML, and wrong SILENTLY.

    A YAML namespace lives in its own ``<ns_dir>/entries`` directory, exactly as
    a SQLite namespace lives in its own file. The first version of ``_open_stores``
    treated every non-SQLite config as "shared", so a cross-namespace YAML move
    opened the DESTINATION's directory, found zero source rows and reported a
    clean no-op -- the worst possible answer for a bulk re-key, because it looks
    like success. Exercised via merge, not rename: B71-61 makes rename refuse on
    YAML outright (see ``test_rename_refuses_a_non_transactional_destination...``),
    so merge is the write verb left to prove the crossed-directory resolution.
    """
    config = MemoryConfig(storage_path=str(tmp_path / "store"), storage_backend="yaml")
    with create_backend_from_config(config, OLD) as backend:
        _seed(backend, OLD, ["M-1", "M-2"])

    merged = memory_namespace_merge_impl(OLD, NEW, config=config)

    assert (merged["status"], merged["moved"]) == ("merged", 2)
    with create_backend_from_config(config, NEW) as backend:
        assert backend.count(namespace=NEW) == 2
    with create_backend_from_config(config, OLD) as backend:
        assert backend.count(namespace=OLD) == 0


def test_a_single_store_config_shares_one_backend(tmp_path: Path) -> None:
    """Under ``memory_single_store_path`` both namespaces are one file.

    Opening that file twice would put two connections on a store whose WAL
    mitigation assumes one, so the shared case has to be detected, and the move
    then runs inside a single transaction.
    """
    store = tmp_path / "user" / "memory.db"
    store.parent.mkdir(parents=True)
    config = MemoryConfig(storage_path=str(store.parent), memory_single_store_path=str(store))
    with create_backend_from_config(config, OLD) as backend:
        _seed(backend, OLD, ["M-1", "M-2", "M-3"])

    renamed = memory_namespace_rename_impl(OLD, NEW, config=config)

    assert (renamed["status"], renamed["moved"]) == ("renamed", 3)
    assert sorted(p.name for p in store.parent.rglob("memory.db")) == ["memory.db"]
    with create_backend_from_config(config, NEW) as backend:
        assert backend.count(namespace=NEW) == 3
        assert backend.count(namespace=OLD) == 0


def test_store_census_spans_a_single_store_and_a_split_layout(tmp_path: Path) -> None:
    """One census function, both layouts — the diagnose tool and trw-mcp share it."""
    split = MemoryConfig(storage_path=str(tmp_path / "split"))
    with create_backend_from_config(split, OLD) as backend:
        _seed(backend, OLD, ["M-1"])
    with create_backend_from_config(split, OTHER) as backend:
        _seed(backend, OTHER, ["M-2", "M-3"])
    assert store_census(split) == {OLD: 1, OTHER: 2}

    store = tmp_path / "one" / "memory.db"
    store.parent.mkdir(parents=True)
    single = MemoryConfig(storage_path=str(store.parent), memory_single_store_path=str(store))
    with create_backend_from_config(single, OLD) as backend:
        _seed(backend, OLD, ["M-1"])
        _seed(backend, OTHER, ["M-2", "M-3"])
    assert store_census(single) == {OLD: 1, OTHER: 2}


# ---------------------------------------------------------------------------
# merge paging: the window has to advance past a batch of conflicts
# ---------------------------------------------------------------------------


def _seed_ranked(backend: StorageBackend, namespace: str, ids: list[str]) -> None:
    """Seed *ids* newest-first, one minute apart, so the listing order is pinned.

    ``list_entries`` orders by ``updated_at`` desc, so ``ids[0]`` is the first
    row any page returns. Pinning it is what lets a test say "the three NEWEST
    rows conflict" and mean it.
    """
    base = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    for rank, entry_id in enumerate(ids):
        entry = make_entry(entry_id=entry_id, namespace=namespace, content=f"{entry_id} in {namespace}")
        backend.store(entry.model_copy(update={"updated_at": base - timedelta(minutes=rank)}))


def test_merge_moves_rows_ranked_below_a_full_window_of_conflicts(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stranding bug: a whole window of conflicts used to END the merge.

    Ten source rows, batch limit three, and the three NEWEST conflict with the
    destination. Under the old offset-window loop the first pass skipped all
    three, the second pass re-read the identical window, the in-memory ``seen``
    filter emptied it, and the loop broke -- reporting ``status="merged"`` while
    seven perfectly movable rows stayed behind. The keyset cursor advances past
    the conflicts, so the remaining seven move.
    """
    monkeypatch.setattr("trw_memory.namespaces.curate._BATCH_LIMIT", 3)
    source_ids = [f"M-{index:02d}" for index in range(10)]
    _seed_ranked(backend, OLD, source_ids)
    conflicts = source_ids[:3]
    for entry_id in conflicts:
        backend.store(make_entry(entry_id=entry_id, namespace=NEW, content="the destination's version"))

    result = merge_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.status, result.source_rows, result.moved, result.skipped) == ("merged", 10, 7, 3)
    assert backend.count(namespace=NEW) == 10, "3 destination rows + the 7 that moved"
    for entry_id in source_ids[3:]:
        assert backend.get(entry_id, namespace=NEW) is not None, f"{entry_id} was stranded in the source"
    # The source keeps exactly the conflicts -- no more, no fewer.
    assert backend.count(namespace=OLD) == 3
    assert sorted(entry.id for entry in backend.list_entries(namespace=OLD, limit=100)) == sorted(conflicts)


def test_merge_terminates_when_every_row_conflicts_across_many_windows(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is movable, several windows deep: still terminates, still honest."""
    monkeypatch.setattr("trw_memory.namespaces.curate._BATCH_LIMIT", 3)
    source_ids = [f"M-{index:02d}" for index in range(10)]
    _seed_ranked(backend, OLD, source_ids)
    for entry_id in source_ids:
        backend.store(make_entry(entry_id=entry_id, namespace=NEW, content="the destination's version"))

    result = merge_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert (result.status, result.moved, result.skipped) == ("merged", 0, 10)
    assert backend.count(namespace=OLD) == 10
    assert backend.count(namespace=NEW) == 10


def test_a_merge_that_leaves_rows_behind_raises_instead_of_reporting_merged(tmp_path: Path) -> None:
    """``status="merged"`` must be unreachable for a partial merge.

    The counts a verb returns are self-reported. This drives a backend whose
    listing under-serves -- the exact shape of the stranding bug -- and pins the
    independent post-condition: the source has to hold EXACTLY the skipped rows
    or the whole move rolls back with an error naming the numbers.
    """

    class _UnderServingStore(SQLiteBackend):
        """A store whose listing stops early; everything else is the real path."""

        def list_entries(self, **kwargs: object) -> list[MemoryEntry]:  # type: ignore[override]
            return []

    store = _UnderServingStore(tmp_path / "memory.db")
    try:
        _seed(store, OLD, ["M-1", "M-2", "M-3"])

        with pytest.raises(StorageError, match="incomplete move"):
            merge_namespace(NamespaceStores.shared(store), OLD, NEW)

        assert store.count(namespace=OLD) == 3, "the failed move must roll back"
        assert store.count(namespace=NEW) == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# graph edges travel with their rows
# ---------------------------------------------------------------------------


def _edge(backend: SQLiteBackend, namespace: str, source: str, target: str) -> None:
    from trw_memory._graph_primitives import _upsert_edge

    with backend._lock:
        _upsert_edge(backend._conn, source, target, "related_to", 0.8, "2026-09-23T00:00:00+00:00", namespace=namespace)
        backend._conn.commit()


def test_a_rename_carries_the_graph_edges(backend: SQLiteBackend) -> None:
    _seed(backend, OLD, ["M-1", "M-2", "M-3"])
    _edge(backend, OLD, "M-1", "M-2")
    _edge(backend, OLD, "M-2", "M-3")

    rename_namespace(NamespaceStores.shared(backend), OLD, NEW)

    assert sorted((e.source_id, e.target_id) for e in backend.graph_edges(NEW)) == [("M-1", "M-2"), ("M-2", "M-3")]
    assert backend.graph_edges(OLD) == []


def test_a_cross_store_merge_carries_the_moved_rows_edges(tmp_path: Path) -> None:
    source, destination = SQLiteBackend(tmp_path / "project.db"), SQLiteBackend(tmp_path / "user.db")
    try:
        _seed(source, "default", ["M-1", "M-2"])
        _edge(source, "default", "M-1", "M-2")

        merge_namespace(NamespaceStores(source=source, destination=destination), "default", NEW)
        again = merge_namespace(NamespaceStores(source=source, destination=destination), "default", NEW)

        (edge,) = destination.graph_edges(NEW)
        assert (edge.source_id, edge.target_id, edge.edge_type, edge.weight) == ("M-1", "M-2", "related_to", 0.8)
        assert again.status == "noop"
    finally:
        source.close()
        destination.close()


# ---------------------------------------------------------------------------
# B71-96 (PRD-CORE-307 FR06): the served verbs move in batches, each its own lane job
# ---------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryConfig:
    """The served tools read ``MemoryConfig()`` from the environment; batches of two rows."""
    from trw_memory.daemon._lane import LaneBudget
    from trw_memory.daemon._offload import shutdown_offload_pool
    from trw_memory.tools import namespace_admin

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    small = LaneBudget("maintenance", seconds=0.25, rows=2, background=True)
    monkeypatch.setattr(namespace_admin, "MAINTENANCE", small, raising=False)
    shutdown_offload_pool()
    yield MemoryConfig()
    shutdown_offload_pool()


async def _serve(verb: str, source: str = OLD, destination: str = NEW) -> dict[str, object]:
    from trw_memory.server import mcp

    tool = await mcp.get_tool(f"memory_namespace_{verb}")
    result = await tool.run({"source": source, "destination": destination})
    assert isinstance(result.structured_content, dict)
    return result.structured_content


def _ids(config: MemoryConfig, namespace: str) -> list[str]:
    with create_backend_from_config(config, namespace) as store:
        return sorted(entry.id for entry in store.list_entries(namespace=namespace, limit=1_000))


async def test_a_merge_larger_than_one_batch_gives_the_lane_to_another_tenant_between_batches(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B71-96: the move was ONE lane job, so a forget from another tenant waited for the whole namespace."""
    from trw_memory.daemon import _lane
    from trw_memory.tools import namespace_admin

    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, [f"M-{index}" for index in range(5)])
    order: list[str] = []
    real = namespace_admin.memory_namespace_merge_impl

    def _batch(*args: object, **kwargs: object) -> dict[str, object]:
        if not order:  # another tenant's interactive job queues while the first batch holds the lane
            _lane.submit(_lane.INTERACTIVE, OTHER, order.append, "other tenant")
        order.append("batch")
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(namespace_admin, "memory_namespace_merge_impl", _batch)

    reply = await _serve("merge")

    assert (reply["status"], reply["moved"], reply.get("complete")) == ("merged", 5, True), reply
    assert order.count("batch") == 3, order
    assert order.index("other tenant") < len(order) - 1, f"no batch ran after the other tenant's job: {order}"
    assert _ids(served, NEW) == [f"M-{index}" for index in range(5)]


@pytest.mark.parametrize("verb", ["merge", "rename"])
async def test_a_call_whose_budget_ends_mid_move_resumes_on_the_next_call(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """Each call moves one batch here (its budget is spent), says so, and the next call resumes: no row lost or
    duplicated, a merge's conflict still skipped, and a resumed rename accepts its own half-moved destination."""
    from trw_memory.daemon import _lane

    monkeypatch.setattr(_lane, "CALL_SECONDS", 0.0)
    source_ids = [f"M-{index}" for index in range(5)]
    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, source_ids)
    if verb == "merge":
        with create_backend_from_config(served, NEW) as store:
            store.store(make_entry(entry_id="M-2", namespace=NEW, content="the destination's version"))

    replies = [await _serve(verb)]
    while replies[-1].get("complete") is False:
        assert replies[-1]["status"] == "moving", replies[-1]
        assert len(replies) < 10, replies
        replies.append(await _serve(verb))

    assert len(replies) >= 3, replies
    final = replies[-1]
    assert final["status"] == ("merged" if verb == "merge" else "renamed"), final
    expected = (5, 4, 1) if verb == "merge" else (5, 5, 0)
    assert (final["source_rows"], final["moved"], final["skipped"]) == expected, final
    assert _ids(served, NEW) == source_ids
    assert _ids(served, OLD) == (["M-2"] if verb == "merge" else [])


async def test_a_rename_refuses_a_row_it_did_not_move_even_mid_move(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty-destination check survives batching: a foreign row before the move, or one another tenant
    stores between two batches, is refused; the rows this rename already moved are not."""
    from trw_memory.daemon import _lane

    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, ["M-1"])
    with create_backend_from_config(served, NEW) as store:
        _seed(store, NEW, ["M-9"])
    refused = await _serve("rename")
    assert refused["status"] == "invalid" and "Use merge" in str(refused["error"]), refused

    monkeypatch.setattr(_lane, "CALL_SECONDS", 0.0)
    fresh = "project:fresh-44444444"
    with create_backend_from_config(served, OTHER) as store:
        _seed(store, OTHER, ["O-1", "O-2", "O-3", "O-4"])
    first = await _serve("rename", OTHER, fresh)
    assert (first["status"], first["moved"]) == ("moving", 2), first
    with create_backend_from_config(served, fresh) as store:
        _seed(store, fresh, ["X-1"])

    again = await _serve("rename", OTHER, fresh)

    assert again["status"] == "invalid" and "Use merge" in str(again["error"]), again
    assert len(_ids(served, OTHER)) == 2, "the refusal moved nothing"


async def test_a_resumed_rename_never_overwrites_a_row_written_under_an_upcoming_source_id(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-307 sol r3 P1: the resume check compared only counts, so deleting one moved row and writing a
    destination row under an upcoming source id kept the count and the next batch overwrote the new row."""
    from trw_memory.daemon import _lane

    monkeypatch.setattr(_lane, "CALL_SECONDS", 0.0)
    fresh = "project:fresh-55555555"
    with create_backend_from_config(served, OTHER) as store:
        _seed(store, OTHER, ["O-1", "O-2", "O-3", "O-4"])
    first = await _serve("rename", OTHER, fresh)
    assert (first["status"], first["moved"]) == ("moving", 2), first
    with create_backend_from_config(served, fresh) as store:
        moved = sorted(entry.id for entry in store.list_entries(namespace=fresh, limit=10))
        store.delete(moved[0], namespace=fresh)
        upcoming = next(entry_id for entry_id in ["O-1", "O-2", "O-3", "O-4"] if entry_id not in moved)
        store.store(make_entry(entry_id=upcoming, namespace=fresh, content="new destination content must survive"))

    again = await _serve("rename", OTHER, fresh)

    assert again["status"] == "invalid" and "Use merge" in str(again["error"]), again
    with create_backend_from_config(served, fresh) as store:
        kept = store.get(upcoming, namespace=fresh)
    assert kept is not None and kept.content == "new destination content must survive"


async def test_a_namespace_past_the_old_whole_move_cap_moves_with_its_edges(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rc9 refused a namespace over CURATE_ROWS_MAX rows and edges (50,000); batched, any size moves."""
    from trw_memory.tools import namespace_admin

    monkeypatch.setattr(namespace_admin, "CURATE_ROWS_MAX", 2, raising=False)
    with create_backend_from_config(served, OLD) as store:
        assert isinstance(store, SQLiteBackend)
        _seed(store, OLD, ["M-1", "M-2", "M-3"])
        _edge(store, OLD, "M-1", "M-3")
        _edge(store, OLD, "M-2", "M-3")

    reply = await _serve("rename")

    assert (reply["status"], reply["moved"], reply.get("complete")) == ("renamed", 3, True), reply
    with create_backend_from_config(served, NEW) as store:
        assert sorted((e.source_id, e.target_id) for e in store.graph_edges(NEW)) == [("M-1", "M-3"), ("M-2", "M-3")]


async def test_a_row_written_behind_the_cursor_between_batches_is_moved_by_a_second_pass(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane is free between batches, so a tenant can store a source row the scan already passed (the
    newest sorts first); the drained-source check scans again instead of stranding or refusing it."""
    from trw_memory.daemon import _lane

    monkeypatch.setattr(_lane, "CALL_SECONDS", 0.0)
    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, ["M-1", "M-2", "M-3", "M-4"])
    first = await _serve("merge")
    assert (first["status"], first["moved"]) == ("moving", 2), first
    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, ["M-late"])

    replies = [first]
    while replies[-1].get("complete") is False and len(replies) < 10:
        replies.append(await _serve("merge"))

    assert (replies[-1]["status"], replies[-1]["moved"], replies[-1]["skipped"]) == ("merged", 5, 0), replies
    assert _ids(served, NEW) == ["M-1", "M-2", "M-3", "M-4", "M-late"]
    assert _ids(served, OLD) == []


def test_a_longer_sibling_slug_sharing_the_prefix_is_not_reported(tmp_path: Path) -> None:
    """B71-124 M3: ``project:trw-framework-extra-*`` is a DIFFERENT project (slug
    ``trw-framework-extra``), not a moved copy of ``trw-framework``, even though its
    namespace string starts with the same ``project:trw-framework-`` prefix.
    """
    config = _census_config(tmp_path)
    _seed_via_config(config, "project:trw-framework-extra-44444444", ["M-1"])

    assert detect_moved_checkout(NEW, store_census(config)) is None


def test_a_true_same_slug_sibling_is_still_reported(tmp_path: Path) -> None:
    """The same-prefix guard above must not swallow the real moved-checkout case."""
    config = _census_config(tmp_path)
    _seed_via_config(config, OLD, ["M-1"])

    observation = detect_moved_checkout(NEW, store_census(config))

    assert observation is not None
    assert [candidate.namespace for candidate in observation.candidates] == [OLD]


@pytest.mark.parametrize("verb", ["merge", "rename"])
async def test_a_half_moved_source_resumes_after_the_daemon_forgets_it_in_memory(
    served: MemoryConfig, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """CORE-331 FR03 (CORE-307 audit row f): _MOVES is in-memory only, so a daemon restart used to
    forget a half-done batch -- a resumed merge re-counted from zero (understating moved/source_rows
    in its final reply) and a resumed rename was refused, pointing at merge, because its own
    drained-source check reads progress.moved to tell an already-moved row from a foreign one and
    that counter reset to 0. Dropping the dict entry (simulating the restart) must do neither: the
    next call reloads the persisted progress and resumes exactly where the dropped-from-memory batch
    left off."""
    from trw_memory.daemon import _lane
    from trw_memory.tools import namespace_admin

    monkeypatch.setattr(_lane, "CALL_SECONDS", 0.0)
    source_ids = [f"M-{index}" for index in range(5)]
    with create_backend_from_config(served, OLD) as store:
        _seed(store, OLD, source_ids)
    if verb == "merge":
        with create_backend_from_config(served, NEW) as store:
            store.store(make_entry(entry_id="M-2", namespace=NEW, content="the destination's version"))

    first = await _serve(verb)
    assert first["status"] == "moving", first
    namespace_admin._MOVES.clear()  # the daemon-restart simulation: in-memory progress is gone

    replies = [first]
    while replies[-1].get("complete") is False:
        assert replies[-1]["status"] == "moving", replies[-1]
        assert len(replies) < 10, replies
        replies.append(await _serve(verb))

    final = replies[-1]
    assert final["status"] == ("merged" if verb == "merge" else "renamed"), final
    expected = (5, 4, 1) if verb == "merge" else (5, 5, 0)
    assert (final["source_rows"], final["moved"], final["skipped"]) == expected, final
    assert _ids(served, NEW) == source_ids
    assert _ids(served, OLD) == (["M-2"] if verb == "merge" else [])
