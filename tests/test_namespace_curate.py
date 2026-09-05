"""PRD-CORE-253 FR05 curate verbs + FR01 moved-checkout detection.

Every test drives a real ``SQLiteBackend`` on disk: a bulk re-key is exactly
where rows go missing, so the assertions are on what the store actually holds
afterwards rather than on what the function reported.
"""

from __future__ import annotations

import hashlib
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
    like success.
    """
    config = MemoryConfig(storage_path=str(tmp_path / "store"), storage_backend="yaml")
    with create_backend_from_config(config, OLD) as backend:
        _seed(backend, OLD, ["M-1", "M-2"])

    renamed = memory_namespace_rename_impl(OLD, NEW, config=config)

    assert (renamed["status"], renamed["moved"]) == ("renamed", 2)
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
