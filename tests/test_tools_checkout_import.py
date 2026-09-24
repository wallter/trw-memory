"""PRD-CORE-280 FR03 -- a project store moves into the daemon's namespace through the daemon.

``memory migrate --to user`` hands the daemon a working copy of the checkout's
project store; ``memory_import_checkout`` folds its ``default`` rows, vectors and
edges into the granted project namespace (the destination wins an id collision,
and a rerun moves nothing twice) and counts what it holds of the migrated ids.
The daemon reads only files inside the checkout its token was minted for.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests.conftest import make_entry
from trw_memory._graph_primitives import _upsert_edge
from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespaces.curate import NamespaceStores
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.checkout_import import memory_import_checkout_impl

_NS = "project:acme-1a2b3c4d"
_DIM = 4
_AT = datetime(2026, 9, 23, tzinfo=timezone.utc)


def _row(entry_id: str, namespace: str) -> MemoryEntry:
    """The checkout's row *entry_id*: fixed times, so a copy made later is still the same row."""
    entry = make_entry(
        entry_id=entry_id, namespace=namespace, content=f"row {entry_id}", created_at=_AT, last_accessed_at=_AT
    )
    return entry.model_copy(update={"updated_at": _AT, "valid_from": _AT})


def _project_store(path: Path, ids: list[str], *, edge: tuple[str, str] | None = None, vectors: bool = True) -> None:
    store = SQLiteBackend(path, dim=_DIM)
    try:
        for index, entry_id in enumerate(ids):
            store.store(_row(entry_id, "default"))
            if not vectors:
                continue
            vector = [0.0] * _DIM
            vector[index % _DIM] = 1.0
            store.upsert_vector(entry_id, vector, namespace="default")
        if edge is not None:
            with store._lock:
                _upsert_edge(store._conn, *edge, "related_to", 0.5, "2026-09-23T00:00:00+00:00", namespace="default")
                store._conn.commit()
    finally:
        store.close()


def _identical_copy(store: SQLiteBackend, entry_id: str, index: int = 0) -> None:
    """What the copy's *entry_id* becomes once imported: the same row and vector in ``_NS``."""
    store.store(_row(entry_id, _NS))
    vector = [0.0] * _DIM
    vector[index % _DIM] = 1.0
    store.upsert_vector(entry_id, vector, namespace=_NS)


@pytest.fixture
def user_store(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "user.db", dim=_DIM)
    if not store.vec_available:
        pytest.skip("sqlite-vec unavailable")
    yield store
    store.close()


def test_an_import_moves_rows_vectors_and_edges_and_a_rerun_moves_nothing(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    ids = ["L-1", "L-2", "L-3"]
    work = tmp_path / "work.db"
    _project_store(work, ids, edge=("L-1", "L-2"))

    user_store.store(make_entry(entry_id="L-other", namespace=_NS))
    first = memory_import_checkout_impl(_NS, str(work), ids, backend=user_store)
    _project_store(work, ids, edge=("L-1", "L-2"))  # a fresh working copy, as a rerun takes
    again = memory_import_checkout_impl(_NS, str(work), [*ids, "L-missing"], backend=user_store)

    assert (first["moved"], first["skipped"], first["held"]) == (3, 0, {"rows": 3, "vectors": 3, "edges": 1})
    assert (again["moved"], again["skipped"], again["held"]) == (0, 3, {"rows": 3, "vectors": 3, "edges": 1})
    assert user_store.count(namespace=_NS) == 4, "no duplicate rows"
    assert user_store.existing_vector_ids(namespace=_NS) == set(ids)
    assert [(e.source_id, e.target_id) for e in user_store.graph_edges(_NS)] == [("L-1", "L-2")]


def test_a_rerun_over_rows_the_checkout_had_synced_moves_nothing_and_refuses_nothing(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """The store clears a row's sync mark when it takes it, so a retry after a lost reply compares equal."""
    ids = ["L-1", "L-2"]
    work = tmp_path / "work.db"

    def synced_copy() -> None:
        _project_store(work, ids)
        store = SQLiteBackend(work, dim=_DIM)
        try:
            for entry_id in ids:
                store.update(entry_id, namespace="default", last_synced_at=_AT, sync_hash="stale")
        finally:
            store.close()

    synced_copy()
    first = memory_import_checkout_impl(_NS, str(work), ids, backend=user_store)
    synced_copy()  # the retry takes a fresh working copy of the same checkout
    again = memory_import_checkout_impl(_NS, str(work), ids, backend=user_store)

    assert (first["status"], first["moved"]) == ("ok", 2)
    assert (again["status"], again["moved"], again["skipped"]) == ("ok", 0, 2)


def test_an_id_held_with_other_content_or_vector_is_refused_and_named(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """The destination wins a collision, so a differing source row is never imported: refuse (FR03)."""
    ids = ["L-1", "L-2", "L-3"]
    work = tmp_path / "work.db"
    _project_store(work, ids)
    user_store.store(make_entry(entry_id="L-2", namespace=_NS, content="another learning"))
    user_store.upsert_vector("L-2", [0.0, 1.0, 0.0, 0.0], namespace=_NS)
    _identical_copy(user_store, "L-3", 2)
    user_store.upsert_vector("L-3", [0.0, 0.0, 0.0, 1.0], namespace=_NS)  # only the vector differs

    answer = memory_import_checkout_impl(_NS, str(work), ids, backend=user_store)

    assert answer["status"] == "conflict"
    assert answer["conflicts"] == ["L-2", "L-3"]
    assert "L-2" in str(answer["error"])


def test_a_refused_import_copies_nothing_into_the_namespace(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """The collisions are compared BEFORE any copy, so a refusal leaves the namespace as it was."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-stray"])
    user_store.store(make_entry(entry_id="L-1", namespace=_NS, content="another learning"))
    before = user_store.count(namespace=_NS)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-stray"], backend=user_store)

    assert (answer["status"], answer.get("conflicts")) == ("conflict", ["L-1"])
    assert user_store.count(namespace=_NS) == before
    assert user_store.get("L-stray", namespace=_NS) is None
    assert user_store.existing_vector_ids(namespace=_NS) == set()


def test_an_obsolete_colliding_row_is_compared_too(tmp_path: Path, user_store: SQLiteBackend) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    source = SQLiteBackend(work, dim=_DIM)
    source.update("L-1", namespace="default", status=MemoryStatus.OBSOLETE)
    source.close()
    user_store.store(make_entry(entry_id="L-1", namespace=_NS, content="another learning"))

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (answer["status"], answer.get("conflicts")) == ("conflict", ["L-1"])


def test_a_copy_it_cannot_list_back_whole_is_refused(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2"])
    monkeypatch.setattr(SQLiteBackend, "count", lambda self, namespace=None: 3)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2"], backend=user_store)

    assert answer["status"] == "conflict"
    assert "could not list every row" in str(answer["error"])


def test_a_merge_that_skips_other_rows_than_it_compared_is_refused(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _identical_copy(user_store, "L-1")  # no conflict
    from trw_memory.tools import checkout_import

    real = checkout_import.merge_namespace

    def _overcounted(*args: object) -> object:
        result = real(*args)  # type: ignore[arg-type]
        return result.model_copy(update={"skipped": result.skipped + 1})

    monkeypatch.setattr(checkout_import, "merge_namespace", _overcounted)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert answer["status"] == "conflict"
    assert "skipped 2 rows, not the 1 compared" in str(answer["error"])


def _merge_after(monkeypatch: pytest.MonkeyPatch, between: Callable[[SQLiteBackend], None]) -> None:
    """Run *between* on the destination after the compare, just before the merge."""
    from trw_memory.tools import checkout_import

    real = checkout_import.merge_namespace

    def _interleaved(stores: NamespaceStores, source: str, destination: str) -> object:
        between(stores.destination)  # type: ignore[arg-type]
        return real(stores, source, destination)

    monkeypatch.setattr(checkout_import, "merge_namespace", _interleaved)


def test_no_other_writer_can_change_a_compared_row_before_the_merge(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compare and the merge share the destination's transaction: a second connection waits it out."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _identical_copy(user_store, "L-1")
    refused: list[str] = []

    def _concurrent_edit(_destination: SQLiteBackend) -> None:
        with contextlib.closing(sqlite3.connect(tmp_path / "user.db", timeout=0)) as other:
            try:
                other.execute("UPDATE memories SET content = 'changed' WHERE id = 'L-1'")
                other.commit()
            except sqlite3.OperationalError as exc:
                refused.append(str(exc))

    _merge_after(monkeypatch, _concurrent_edit)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert answer["status"] == "ok"
    assert refused and "locked" in refused[0]
    held = user_store.get("L-1", namespace=_NS)
    assert held is not None and held.content == "row L-1"


def test_a_collision_that_appears_after_the_compare_rolls_the_whole_import_back(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row landing between the compare and the merge is skipped unchecked: refuse, and change nothing."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2", "L-3"])
    _identical_copy(user_store, "L-1")
    before = {entry.id for entry in user_store.list_entries(namespace=_NS, limit=10)}
    _merge_after(monkeypatch, lambda destination: _identical_copy(destination, "L-2", 1))

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2", "L-3"], backend=user_store)

    assert answer["status"] == "conflict"
    assert "skipped 2 rows, not the 1 compared" in str(answer["error"])
    assert {entry.id for entry in user_store.list_entries(namespace=_NS, limit=10)} == before
    assert user_store.existing_vector_ids(namespace=_NS) == {"L-1"}
    copy = SQLiteBackend(work, dim=_DIM)
    try:
        assert copy.count(namespace="default") == 3, "the copy keeps every row it did not hand over"
    finally:
        copy.close()


def test_a_vector_it_cannot_read_refuses_the_import(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unread vector is not an absent one: the collision cannot be compared, so refuse and copy nothing."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2"])
    _identical_copy(user_store, "L-1")
    user_store.upsert_vector("L-1", [0.0, 0.0, 0.0, 1.0], namespace=_NS)  # a differing vector

    def _io_error(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("trw_memory.storage._vector_provenance_reads.iter_bind_chunks", _io_error)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2"], backend=user_store)

    assert answer["status"] == "conflict"
    assert "could not read the vectors to compare" in str(answer["error"])
    assert user_store.get("L-2", namespace=_NS) is None


def test_a_strict_vector_read_raises_where_the_lenient_one_returns_none(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "s.db", dim=_DIM)
    try:
        if not store.vec_available:
            pytest.skip("sqlite-vec unavailable")
        store.store(make_entry(entry_id="L-1", namespace="default"))
        store.upsert_vector("L-1", [1.0, 0.0, 0.0, 0.0], namespace="default")
        with store._lock:
            store._conn.execute("DROP TABLE vec_index")
            store._conn.commit()

        assert store.get_vector_records(["L-1"], namespace="default") == {}
        with pytest.raises(StorageError):
            store.vector_records_or_raise(["L-1"], namespace="default")
    finally:
        store.close()


@pytest.mark.parametrize(
    "field", [{"status": MemoryStatus.OBSOLETE}, {"metadata": {"k": "v"}}, {"tags": ["other"]}, {"importance": 0.9}]
)
def test_a_collision_differing_only_outside_content_is_refused(
    tmp_path: Path, user_store: SQLiteBackend, field: dict[str, object]
) -> None:
    """The copy's row would be dropped whole, so every field it carries is compared, not just its text."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _identical_copy(user_store, "L-1")
    user_store.update("L-1", namespace=_NS, **field)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (answer["status"], answer.get("conflicts")) == ("conflict", ["L-1"])


def test_a_vector_only_the_namespace_holds_is_a_difference(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """Presence is compared both ways: a copy row without a vector does not match a held row with one."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"], vectors=False)
    _identical_copy(user_store, "L-1")

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (answer["status"], answer.get("conflicts")) == ("conflict", ["L-1"])


def test_an_import_past_its_deadline_rolls_back_and_answers_busy(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """The daemon's write lock is held for a bounded time; past it, nothing lands and the caller retries."""
    ids = [f"L-{index}" for index in range(50)]
    work = tmp_path / "work.db"
    _project_store(work, ids)
    before = user_store.count(namespace=_NS)

    answer = memory_import_checkout_impl(_NS, str(work), ids, backend=user_store, deadline_seconds=0.0)

    assert answer["status"] == "busy"
    assert "retry" in str(answer["error"])
    assert user_store.count(namespace=_NS) == before
    assert user_store.existing_vector_ids(namespace=_NS) == set()
    copy = SQLiteBackend(work, dim=_DIM)
    try:
        assert copy.count(namespace="default") == len(ids), "the copy keeps every row"
    finally:
        copy.close()


def test_a_deadline_interrupts_a_statement_that_runs_past_it(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """The deadline stops the scan itself, not only a check between phases."""
    import time

    from trw_memory.tools.checkout_import import _interrupted_after

    with (
        _interrupted_after(user_store, time.monotonic() - 1),
        pytest.raises(sqlite3.OperationalError, match="interrupt"),
    ):
        user_store._conn.execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1000000) SELECT count(*) FROM n"
        ).fetchone()
    assert tuple(user_store._conn.execute("SELECT 1").fetchone()) == (1,), "the handler is removed afterwards"


def test_a_write_lock_held_elsewhere_answers_busy_within_the_deadline(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """The deadline covers the wait for the write lock, which SQLite spends with no progress handler."""
    import time

    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    before = user_store.count(namespace=_NS)
    with contextlib.closing(sqlite3.connect(tmp_path / "user.db", isolation_level=None)) as other:
        other.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store, deadline_seconds=0.5)
        waited = time.monotonic() - started
        other.execute("ROLLBACK")

    assert answer["status"] == "busy"
    assert waited < 3, f"waited {waited:.1f}s for a 0.5s deadline"
    assert user_store.count(namespace=_NS) == before
    copy = SQLiteBackend(work, dim=_DIM)
    try:
        assert copy.count(namespace="default") == 1
    finally:
        copy.close()


def test_a_copy_commit_that_fails_after_the_namespace_committed_is_uncertain(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination commits first; once it has, the reply never claims a rollback."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    real = SQLiteBackend.transaction

    @contextlib.contextmanager
    def _copy_commit_fails(self: SQLiteBackend) -> Iterator[SQLiteBackend]:
        with real(self) as txn:
            yield txn
            if Path(self._db_path) == work and self._skip_commit_depth == 1:  # the copy's outermost commit
                raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SQLiteBackend, "transaction", _copy_commit_fails)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert answer["status"] == "uncertain"
    assert "rerun" in str(answer["error"])
    assert user_store.get("L-1", namespace=_NS) is not None, "the namespace did commit"


class _Captured:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self) -> object:
        return lambda fn: self.tools.setdefault(fn.__name__, fn)


def _import_over_transport(root: Path, source: Path, opened: list[str], monkeypatch: pytest.MonkeyPatch) -> object:
    from trw_memory.tools.checkout_import import register_checkout_import_tools

    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config",
        lambda _cfg, namespace: opened.append(namespace) or nullcontext(object()),
    )
    server = _Captured()
    register_checkout_import_tools(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        return asyncio.run(server.tools["memory_import_checkout"](namespace=_NS, source_path=str(source), ids=[]))  # type: ignore[operator]
    finally:
        auth_context_var.reset(reset)


def test_a_source_outside_the_granted_checkout_is_refused_before_any_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    answer = _import_over_transport(tmp_path / "repo", tmp_path / "elsewhere" / "memory.db", opened, monkeypatch)

    assert isinstance(answer, dict)
    assert answer["status"] == "refused"
    assert opened == []
