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
import hashlib
import os
import shutil
import sqlite3
import sys
import threading
import time
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
from trw_memory._inode_pin import current_identity
from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespaces.curate import NamespaceStores
from trw_memory.storage import _connection, _schema, _schema_backup
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import _checkout_merge, checkout_import
from trw_memory.tools import entry as entry_module
from trw_memory.tools.checkout_import import _private_checkout_copy, memory_import_checkout_impl

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

    real = _checkout_merge.merge_namespace

    def _overcounted(*args: object) -> object:
        result = real(*args)  # type: ignore[arg-type]
        return result.model_copy(update={"skipped": result.skipped + 1})

    monkeypatch.setattr(_checkout_merge, "merge_namespace", _overcounted)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert answer["status"] == "conflict"
    assert "skipped 2 rows, not the 1 compared" in str(answer["error"])


def _merge_after(monkeypatch: pytest.MonkeyPatch, between: Callable[[SQLiteBackend], None]) -> None:
    """Run *between* on the destination after the compare, just before the merge."""

    real = _checkout_merge.merge_namespace

    def _interleaved(stores: NamespaceStores, source: str, destination: str) -> object:
        between(stores.destination)  # type: ignore[arg-type]
        return real(stores, source, destination)

    monkeypatch.setattr(_checkout_merge, "merge_namespace", _interleaved)


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

    from trw_memory.tools._checkout_merge import _interrupted_after

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


_ENDLESS = "WITH RECURSIVE r(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM r) SELECT x FROM r WHERE x < 0"


@pytest.mark.parametrize(
    ("schema", "named"),
    [
        (f"CREATE VIEW memories AS {_ENDLESS};", "view memories"),
        ("CREATE TABLE t(x); CREATE TRIGGER on_t AFTER INSERT ON t BEGIN SELECT 1; END;", "trigger on_t"),
        ("CREATE VIRTUAL TABLE memories USING fts5(content);", "table memories"),
    ],
)
def test_a_copy_with_schema_trw_memory_never_writes_is_refused_before_any_backend_reads_it(
    tmp_path: Path, user_store: SQLiteBackend, schema: str, named: str
) -> None:
    """rc8 (B71-74): a pre-migration copy whose ``memories`` is an endless view hung the backend's open."""
    hostile = tmp_path / "hostile.db"
    conn = sqlite3.connect(hostile)
    conn.executescript(schema)
    conn.execute("PRAGMA user_version = 7")  # takes the migration route, which reads memories
    conn.commit()
    conn.close()
    outcome: dict[str, object] = {}

    def run() -> None:
        outcome.update(memory_import_checkout_impl(_NS, str(hostile), [], backend=user_store, deadline_seconds=2.0))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "the import never returned"
    assert outcome["status"] == "invalid"
    assert named in str(outcome["error"])


def _import_within(source: Path, user_store: SQLiteBackend, ids: list[str], seconds: float) -> dict[str, object]:
    outcome: dict[str, object] = {}
    worker = threading.Thread(
        target=lambda: outcome.update(
            memory_import_checkout_impl(_NS, str(source), ids, backend=user_store, deadline_seconds=seconds)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "the import never returned"
    return outcome


def test_the_backend_open_of_an_admitted_copy_runs_under_the_import_deadline(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sol, rc8 delta: a v7 copy passes the schema check, then its v8 migration ran with no deadline."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    with contextlib.closing(sqlite3.connect(work)) as conn:
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    monkeypatch.setitem(_schema._MIGRATIONS, 8, lambda cursor: cursor.execute(_ENDLESS).fetchall())

    started = time.monotonic()
    assert _import_within(work, user_store, ["L-1"], 1.0)["status"] == "busy"
    assert time.monotonic() - started < 10
    assert user_store.get("L-1", namespace=_NS) is None


def test_the_migration_snapshot_of_an_admitted_copy_stops_at_the_import_deadline(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sol round 2: backup() runs no progress handler, so the snapshot copied past the deadline."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    with contextlib.closing(sqlite3.connect(work)) as conn:
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    opened = _schema_backup._open_snapshot_source
    monkeypatch.setattr(_schema_backup, "_open_snapshot_source", lambda path: (time.sleep(1.5), opened(path))[1])
    migrated: list[int] = []
    monkeypatch.setitem(_schema._MIGRATIONS, 8, lambda _cursor: migrated.append(8))

    assert _import_within(work, user_store, ["L-1"], 1.0)["status"] == "busy"
    assert migrated == [], "the migration ran after the deadline had passed"


def test_a_copy_the_backend_cannot_migrate_is_invalid_and_leaves_no_verified_mark(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """sol, rc8 delta: the backend's open was outside the invalid mapping, and every copy stayed marked."""
    broken, work = tmp_path / "broken.db", tmp_path / "work.db"
    _project_store(broken, ["L-2"])
    with contextlib.closing(sqlite3.connect(broken)) as conn:
        conn.execute("PRAGMA user_version = 99")  # written by a newer build: the backend's open refuses it
        conn.commit()
    _project_store(work, ["L-1"])

    assert _import_within(broken, user_store, [], 5.0)["status"] == "invalid"
    assert _import_within(work, user_store, ["L-1"], 5.0)["status"] == "ok"
    assert not _connection._VERIFIED_STORES & {_connection._store_identity(broken), _connection._store_identity(work)}


@pytest.mark.parametrize(
    "script",
    [
        "CREATE TABLE memories(id TEXT PRIMARY KEY); PRAGMA user_version = 7;",  # migrated, but keeps its shape
        "CREATE TABLE unrelated(x); PRAGMA user_version = 8;",  # stamped current: no migration creates memories
    ],
)
def test_a_copy_this_build_cannot_read_is_invalid_not_a_tool_error(
    tmp_path: Path, user_store: SQLiteBackend, script: str
) -> None:
    """C9 (B71-74 residual): the backend admitted the copy, then the merge's first read raised out of the tool."""
    thin = tmp_path / "thin.db"
    with contextlib.closing(sqlite3.connect(thin)) as conn:
        conn.executescript(script)

    answer = _import_within(thin, user_store, [], 5.0)

    assert answer["status"] == "invalid"
    assert "not a project store" in str(answer["error"])


def test_a_copy_row_this_build_cannot_parse_gets_a_structured_reply(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """C9: the reader skips a row it cannot parse, so the copy does not list back whole: ``conflict``."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    with contextlib.closing(sqlite3.connect(work)) as conn:
        conn.execute("UPDATE memories SET importance = 'high'")
        conn.commit()

    assert _import_within(work, user_store, ["L-1"], 5.0)["status"] == "conflict"
    assert user_store.get("L-1", namespace=_NS) is None


def test_a_copy_edge_the_namespace_cannot_hold_refuses_the_import(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """sol, C9 round 2: an edge breaking the weight CHECK was dropped by INSERT OR IGNORE, and the import said ok."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2"], edge=("L-1", "L-2"))
    with contextlib.closing(sqlite3.connect(work)) as conn:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'memory_graph_edges'").fetchone()[0]
        conn.executescript(
            "ALTER TABLE memory_graph_edges RENAME TO old_edges;"
            + ddl.replace("CHECK (weight >= 0.0 AND weight <= 1.0)", "")
            + "; INSERT INTO memory_graph_edges SELECT * FROM old_edges; DROP TABLE old_edges;"
            "UPDATE memory_graph_edges SET weight = 5.0;"
        )

    answer = _import_within(work, user_store, ["L-1", "L-2"], 5.0)

    assert answer["status"] == "invalid"
    assert "('L-1', 'L-2', 'related_to')" in str(answer["error"])
    assert user_store.get("L-1", namespace=_NS) is None, "the refusal rolled the merge back"
    with contextlib.closing(sqlite3.connect(work)) as conn:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 2, "and the copy's deletes"


def test_a_copy_vector_of_another_dimension_is_reported_not_silently_dropped(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """sol, C9 round 3: the namespace's vector write skips a wrong-length vector, and the import said only ok.

    The row moves (trw_assess: carry and report, 1.0); the reply names the vector for ``memory_reembed``.
    """
    work = tmp_path / "work.db"
    wide = SQLiteBackend(work, dim=_DIM * 2)
    try:
        wide.store(_row("L-1", "default"))
        wide.upsert_vector("L-1", [1.0] + [0.0] * (_DIM * 2 - 1), namespace="default")
    finally:
        wide.close()

    answer = _import_within(work, user_store, ["L-1"], 5.0)

    assert (answer["status"], answer.get("vectors_not_carried")) == ("ok", ["L-1"])
    assert user_store.get("L-1", namespace=_NS) is not None
    assert not user_store.existing_vector_ids(namespace=_NS)


def test_a_retry_after_vectors_were_not_carried_converges_not_conflicts(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """C12 pre-rc9 P1: the first import moved the row but not its other-dimension vector; a retry after a
    lost reply (or a crash before the checkout pin) compared that vector with the namespace's absent one
    and answered conflict, stranding ``memory migrate --apply``. A vector the merge would not carry is not
    compared either."""

    def wide_copy(name: str) -> Path:  # each attempt imports a fresh copy of the checkout's store
        path = tmp_path / name
        wide = SQLiteBackend(path, dim=_DIM * 2)
        try:
            wide.store(_row("L-1", "default"))
            wide.upsert_vector("L-1", [1.0] + [0.0] * (_DIM * 2 - 1), namespace="default")
        finally:
            wide.close()
        return path

    first = _import_within(wide_copy("first.db"), user_store, ["L-1"], 5.0)
    assert (first["status"], first.get("vectors_not_carried")) == ("ok", ["L-1"])

    # The reply was lost (or the caller crashed before the pin): migrate runs it again.
    retry = _import_within(wide_copy("retry.db"), user_store, ["L-1"], 5.0)

    assert retry["status"] == "ok", retry
    assert (retry["moved"], retry["skipped"], retry.get("vectors_not_carried")) == (0, 1, ["L-1"])

    # The rule skips only the vector: a namespace row that changed since is still a conflict.
    user_store.update("L-1", namespace=_NS, content="edited after the first import")
    edited = _import_within(wide_copy("edited.db"), user_store, ["L-1"], 5.0)
    assert (edited["status"], edited.get("conflicts")) == ("conflict", ["L-1"])


def test_an_identical_row_and_embedding_with_other_provenance_is_a_conflict(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """sol, C9 round 3: collisions compared the embedding alone, so the copy's provenance was dropped as a skip."""
    from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance, input_digest, vector_digest

    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    vector = [1.0] + [0.0] * (_DIM - 1)
    user_store.store(_row("L-1", _NS))
    space = EmbeddingSpace("a" * 64, "test-encoder:a", _DIM)
    provenance = VectorProvenance(space, input_digest("row L-1"), vector_digest(vector))
    user_store.upsert_vector("L-1", vector, namespace=_NS, provenance=provenance)

    answer = _import_within(work, user_store, ["L-1"], 5.0)

    assert (answer["status"], answer.get("conflicts")) == ("conflict", ["L-1"])


def test_another_thread_holding_the_store_makes_the_import_busy_at_its_deadline(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """sol, C9 round 1: the in-process lock was waited for with no timeout."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    release = threading.Event()

    def hold() -> None:
        with user_store._lock:
            release.wait(20)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    time.sleep(0.1)
    started = time.monotonic()
    try:
        assert _import_within(work, user_store, ["L-1"], 1.0)["status"] == "busy"
        assert time.monotonic() - started < 10
    finally:
        release.set()
        holder.join(5)


def _import_over_transport(
    root: Path,
    source: Path,
    opened: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: object | None = None,
    ids: list[str] | None = None,
) -> object:
    from trw_memory.tools.checkout_import import register_checkout_import_tools

    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config",
        lambda _cfg, namespace: opened.append(namespace) or nullcontext(object() if backend is None else backend),
    )
    server = _Captured()
    register_checkout_import_tools(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        call = server.tools["memory_import_checkout"](namespace=_NS, source_path=str(source), ids=ids or [])  # type: ignore[operator]
        return asyncio.run(call)
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


def test_over_transport_the_registered_tool_opens_a_private_copy_and_cleans_it_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper -- not just the helper -- routes through ``_private_checkout_copy`` and removes it after."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "memory.db").write_bytes(b"legitimate inside bytes")
    seen_source_paths: list[str] = []
    real_impl = checkout_import.memory_import_checkout_impl

    def _spy(namespace: str, source_path: str, ids: list[str], *, backend: object, lane: object = None) -> object:
        seen_source_paths.append(source_path)
        return real_impl(namespace, source_path, ids, backend=backend)  # type: ignore[arg-type]

    monkeypatch.setattr(checkout_import, "memory_import_checkout_impl", _spy)
    opened: list[str] = []

    answer = _import_over_transport(root, root / "memory.db", opened, monkeypatch)

    assert opened == [_NS], "the wrapper reached backend creation: the private copy opened cleanly"
    assert len(seen_source_paths) == 1
    private_copy = seen_source_paths[0]
    assert private_copy != str(root / "memory.db"), "the impl never sees the caller's own path"
    assert "import-tmp" in private_copy
    assert not Path(private_copy).parent.exists(), "the private copy's directory is removed once the call returns"
    assert isinstance(answer, dict)


def _import_tmp_entries() -> list[Path]:
    from trw_memory.daemon._paths import DaemonPaths

    import_tmp = DaemonPaths.resolve(create=True).user_memory_dir / "import-tmp"
    return sorted(import_tmp.rglob("*")) if import_tmp.exists() else []


def test_imports_of_a_pre_migration_store_leave_nothing_in_import_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_store: SQLiteBackend
) -> None:
    """C12-R: opening a populated pre-v8 copy writes a schema backup beside it; no import may keep one.

    The copy is the only thing the daemon migrates, and the caller's own file is the real backup,
    so the backup, the copy and anything else SQLite made beside it go when the import returns:
    after an import, after an idempotent rerun, and after a refusal alike.
    """
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "memory.db"
    _project_store(source, ["L-1", "L-2"])
    with contextlib.closing(sqlite3.connect(source)) as conn:
        conn.execute("PRAGMA user_version = 7")  # pre-v8: opening it takes a pre-schema-8 snapshot
        conn.commit()
    before = source.read_bytes()

    def run() -> dict[str, object]:
        answer = _import_over_transport(root, source, [], monkeypatch, backend=user_store, ids=["L-1", "L-2"])
        assert isinstance(answer, dict)
        assert _import_tmp_entries() == [], answer
        return answer

    first, again = run(), run()
    user_store.update("L-1", namespace=_NS, content="changed in the namespace")
    refused = run()

    assert (first["status"], first["moved"], again["status"], again["moved"]) == ("ok", 2, "ok", 0)
    assert refused["status"] == "conflict", refused
    assert source.read_bytes() == before, "the caller's own store is never touched"


def test_a_private_copy_swapped_during_use_is_reported_uncertain_not_silently_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-SEC-016 round-2 finding 3: identity is checked again, by path, right after the SQLite open too.

    Simulates a principal replacing the private copy's file (same path, a
    DIFFERENT inode) DURING ``memory_import_checkout_impl``'s own execution
    -- exactly the residual window the daemon's own store open already
    accepts in writing (G4), narrowed the same way here: the reply names it
    ``uncertain`` (rerun; the import is idempotent) rather than silently
    reporting the swapped-in content's import as a clean success.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "memory.db").write_bytes(b"legitimate inside bytes")
    real_impl = checkout_import.memory_import_checkout_impl

    def _call_then_swap(
        namespace: str, source_path: str, ids: list[str], *, backend: object, lane: object = None
    ) -> object:
        result = real_impl(namespace, source_path, ids, backend=backend)  # type: ignore[arg-type]
        # AFTER the impl call returns (its own answer is not this test's
        # concern -- the harness's fake backend makes it "invalid" either
        # way), something replaces the file at the same path before the
        # wrapper's own post-use identity check runs.
        Path(source_path).unlink()
        Path(source_path).write_bytes(b"swapped content")
        return result

    monkeypatch.setattr(checkout_import, "memory_import_checkout_impl", _call_then_swap)
    opened: list[str] = []

    answer = _import_over_transport(root, root / "memory.db", opened, monkeypatch)

    assert isinstance(answer, dict) and answer["status"] == "uncertain"
    assert "replaced" in str(answer["error"])


def test_a_private_copy_unchanged_through_use_reports_the_real_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression control: the identity checks must not false-positive on the ordinary, unraced path."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "memory.db").write_bytes(b"legitimate inside bytes")

    answer = _import_over_transport(root, root / "memory.db", [], monkeypatch)

    assert isinstance(answer, dict)
    assert answer.get("status") != "uncertain"
    assert answer.get("status") != "refused"


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
def test_a_concurrent_swap_through_the_served_import_never_reaches_the_outside_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-SEC-016 round-4 finding 4: an end-to-end race through the REAL served tool, not the internal helper.

    ``TestSymlinkSwapRace`` calls ``_private_checkout_copy`` directly, which
    proves the opener's own walk is safe but never exercises the served
    ``memory_import_checkout`` tool as a whole -- namespace authorization, a
    REAL destination backend, and the actual merge. This test races the
    SAME symlink swap across the whole served call, with a real SQLite
    destination namespace, and checks both ends: the destination namespace
    never holds the outside store's row, and the outside store's own bytes
    never change.
    """
    from trw_memory.integrations._backend import create_backend_from_config
    from trw_memory.tools.checkout_import import register_checkout_import_tools

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "dest-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    root = tmp_path / "repo"
    real_dir = tmp_path / "x-real-backup"
    (root / "x" / "y").mkdir(parents=True)
    _project_store(root / "x" / "y" / "memory.db", ["L-1"])
    source_path = str(root / "x" / "y" / "memory.db")

    outside_root = tmp_path / "outside"
    (outside_root / "y").mkdir(parents=True)
    redirected_leaf = outside_root / "y" / "memory.db"  # exactly where root/x/y/memory.db resolves once swapped
    _project_store(redirected_leaf, ["L-EVIL"])  # a distinct id: if this ever merged, it would be unmistakable
    sha_before = hashlib.sha256(redirected_leaf.read_bytes()).hexdigest()

    server = _Captured()
    register_checkout_import_tools(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))

    stop = threading.Event()

    def flipper() -> None:
        while not stop.is_set():
            x = root / "x"
            try:
                if x.is_symlink():
                    x.unlink()
                    shutil.move(str(real_dir), str(x))
                elif x.is_dir():
                    shutil.move(str(x), str(real_dir))
                    x.symlink_to(outside_root)
            except OSError:  # trw-fail-silent-allow: a lost race against the main thread's own call is expected on every iteration; the test asserts the OUTCOME, not that every flip lands
                pass

    thread = threading.Thread(target=flipper, daemon=True)
    thread.start()
    outcomes: list[object] = []
    try:
        # At least 20 racing attempts, and on until one lands: each attempt wins
        # the race only now and then, so a fixed count failed ~10% of runs with
        # every attempt refused (measured 2026-09-24, 7/70 on int-700).
        for attempt in range(200):
            answer = asyncio.run(
                server.tools["memory_import_checkout"](namespace=_NS, source_path=source_path, ids=["L-1"])  # type: ignore[operator]
            )
            outcomes.append(answer)
            if attempt >= 19 and any(isinstance(o, dict) and o.get("status") == "ok" for o in outcomes):
                break
    finally:
        stop.set()
        thread.join(timeout=5)
        auth_context_var.reset(reset)

    assert hashlib.sha256(redirected_leaf.read_bytes()).hexdigest() == sha_before, "the outside store was written to"
    for answer in outcomes:
        assert isinstance(answer, dict)
        assert "L-EVIL" not in str(answer)

    with create_backend_from_config(MemoryConfig(), _NS) as destination:
        assert destination.get("L-EVIL", namespace=_NS) is None, "the outside store's row must never land"
        held = destination.get("L-1", namespace=_NS)
        assert held is not None and held.content == "row L-1", "the legitimate row must still have moved"


def test_a_path_swapped_while_it_is_being_resolved_is_refused_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPython's non-strict realpath lstat()s a symlink and then readlink()s it unguarded, so a
    symlink removed in between raises FileNotFoundError out of Path.resolve(); the served tool
    answers "refused" instead of crashing (seen on Linux under the concurrent-swap race above)."""
    root = tmp_path / "repo"
    root.mkdir()
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))

    def _vanished(self: Path, strict: bool = False) -> Path:
        raise FileNotFoundError(2, "No such file or directory", str(root / "x"))

    monkeypatch.setattr(Path, "resolve", _vanished)
    try:
        answer = checkout_import.checkout_path(str(root / "x" / "memory.db"), "memory_import_checkout", within=True)
    finally:
        auth_context_var.reset(reset)
    assert isinstance(answer, dict) and answer["status"] == "refused"
    assert "changed while it was being resolved" in str(answer["error"])


@pytest.mark.slow
def test_the_no_follow_copy_fits_inside_the_import_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PRD-SEC-016 NFR01 -- the evidence artifact this AC names.

    "A 20,000-row working copy imports inside IMPORT_DEADLINE_SECONDS...
    copy included. The reply reports the copy's wall time" -- read against
    ``IMPORT_DEADLINE_SECONDS``'s own existing (pre-PRD-SEC-016, from
    PRD-CORE-280) role: it bounds the WHOLE transaction, copy AND the
    destination compare-and-merge together, and already answers ``busy``
    (a rollback, not a failure) if that whole window is exceeded. What
    PRD-SEC-016's FR02 ADDED to that existing, already-deadline-gated
    transaction is the no-follow byte copy step itself -- this test measures
    THAT addition in isolation, which is what NFR01's own evidence-artifact
    name (``test_the_no_follow_copy_fits_inside_the_import_deadline``) says
    it verifies, and what the reply's new ``copy_seconds`` field (added
    alongside this test) reports.

    Measured finding, recorded rather than silently worked around: a fully
    end-to-end 20k-row run (copy AND merge, through the real served tool)
    on this dev Mac exceeds ``IMPORT_DEADLINE_SECONDS`` and rolls back with
    ``status: busy`` -- profiling attributes essentially all of that time to
    ``namespaces.curate.merge_namespace``'s existing row-by-row delete+store
    loop (~26s of the ~30s total, ~186k individual `sqlite3.Connection.execute`
    calls for ~14k row-pairs), a PRE-EXISTING characteristic of the merge
    engine from PRD-CORE-280, wholly unrelated to the checkout-boundary/
    symlink defenses this PRD is about, and out of scope to change here
    (`namespaces/curate.py` is not a PRD-SEC-016 file, and a concurrent
    session is independently debugging adjacent storage-open-path code).
    The FR02 copy step this test actually measures is comfortably inside
    budget: ~0.01s for a 20k-row/~15MB working copy.
    """
    from trw_memory.tools.checkout_import import IMPORT_DEADLINE_SECONDS, _private_checkout_copy

    root = tmp_path / "repo"
    root.mkdir()
    work = root / "work.db"
    row_count = 20_000
    store = SQLiteBackend(work, dim=_DIM)
    try:
        with store.transaction():
            for i in range(row_count):
                entry_id = f"L-{i:06d}"
                store.store(
                    make_entry(
                        entry_id=entry_id,
                        namespace="default",
                        content=f"row {entry_id}",
                        created_at=_AT,
                        last_accessed_at=_AT,
                    )
                )
    finally:
        store.close()
    work_size_mb = work.stat().st_size / 1_000_000

    started = time.monotonic()
    result = _private_checkout_copy(str(root), str(work), "memory_import_checkout")
    wall_seconds = time.monotonic() - started

    assert not isinstance(result, dict), result
    try:
        assert wall_seconds < IMPORT_DEADLINE_SECONDS, (
            f"the no-follow copy of a {work_size_mb:.1f}MB/{row_count}-row store took "
            f"{wall_seconds:.3f}s (budget {IMPORT_DEADLINE_SECONDS}s)"
        )
        assert result.copy_seconds < IMPORT_DEADLINE_SECONDS
        # The copy is a byte-for-byte duplicate: same row count when reopened.
        copy_backend = SQLiteBackend(Path(result.path), dim=_DIM)
        try:
            assert copy_backend.count(namespace="default") == row_count
        finally:
            copy_backend.close()
    finally:
        result.pin.close()
        Path(result.path).unlink(missing_ok=True)
        Path(result.path).parent.rmdir()


@pytest.mark.slow
def test_a_served_20k_row_import_reports_copy_seconds_in_the_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reply carries ``copy_seconds`` end to end through the SERVED tool, not just the internal helper.

    Uses a small enough row count (the merge, not the copy, is what scales
    poorly per the finding above) that the whole served call completes
    inside the deadline, so this specifically exercises the wrapper's own
    ``result["copy_seconds"] = copied.copy_seconds`` wiring against a real
    destination backend -- not a re-measurement of the 20k-row copy itself
    (covered directly, without the slower merge, by the test above).
    """
    from trw_memory.tools.checkout_import import IMPORT_DEADLINE_SECONDS, register_checkout_import_tools

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "dest-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    root = tmp_path / "repo"
    root.mkdir()
    work = root / "work.db"
    _project_store(work, [f"L-{i:04d}" for i in range(50)])

    server = _Captured()
    register_checkout_import_tools(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        answer = asyncio.run(
            server.tools["memory_import_checkout"](namespace=_NS, source_path=str(work), ids=[])  # type: ignore[operator]
        )
    finally:
        auth_context_var.reset(reset)

    assert isinstance(answer, dict)
    assert answer.get("status") == "ok", answer
    assert "copy_seconds" in answer, "the reply must report the copy step's own wall time (NFR01)"
    copy_seconds = answer["copy_seconds"]
    assert isinstance(copy_seconds, int | float)
    assert 0 <= copy_seconds < IMPORT_DEADLINE_SECONDS


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestSymlinkSwapRace:
    """PRD-SEC-016 FR02 -- ``checkout_path`` validates by resolve(); the import must not then open by name.

    ``checkout_path`` (``tools/entry.py``) already proved *source_path* resolves
    inside the granted root at validation time. These tests reproduce the exact
    window the FR describes -- a component swapped for an outside-root symlink
    AFTER that validation and BEFORE the file is actually opened -- and prove the
    walk in :func:`trw_memory.tools.entry.open_checkout_file_fd` refuses it rather
    than following the swap, whether the swapped component is an intermediate
    directory or the final file itself.
    """

    def _outside_store(self, tmp_path: Path) -> tuple[Path, str]:
        outside_dir = tmp_path / "outside" / ".trw"
        outside_dir.mkdir(parents=True)
        outside_db = outside_dir / "memory.db"
        outside_db.write_bytes(b"outside secret bytes, never meant to leave its own checkout")
        return outside_db, hashlib.sha256(outside_db.read_bytes()).hexdigest()

    def test_an_intermediate_directory_swapped_after_validation_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        (root / "x" / "y").mkdir(parents=True)
        (root / "x" / "y" / "memory.db").write_bytes(b"legitimate inside bytes")
        outside_db, sha_before = self._outside_store(tmp_path)
        source_path = str(root / "x" / "y" / "memory.db")

        validated = checkout_import.checkout_path(source_path, "memory_import_checkout", within=True)
        assert validated == source_path  # the check passes: the path is real and inside root, right now

        shutil.rmtree(root / "x")  # the swap: "x" becomes a symlink into the outside checkout
        (root / "x").symlink_to(tmp_path / "outside")

        result = _private_checkout_copy(str(root), source_path, "memory_import_checkout")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert hashlib.sha256(outside_db.read_bytes()).hexdigest() == sha_before

    def test_the_final_component_swapped_after_validation_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / "memory.db").write_bytes(b"legitimate inside bytes")
        outside_db, sha_before = self._outside_store(tmp_path)
        source_path = str(root / "memory.db")

        validated = checkout_import.checkout_path(source_path, "memory_import_checkout", within=True)
        assert validated == source_path

        (root / "memory.db").unlink()
        (root / "memory.db").symlink_to(outside_db)

        result = _private_checkout_copy(str(root), source_path, "memory_import_checkout")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert hashlib.sha256(outside_db.read_bytes()).hexdigest() == sha_before

    def test_a_symlinked_import_tmp_is_refused_not_followed(self, tmp_path: Path) -> None:
        from trw_memory.daemon._paths import DaemonPaths

        root = tmp_path / "repo"
        root.mkdir()
        (root / "memory.db").write_bytes(b"inside bytes")
        user_dir = DaemonPaths.resolve(create=True).user_memory_dir
        shutil.rmtree(user_dir / "import-tmp", ignore_errors=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (user_dir / "import-tmp").symlink_to(elsewhere)

        result = _private_checkout_copy(str(root), str(root / "memory.db"), "memory_import_checkout")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert list(elsewhere.iterdir()) == [], "no copy was written through the link"

    def test_no_swap_the_private_copy_still_succeeds_and_reads_only_the_inside_file(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / "memory.db").write_bytes(b"legitimate inside bytes")

        result = _private_checkout_copy(str(root), str(root / "memory.db"), "memory_import_checkout")

        assert not isinstance(result, dict)
        assert Path(result.path).read_bytes() == b"legitimate inside bytes"
        assert Path(result.path) != root / "memory.db", "a private copy, never the caller's own working file"
        assert result.identity == current_identity(result.path), "identity matches the fresh copy on disk"
        result.pin.close()

    def test_200_race_iterations_never_open_outside_and_never_leave_it_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A background thread flips the component at high frequency across many end-to-end import calls.

        PRD-SEC-016 round-2 finding 4: the earlier version of this test swapped
        "x" for a symlink to ``outside`` while hashing ``outside/.trw/memory.db``
        -- a path the redirected leaf (``outside/x``'s stand-in resolves
        ``root/x/y/memory.db`` to ``outside/y/memory.db``, never
        ``outside/.trw/memory.db``) never actually reached. Since that leaf was
        absent, EVERY iteration refused for the mundane reason "file not
        found," regardless of whether the opener's no-follow walk did anything
        at all -- an unsafe by-name implementation would have produced the
        identical "all refused, hash unchanged" result. This version puts a
        real, distinguishable file exactly at the redirected leaf and asserts
        on its content and on which paths a by-name ``os.open`` ever touched.
        """
        root = tmp_path / "repo"
        real_dir = tmp_path / "x-real-backup"
        (root / "x" / "y").mkdir(parents=True)
        (root / "x" / "y" / "memory.db").write_bytes(b"legitimate inside bytes")
        outside_root = tmp_path / "outside"
        (outside_root / "y").mkdir(parents=True)
        redirected_leaf = outside_root / "y" / "memory.db"  # exactly where root/x/y/memory.db resolves once swapped
        redirected_leaf.write_bytes(b"OUTSIDE SECRET: the redirected leaf, never legitimate content")
        sha_before = hashlib.sha256(redirected_leaf.read_bytes()).hexdigest()
        source_path = str(root / "x" / "y" / "memory.db")

        opened_by_name: list[str] = []
        real_open = os.open

        def _tracking_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
            if dir_fd is None:  # a dir_fd-relative open never resolves a path string on its own
                opened_by_name.append(str(path))
            return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[call-overload]

        monkeypatch.setattr("os.open", _tracking_open)

        stop = threading.Event()

        def flipper() -> None:
            while not stop.is_set():
                x = root / "x"
                try:
                    if x.is_symlink():
                        x.unlink()
                        shutil.move(str(real_dir), str(x))
                    elif x.is_dir():
                        shutil.move(str(x), str(real_dir))
                        x.symlink_to(outside_root)
                except OSError:  # trw-fail-silent-allow: a lost race against the main thread's own open attempt is expected on every iteration; the test asserts the OUTCOME (refused or inside-only), not that every flip lands
                    pass

        thread = threading.Thread(target=flipper, daemon=True)
        thread.start()
        try:
            outcomes: list[dict[str, object]] = []
            for _ in range(200):
                result = _private_checkout_copy(str(root), source_path, "memory_import_checkout")
                if isinstance(result, dict):
                    outcomes.append(result)
                else:
                    content = Path(result.path).read_bytes()
                    assert content == b"legitimate inside bytes", "the redirected leaf's content must never surface"
                    result.pin.close()
                    Path(result.path).unlink()
        finally:
            stop.set()
            thread.join(timeout=5)

        assert hashlib.sha256(redirected_leaf.read_bytes()).hexdigest() == sha_before, (
            "the redirected leaf was written to"
        )
        assert all(o["status"] == "refused" for o in outcomes)
        outside_opens = [p for p in opened_by_name if Path(p).is_absolute() and Path(p).is_relative_to(outside_root)]
        assert outside_opens == [], f"a by-name open reached under the outside checkout: {outside_opens}"


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestGrantRootAncestorNotReresolved:
    """PRD-SEC-016 round-2 finding 1 -- entry.py must use the grant's *root* verbatim, never re-resolve it.

    ``daemon/_grants.py::mint_grant`` resolves *root* exactly ONCE, at mint
    time, and stores the resolved string. If ``checkout_path``/
    ``open_checkout_file_fd`` called ``Path(root).resolve()`` again at request
    time, an ancestor of *root* swapped for a symlink AFTER mint -- reachable
    by a DIFFERENT tenant's grant when one checkout is nested inside another
    -- would be silently followed, redirecting "the granted checkout" itself.
    """

    def test_open_checkout_file_fd_refuses_when_an_ancestor_of_root_is_swapped_after_mint(self, tmp_path: Path) -> None:
        from trw_memory.tools.entry import open_checkout_file_fd

        outer = tmp_path / "outer-checkout"  # a DIFFERENT tenant's checkout
        inner = outer / "inner-checkout"  # nested inside it: this tenant's own grant root
        inner.mkdir(parents=True)
        (inner / "memory.db").write_bytes(b"legitimate inside bytes")
        root = str(inner)  # what mint_grant recorded for the INNER checkout's token, at mint time

        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "inner-checkout").mkdir(parents=True)
        (elsewhere / "inner-checkout" / "memory.db").write_bytes(b"OUTSIDE SECRET")

        # The outer tenant's OWN grant lets it write anywhere inside "outer-checkout" --
        # including swapping "outer-checkout" itself out from under the inner grant.
        shutil.rmtree(outer)
        outer.symlink_to(elsewhere)

        result = open_checkout_file_fd(root, str(inner / "memory.db"), "test")

        assert isinstance(result, dict) and result["status"] == "refused"

    def test_no_resolve_call_survives_on_the_grant_root_in_entry_py(self) -> None:
        """A regression trap for the exact bug: `Path(root).resolve()` must not reappear.

        Not a proof of correctness by itself (see the fixture test above for
        that) -- but the round-2 finding was a ONE-LINE reintroduction risk,
        and a source guard catches a future regression even if nobody thinks
        to re-run the fixture test.
        """
        source = Path(entry_module.__file__).read_text()
        assert "Path(root).resolve()" not in source, (
            "entry.py must use the grant's root verbatim (Path(root)), never re-resolve it at request time"
        )


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestSidecarSymlinkEscape:
    """PRD-SEC-016 round-7 finding 2 -- a symlinked WAL/SHM sidecar must never be followed to decide presence.

    The old check, ``Path(source_path + suffix).exists()``, follows ordinary
    (symlink-following) path resolution: a ``work.db-wal`` planted as a
    symlink turns "does this sidecar exist" into "does <attacker-chosen
    path> exist" -- an existence oracle over any path the daemon user can
    stat, requiring no race at all (unlike FR02's main-file swap, which
    needs to win a timing window). The fix anchors the check on the same
    no-follow-verified parent directory descriptor the leaf itself came
    from, and refuses a symlinked sidecar regardless of what it points at.
    """

    def test_a_regular_wal_sidecar_still_refuses_the_import(self, tmp_path: Path) -> None:
        """Regression control: the ORIGINAL behavior (a real, uncheckpointed WAL file) must be unchanged."""
        root = tmp_path / "checkout"
        root.mkdir()
        work = root / "work.db"
        work.write_bytes(b"main file bytes")
        (root / "work.db-wal").write_bytes(b"uncommitted wal bytes")

        result = _private_checkout_copy(str(root), str(work), "test_import")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert "work.db-wal" in str(result["error"])
        assert "sidecar present" in str(result["error"])

    def test_a_symlinked_wal_sidecar_pointing_outside_the_checkout_refuses_without_following_it(
        self, tmp_path: Path
    ) -> None:
        """The existence-oracle case: a symlinked sidecar must refuse regardless of whether its target exists."""
        root = tmp_path / "checkout"
        root.mkdir()
        work = root / "work.db"
        work.write_bytes(b"main file bytes")

        outside_target = tmp_path / "outside-secret.txt"
        outside_target.write_text("another tenant's file")
        (root / "work.db-wal").symlink_to(outside_target)

        result = _private_checkout_copy(str(root), str(work), "test_import")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert "work.db-wal" in str(result["error"])
        # The refusal must never disclose whether the symlink's TARGET exists --
        # that disclosure is the oracle this fix closes.
        assert "outside-secret" not in str(result)

    def test_a_dangling_symlinked_wal_sidecar_still_refuses(self, tmp_path: Path) -> None:
        """A symlink to a NONEXISTENT target must still refuse -- the old ``.exists()`` check would have missed it."""
        root = tmp_path / "checkout"
        root.mkdir()
        work = root / "work.db"
        work.write_bytes(b"main file bytes")
        (root / "work.db-wal").symlink_to(tmp_path / "does-not-exist.txt")

        result = _private_checkout_copy(str(root), str(work), "test_import")

        assert isinstance(result, dict) and result["status"] == "refused"
        assert "work.db-wal" in str(result["error"])

    def test_no_sidecar_present_copies_normally(self, tmp_path: Path) -> None:
        """Regression control: the ordinary, no-sidecar case must not be refused by the new anchored check."""
        root = tmp_path / "checkout"
        root.mkdir()
        work = root / "work.db"
        work.write_bytes(b"main file bytes")

        result = _private_checkout_copy(str(root), str(work), "test_import")

        assert not isinstance(result, dict), result
        result.pin.close()
        Path(result.path).unlink(missing_ok=True)


class TestPrivateCopyBounds:
    """C12 (7.0 freeze): the private copy is bounded, and a failed one never stays on the daemon's disk."""

    @staticmethod
    def _leftovers() -> list[Path]:
        from trw_memory.daemon._paths import DaemonPaths

        return sorted((DaemonPaths.resolve(create=True).user_memory_dir / "import-tmp").rglob("*"))

    @staticmethod
    def _source(tmp_path: Path) -> tuple[Path, Path]:
        root = tmp_path / "repo"
        root.mkdir()
        (root / "memory.db").write_bytes(b"legitimate inside bytes" * 64)
        return root, root / "memory.db"

    def test_a_copy_that_runs_out_of_space_is_refused_and_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import errno

        def out_of_space(_source: object, out: object, _started: float) -> None:
            out.write(b"partial")  # type: ignore[attr-defined]
            out.flush()  # type: ignore[attr-defined]
            raise OSError(errno.ENOSPC, "No space left on device")

        root, source = self._source(tmp_path)
        monkeypatch.setattr(checkout_import, "_bounded_copy", out_of_space)

        result = _private_checkout_copy(str(root), str(source), "memory_import_checkout")

        assert isinstance(result, dict) and result["status"] == "refused", result
        assert "No space left" in str(result["error"])
        assert self._leftovers() == []

    def test_an_oversized_sparse_source_is_refused_before_anything_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        with (root / "memory.db").open("wb") as handle:
            handle.truncate(1024 * 1024)  # sparse: a large logical size, next to no blocks
        monkeypatch.setattr(checkout_import, "IMPORT_COPY_MAX_BYTES", 4096)

        result = _private_checkout_copy(str(root), str(root / "memory.db"), "memory_import_checkout")

        assert isinstance(result, dict) and result["status"] == "refused", result
        assert "import limit" in str(result["error"])
        assert self._leftovers() == []

    def test_a_source_that_grows_past_the_limit_mid_copy_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        monkeypatch.setattr(checkout_import, "IMPORT_COPY_MAX_BYTES", 10)
        monkeypatch.setattr(checkout_import, "_COPY_CHUNK", 4)
        out = io.BytesIO()

        with pytest.raises(OSError, match="grew past"):
            checkout_import._bounded_copy(io.BytesIO(b"x" * 12), out, time.monotonic())
        assert len(out.getvalue()) <= 10

    def test_a_copy_past_its_deadline_is_refused(self) -> None:
        import io

        started = time.monotonic() - checkout_import.IMPORT_COPY_DEADLINE_SECONDS - 1

        with pytest.raises(OSError, match="longer than"):
            checkout_import._bounded_copy(io.BytesIO(b"x"), io.BytesIO(), started)


class TestTheCopyIsAQuietSnapshot:
    """C12: a hot journal, a live writer or a swapped name can never produce an inconsistent private copy."""

    @staticmethod
    def _store(root: Path, rows: int = 50) -> Path:
        root.mkdir()
        work = root / "work.db"
        conn = sqlite3.connect(work)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(n,) for n in range(rows)])
        conn.commit()
        conn.close()
        return work

    #: Another process: POSIX locks never conflict within one process.
    _WRITE = (
        "import sqlite3, sys\n"
        "c = sqlite3.connect(sys.argv[1], timeout=0)\n"
        "try:\n"
        "    c.execute('INSERT INTO t VALUES (999)'); c.commit(); print('committed')\n"
        "except sqlite3.OperationalError as e:\n"
        "    print(e)\n"
    )

    @classmethod
    def _other_process_writes(cls, work: Path) -> str:
        import subprocess

        done = subprocess.run([sys.executable, "-c", cls._WRITE, str(work)], capture_output=True, text=True, check=True)
        return done.stdout.strip()

    @staticmethod
    def _residue() -> list[str]:
        from trw_memory.daemon._paths import DaemonPaths

        tmp_dir = DaemonPaths.resolve(create=False).user_memory_dir / "import-tmp"
        return sorted(p.name for p in tmp_dir.iterdir()) if tmp_dir.exists() else []

    @staticmethod
    def _refused(result: object, reason: str) -> None:
        assert isinstance(result, dict) and result["status"] == "refused", result
        assert reason in str(result["error"]), result

    def test_a_hot_journal_refuses_the_import(self, tmp_path: Path) -> None:
        """A real hot journal: the files as a writer that died mid-transaction leaves them."""
        live = self._store(tmp_path / "live")
        writer = sqlite3.connect(live, isolation_level=None)
        writer.execute("PRAGMA cache_size=1")  # spill, so the main file is written mid-transaction
        writer.execute("BEGIN")
        writer.executemany("INSERT INTO t VALUES (?)", [(n,) for n in range(5000)])
        root = tmp_path / "checkout"
        root.mkdir()
        shutil.copyfile(live, root / "work.db")
        shutil.copyfile(live.with_name("work.db-journal"), root / "work.db-journal")
        writer.execute("ROLLBACK")
        writer.close()

        self._refused(_private_checkout_copy(str(root), str(root / "work.db"), "test_import"), "work.db-journal")
        assert self._residue() == []

    def test_a_clean_leftover_journal_is_refused_too(self, tmp_path: Path) -> None:
        """Conservative by design: a PERSIST/TRUNCATE-mode journal left after a commit also refuses."""
        work = self._store(tmp_path / "checkout")
        (tmp_path / "checkout" / "work.db-journal").write_bytes(b"")

        self._refused(_private_checkout_copy(str(work.parent), str(work), "test_import"), "work.db-journal")

    def test_a_sqlite_writer_cannot_commit_while_the_copy_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The copy holds SQLite's own SHARED lock, so it is a snapshot, not a best-effort detector."""
        work = self._store(tmp_path / "checkout")
        blocked: list[str] = []
        real_copy = checkout_import._bounded_copy

        def copy_with_a_writer(source: object, out: object, started: float) -> int:
            blocked.append(self._other_process_writes(work))
            return real_copy(source, out, started)  # type: ignore[arg-type]

        monkeypatch.setattr(checkout_import, "_bounded_copy", copy_with_a_writer)
        result = _private_checkout_copy(str(work.parent), str(work), "test_import")

        assert blocked == ["database is locked"]
        assert not isinstance(result, dict), result
        assert self._other_process_writes(work) == "committed", "the lock must be released once the copy is done"
        try:
            copy = sqlite3.connect(result.path)
            assert copy.execute("SELECT count(*) FROM t").fetchone() == (50,)
            copy.close()
        finally:
            result.pin.close()
            Path(result.path).unlink()

    def test_a_writer_that_ignores_the_locks_is_caught_and_leaves_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = self._store(tmp_path / "checkout")
        real_copy = checkout_import._bounded_copy

        def copy_then_scribble(source: object, out: object, started: float) -> int:
            copied = real_copy(source, out, started)  # type: ignore[arg-type]
            with open(work, "r+b") as raw:
                raw.seek(200)
                raw.write(b"torn")
            return copied

        monkeypatch.setattr(checkout_import, "_bounded_copy", copy_then_scribble)
        self._refused(_private_checkout_copy(str(work.parent), str(work), "test_import"), "changed during the copy")
        assert self._residue() == []

    def test_a_wal_that_appears_during_the_copy_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work = self._store(tmp_path / "checkout")
        real_copy = checkout_import._bounded_copy

        def copy_then_attach(source: object, out: object, started: float) -> int:
            work.with_name("work.db-wal").write_bytes(b"frames")
            return real_copy(source, out, started)  # type: ignore[arg-type]

        monkeypatch.setattr(checkout_import, "_bounded_copy", copy_then_attach)
        self._refused(_private_checkout_copy(str(work.parent), str(work), "test_import"), "work.db-wal")
        assert self._residue() == []

    def test_sidecars_are_never_probed_beside_a_different_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parent is walked separately from the leaf; a name swapped in between is refused (sol P1)."""
        work = self._store(tmp_path / "checkout")
        real_parent = checkout_import.open_checkout_parent_fd

        def swap_then_open(root: str, path: str, operation: str) -> object:
            work.rename(work.with_name("moved.db"))
            work.write_bytes(b"a different file")
            return real_parent(root, path, operation)

        monkeypatch.setattr(checkout_import, "open_checkout_parent_fd", swap_then_open)
        self._refused(_private_checkout_copy(str(work.parent), str(work), "test_import"), "no longer the file")

    def test_another_reader_or_import_of_the_same_file_never_drops_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSIX locks are per (process, inode): any close on that inode drops them (sol P1)."""
        from trw_memory._live_stores import close_reader_fd
        from trw_memory.tools.entry import open_checkout_file_fd

        work = self._store(tmp_path / "checkout")
        root = str(work.parent)
        outcomes: list[str] = []
        real_copy = checkout_import._bounded_copy
        nested: list[object] = []
        entered: list[bool] = []

        def copy_while_others_come_and_go(source: object, out: object, started: float) -> int:
            if not entered:
                entered.append(True)
                reader = open_checkout_file_fd(root, str(work), "memory_read")
                assert isinstance(reader, int)
                close_reader_fd(reader)
                nested.append(_private_checkout_copy(root, str(work), "test_import"))  # a second holder
                outcomes.append(self._other_process_writes(work))
            return real_copy(source, out, started)  # type: ignore[arg-type]

        monkeypatch.setattr(checkout_import, "_bounded_copy", copy_while_others_come_and_go)
        result = _private_checkout_copy(root, str(work), "test_import")

        assert outcomes == ["database is locked"]
        assert self._other_process_writes(work) == "committed"
        for copy in (result, *nested):
            assert not isinstance(copy, dict), copy
            copy.pin.close()  # type: ignore[union-attr]
            Path(copy.path).unlink()  # type: ignore[union-attr]

    def test_a_hard_linked_source_is_refused(self, tmp_path: Path) -> None:
        """SQLite keeps sidecars beside the name a writer used; another name's WAL is invisible here (sol P1)."""
        work = self._store(tmp_path / "checkout")
        os.link(work, work.with_name("alias.db"))

        self._refused(_private_checkout_copy(str(work.parent), str(work), "test_import"), "2 hard links")

    def test_importing_a_store_this_daemon_has_open_is_refused_before_any_descriptor(self, tmp_path: Path) -> None:
        """C15: closing a descriptor on a live store would drop its connection's locks, so none is ever taken."""
        from trw_memory import _live_stores

        root = tmp_path / "checkout"
        root.mkdir(mode=0o700)
        work = root / "work.db"
        holder = SQLiteBackend(work)
        holder._conn.execute("CREATE TABLE t (n INTEGER)")
        holder._conn.execute("BEGIN IMMEDIATE")
        try:
            identity = (work.stat().st_dev, work.stat().st_ino)
            result = _private_checkout_copy(str(root.resolve()), str(work.resolve()), "memory_import_checkout")

            assert isinstance(result, dict) and result["status"] == "refused", result
            assert "has open" in str(result["error"])
            assert identity not in _live_stores._LEASES
            assert all(parked != identity for parked, _fd in _live_stores._PARKED)
            assert self._other_process_writes(work) == "database is locked", "the live write lock must survive"
            assert self._residue() == []
        finally:
            holder._conn.execute("ROLLBACK")
            holder.close()
