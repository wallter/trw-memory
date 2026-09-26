"""A store trw-memory did not write cannot allocate without bound while it is checked and opened (rc8 C12).

An expression in the copy's schema (a CHECK constraint, a generated column, an expression or
partial index) is evaluated by ``quick_check`` and by the migrations' writes, and one expression
can build a huge value in a single step that no progress handler interrupts. Such schema is refused
unless it is trw-memory's own, a virtual table is admitted only exactly as trw-memory writes it
(an fts5 option multiplies the index the open builds), every connection opened under
connection opened on a file registered with ``untrusted_store`` caps each value at
``UNTRUSTED_LENGTH_LIMIT`` and runs under its deadline, whichever thread opens it and however (a reopen too).
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import make_entry
from trw_memory._graph_primitives import _upsert_edge
from trw_memory.exceptions import StorageError
from trw_memory.storage import _connection
from trw_memory.storage._connection import UNTRUSTED_LENGTH_LIMIT, connect, untrusted_store
from trw_memory.storage._stale_handle import ensure_connection_fresh
from trw_memory.storage._stale_handle_detector import sentinel_path
from trw_memory.storage._untrusted_store import verify_untrusted_store
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import _checkout_merge, checkout_import
from trw_memory.tools.checkout_import import memory_import_checkout_impl

#: 64 MiB, four times the cap: small enough to build on the old code, which let it through.
_BIG = 64 * 1024 * 1024


def _forget_verified_stores() -> None:
    """Test seam mirroring the removed ``_connection.forget_verified_stores``."""
    with _connection._VERIFIED_LOCK:
        _connection._VERIFIED_STORES.clear()


def _under_deadline(path: Path, fn: object, *args: object) -> object:
    """Run *fn* the way the import does: with *path* registered as an untrusted store."""
    with untrusted_store(path, time.monotonic() + 30.0):
        return fn(*args)  # type: ignore[operator]


def _own_store(path: Path, *, dim: int) -> None:
    """A store trw-memory writes itself: a row, its vector and a graph edge (a CHECK-bearing table)."""
    store = SQLiteBackend(path, dim=dim)
    try:
        store.store(make_entry(entry_id="L-1", namespace="default", content="row L-1"))
        if store.vec_available:
            store.upsert_vector("L-1", [1.0] + [0.0] * (dim - 1), namespace="default")
        with store._lock:
            _upsert_edge(store._conn, "L-1", "L-2", "related_to", 0.5, "2026-09-25T00:00:00+00:00", namespace="default")
            store._conn.commit()
    finally:
        store.close()


@pytest.mark.parametrize(
    ("schema", "named"),
    [
        (f"CREATE TABLE t(x CHECK (length(hex(zeroblob({_BIG}))) > 0)); INSERT INTO t VALUES (1);", "table t"),
        ("CREATE TABLE t(x, CONSTRAINT c check(x > 0)); INSERT INTO t VALUES (1);", "table t"),
        (f"CREATE TABLE t(x, y AS (hex(zeroblob({_BIG})))); INSERT INTO t(x) VALUES (1);", "table t"),
        (f"CREATE TABLE t(x); CREATE INDEX i ON t(hex(zeroblob({_BIG})));", "index i"),
        (f"CREATE TABLE t(x); CREATE INDEX i ON t(x) WHERE length(hex(zeroblob({_BIG}))) > 0;", "index i"),
        (
            "CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, namespace UNINDEXED, content, detail, tags, "
            "tokenize='unicode61 remove_diacritics 1', prefix='1 2 3 4 5 6 7 8 9 10');",
            "table memories_fts",
        ),
    ],
)
def test_schema_that_carries_expressions_is_refused_before_quick_check(tmp_path: Path, schema: str, named: str) -> None:
    hostile = tmp_path / "hostile.db"
    with contextlib.closing(sqlite3.connect(hostile)) as conn:
        conn.executescript(schema)
        conn.commit()
    _forget_verified_stores()
    with pytest.raises(StorageError, match="never creates") as refused:
        _under_deadline(hostile, verify_untrusted_store, hostile)
    assert named in str(refused.value)


def test_a_connection_opened_under_the_open_deadline_caps_every_value(tmp_path: Path) -> None:
    db = tmp_path / "any.db"
    sqlite3.connect(db).close()

    def build(size: int) -> int:
        conn = connect(db, dbapi=sqlite3, timeout=0.0, check_same_thread=False)
        try:
            return int(conn.execute("SELECT length(zeroblob(?))", (size,)).fetchone()[0])
        finally:
            conn.close()

    assert _under_deadline(db, build, UNTRUSTED_LENGTH_LIMIT) == UNTRUSTED_LENGTH_LIMIT
    with pytest.raises(sqlite3.DatabaseError, match="too big"):
        _under_deadline(db, build, UNTRUSTED_LENGTH_LIMIT + 1)
    assert build(UNTRUSTED_LENGTH_LIMIT + 1) == UNTRUSTED_LENGTH_LIMIT + 1  # a store trw-memory wrote: no cap


def test_trw_memory_own_schema_is_admitted_and_its_vectors_read_under_the_cap(tmp_path: Path) -> None:
    """No false refusal: the two CHECK tables, a table renamed the way schema 5 rebuilt them, and a
    whole default-dimension ``vec_memories`` chunk (1.5 MiB, so a 1 MiB cap would refuse it) read
    as one value under the cap. sqlite-vec itself reads vectors by incremental blob I/O."""
    own = tmp_path / "own.db"
    _own_store(own, dim=384)
    with contextlib.closing(sqlite3.connect(own)) as conn:
        conn.execute("ALTER TABLE memory_graph_edges RENAME TO memory_graph_edges_v5_rebuild")
        conn.execute("ALTER TABLE memory_graph_edges_v5_rebuild RENAME TO memory_graph_edges")
        conn.commit()
    _forget_verified_stores()

    def check_and_read() -> tuple[set[str], int]:
        verify_untrusted_store(own)
        backend = SQLiteBackend(own, dim=384, check_integrity_once=True)
        try:
            if not backend.vec_available:
                pytest.skip("sqlite-vec unavailable")
            chunk = backend._conn.execute("SELECT vectors FROM vec_memories_vector_chunks00").fetchone()[0]
            return set(backend.existing_vector_ids(namespace="default")), len(chunk)
        finally:
            backend.close()

    assert _under_deadline(own, check_and_read) == ({"L-1"}, 1024 * 384 * 4)


@pytest.fixture
def user_store(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "user.db", dim=4)
    yield store
    store.close()


def test_without_sqlite_length_limits_the_import_is_refused_as_unsupported_runtime(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.10's ``sqlite3.Connection`` has no ``setlimit``: the copy is never opened there."""
    source = tmp_path / "project.db"
    _own_store(source, dim=4)
    opened: list[object] = []
    monkeypatch.setattr(_checkout_merge, "verify_untrusted_store", lambda *args: opened.append(args))
    monkeypatch.setattr(checkout_import, "sqlite3", SimpleNamespace(Connection=type("Connection", (), {})))
    outcome = memory_import_checkout_impl("project:acme-1a2b3c4d", str(source), ["L-1"], backend=user_store)
    assert outcome["status"] == "unsupported_runtime"
    assert "3.11" in str(outcome["error"])
    assert opened == []


@pytest.mark.parametrize(
    "vec_options",
    ["float[4], chunk_size=8", "float[04]", "float[4] ", " float[4]", "float[4], embedding2 float[4]"],
)
def test_a_vec0_table_other_than_the_one_trw_memory_writes_is_refused(tmp_path: Path, vec_options: str) -> None:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    store = tmp_path / "vec.db"
    _own_store(store, dim=4)
    with contextlib.closing(sqlite3.connect(store)) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.executescript(
            f"DROP TABLE IF EXISTS vec_memories; CREATE VIRTUAL TABLE vec_memories USING vec0(embedding {vec_options});"
        )
    _forget_verified_stores()
    with pytest.raises(StorageError, match="table vec_memories"):
        _under_deadline(store, verify_untrusted_store, store)


def test_a_vec0_table_of_any_width_trw_memory_writes_is_admitted(tmp_path: Path) -> None:
    """A checkout's vectors need not be as wide as the daemon's: the import compares them later."""
    store = tmp_path / "vec.db"
    _own_store(store, dim=4)
    _forget_verified_stores()
    _under_deadline(store, verify_untrusted_store, store)


def test_a_reopen_of_a_registered_copy_from_another_thread_is_capped_too(tmp_path: Path) -> None:
    """A stale-handle reopen (a recovery sentinel, an inode change) or a decode fallback opens a new
    connection wherever it happens: keyed by the file, it is capped as the first one was (rc9)."""
    copy = tmp_path / "copy.db"
    _own_store(copy, dim=4)
    _forget_verified_stores()
    outcome: list[BaseException | int] = []

    def reopen_and_build() -> None:
        source = SQLiteBackend(copy, dim=4)
        try:
            before = source._conn
            sentinel = sentinel_path(copy)
            sentinel.write_text("")
            os.utime(sentinel, (time.time() + 60, time.time() + 60))
            source._stale_detector._last_checked = time.monotonic() - 3600  # past any check interval
            ensure_connection_fresh(source)
            assert source._conn is not before, "the stale handle was not reopened"
            outcome.append(
                source._conn.execute("SELECT length(zeroblob(?))", (UNTRUSTED_LENGTH_LIMIT + 1,)).fetchone()[0]
            )
        except BaseException as exc:
            outcome.append(exc)
        finally:
            source.close()

    with untrusted_store(copy, time.monotonic() + 30.0):
        worker = threading.Thread(target=reopen_and_build)
        worker.start()
        worker.join(timeout=30)
    assert len(outcome) == 1 and isinstance(outcome[0], sqlite3.DatabaseError) and "too big" in str(outcome[0])
