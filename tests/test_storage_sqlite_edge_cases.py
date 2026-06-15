# ruff: noqa: F401,F811
"""SQLiteBackend edge-case and integration-adjacent tests."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._vector_ops import (
    delete_vector_internal,
    get_stored_embeddings,
    search_vectors,
    upsert_vector,
    vector_exists,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_storage_sqlite_support import backend, make_entry


class TestDeleteByNamespaceGraphEdges:
    """delete_by_namespace must clean orphan memory_graph_edges (GDPR/integrity)."""

    @staticmethod
    def _insert_edge(backend: SQLiteBackend, source_id: str, target_id: str) -> None:
        backend._conn.execute(
            "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, "similarity", 0.9, "2026-06-15T00:00:00+00:00"),
        )
        backend._conn.commit()

    @staticmethod
    def _edge_count(backend: SQLiteBackend) -> int:
        row = backend._conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()
        return int(row[0]) if row else 0

    def test_delete_by_namespace_removes_edges_touching_deleted_rows(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("A", namespace="project:gone"))
        backend.store(make_entry("B", namespace="project:gone"))
        backend.store(make_entry("K", namespace="project:keep"))
        # edge fully inside the deleted ns, plus edges crossing into/out of it,
        # plus one edge entirely within the surviving ns.
        self._insert_edge(backend, "A", "B")  # both deleted
        self._insert_edge(backend, "A", "K")  # source deleted, target kept
        self._insert_edge(backend, "K", "B")  # source kept, target deleted
        self._insert_edge(backend, "K", "K")  # both kept (must survive)
        assert self._edge_count(backend) == 4

        deleted = backend.delete_by_namespace("project:gone")

        assert deleted == 2
        # Only the wholly-surviving K->K edge remains; the three edges with any
        # endpoint in the deleted namespace are gone.
        rows = [
            (str(r[0]), str(r[1]))
            for r in backend._conn.execute(
                "SELECT source_id, target_id FROM memory_graph_edges"
            ).fetchall()
        ]
        assert rows == [("K", "K")]

    def test_delete_by_namespace_no_entries_leaves_edges_untouched(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("K", namespace="project:keep"))
        self._insert_edge(backend, "K", "K")

        deleted = backend.delete_by_namespace("project:empty")

        assert deleted == 0
        assert self._edge_count(backend) == 1


class TestCrossThreadSafety:
    """Verify SQLiteBackend can be used from multiple threads."""

    def test_store_and_get_from_different_thread(self, backend: SQLiteBackend) -> None:
        """Store from main thread, get from a worker thread."""
        import threading

        backend.store(make_entry("cross-1", "shared data"))
        result_holder: list[MemoryEntry | None] = [None]

        def worker() -> None:
            result_holder[0] = backend.get("cross-1")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

        assert result_holder[0] is not None
        assert result_holder[0].content == "shared data"

    def test_concurrent_writes_from_threads(self, backend: SQLiteBackend) -> None:
        """Multiple threads can write concurrently without ProgrammingError."""
        import threading

        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(10):
                    backend.store(make_entry(f"t{thread_id}-{i}", f"data-{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(thread_id,)) for thread_id in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert backend.count() == 40


class TestDeleteVectorAbsentRow:
    """FR03: _delete_vector when no vec_index row is a no-op."""

    def test_delete_vector_absent_row(self, backend: SQLiteBackend) -> None:
        """_delete_vector for a nonexistent entry_id is a no-op."""
        backend.store(make_entry("no-vec", "test content"))

        if not backend._vec_available:
            backend._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_index ("
                "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
                "entry_id TEXT UNIQUE NOT NULL)"
            )
            backend._conn.commit()

        backend._delete_vector("no-vec")
        preserved = backend.get("no-vec")
        assert preserved is not None
        assert preserved.id == "no-vec"
        assert preserved.content == "test content"


class TestStoredEmbeddings:
    def test_get_stored_embeddings_returns_empty_when_vec_unavailable(self, backend: SQLiteBackend) -> None:
        if backend._vec_available:
            pytest.skip("sqlite-vec available; use round-trip test instead")
        assert backend.get_stored_embeddings(["missing"]) == {}

    def test_get_stored_embeddings_round_trip(self, backend: SQLiteBackend) -> None:
        pytest.importorskip("sqlite_vec")
        entry = make_entry("vec-entry", "vector content")
        backend.store(entry)
        backend.upsert_vector(entry.id, [0.1] * backend._dim)

        embeddings = backend.get_stored_embeddings([entry.id, "missing"])

        assert set(embeddings) == {entry.id}
        assert embeddings[entry.id] == pytest.approx([0.1] * backend._dim)

    def test_get_stored_embeddings_real_sql_error_logs_warning(self) -> None:
        """A REAL sqlite error (corruption/IO) must surface at WARNING, not be
        silently swallowed at debug — a bulk-backfill caller reads {} as 'no
        stored embeddings' and re-embeds everything."""
        from structlog.testing import capture_logs

        class _BoomConn:
            def execute(self, _sql: str, _params: Any) -> Any:
                raise sqlite3.OperationalError("disk I/O error")

        with capture_logs() as logs:
            result = get_stored_embeddings(_BoomConn(), threading.Lock(), vec_available=True, entry_ids=["x"])
        assert result == {}
        load_errors = [e for e in logs if e.get("event") == "vector_load_error"]
        assert load_errors and load_errors[0]["log_level"] == "warning"

    def test_get_stored_embeddings_vec_absent_stays_debug(self) -> None:
        """The expected vec0-module-absent case stays at debug (graceful)."""
        from structlog.testing import capture_logs

        class _VecAbsentConn:
            def execute(self, _sql: str, _params: Any) -> Any:
                raise sqlite3.OperationalError("no such module: vec0")

        with capture_logs() as logs:
            result = get_stored_embeddings(_VecAbsentConn(), threading.Lock(), vec_available=True, entry_ids=["x"])
        assert result == {}
        load_errors = [e for e in logs if e.get("event") == "vector_load_error"]
        assert load_errors and load_errors[0]["log_level"] == "debug"


class TestExistingVectorIds:
    def test_existing_vector_ids_returns_empty_when_vec_unavailable(self, backend: SQLiteBackend) -> None:
        if backend._vec_available:
            pytest.skip("sqlite-vec available; covered by populated test")
        assert backend.existing_vector_ids() == set()

    def test_existing_vector_ids_returns_all_stored_ids(self, backend: SQLiteBackend) -> None:
        pytest.importorskip("sqlite_vec")
        a = make_entry("vec-a", "alpha")
        b = make_entry("vec-b", "bravo")
        c = make_entry("vec-c", "charlie")
        for entry in (a, b, c):
            backend.store(entry)
        backend.upsert_vector(a.id, [0.1] * backend._dim)
        backend.upsert_vector(b.id, [0.2] * backend._dim)
        # c intentionally has no vector

        ids = backend.existing_vector_ids()
        assert ids == {a.id, b.id}


class _FetchOneResult:
    def __init__(self, row: tuple[int, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[int, ...] | None:
        return self._row


class _Vec0MissingConn:
    total_changes = 0

    def __init__(self) -> None:
        self.rollback_called = False
        self.commit_called = False
        self.statements: list[str] = []

    def execute(self, sql: str, _params: object = ()) -> _FetchOneResult:
        self.statements.append(sql)
        if "SELECT rowid" in sql:
            return _FetchOneResult((7,))
        if "vec_memories" in sql:
            raise sqlite3.OperationalError("no such module: vec0")
        return _FetchOneResult(None)

    def commit(self) -> None:
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True


class _AlwaysFailConn:
    def __init__(self, error: sqlite3.Error) -> None:
        self.error = error

    def execute(self, _sql: str, _params: object = ()) -> _FetchOneResult:
        raise self.error


class TestOptionalVecModuleUnavailable:
    def test_upsert_vector_preserves_canonical_write_when_vec0_module_is_missing(self) -> None:
        conn = _Vec0MissingConn()

        upsert_vector(
            conn,
            threading.Lock(),
            vec_available=True,
            dim=2,
            entry_id="vec-missing",
            embedding=[0.1, 0.2],
        )

        assert conn.rollback_called is True
        assert conn.commit_called is False
        assert any("vec_memories" in statement for statement in conn.statements)

    def test_delete_vector_internal_ignores_missing_optional_vec0_module(self) -> None:
        conn = _Vec0MissingConn()

        delete_vector_internal(conn, "vec-missing")

        assert any("vec_memories" in statement for statement in conn.statements)

    def test_vector_exists_treats_missing_optional_vec0_module_as_absent(self) -> None:
        conn = _AlwaysFailConn(sqlite3.OperationalError("no such module: vec0"))

        assert vector_exists(conn, vec_available=True, entry_id="vec-missing") is False

    def test_vector_exists_still_raises_non_optional_sqlite_errors(self) -> None:
        conn = _AlwaysFailConn(sqlite3.DatabaseError("database disk image is malformed"))

        with pytest.raises(sqlite3.DatabaseError, match="malformed"):
            vector_exists(conn, vec_available=True, entry_id="still-bad")


class _RecordingConn:
    """Connection stub that records SQL but never opens a real vec0 table.

    Used to prove the dimension-mismatch guard short-circuits BEFORE any
    struct.pack / SQL execution — these tests need no sqlite-vec install.
    """

    total_changes = 0

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rollback_called = False
        self.commit_called = False

    def execute(self, sql: str, _params: object = ()) -> _FetchOneResult:
        self.statements.append(sql)
        return _FetchOneResult(None)

    def commit(self) -> None:
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True


class TestVectorDimensionMismatch:
    """A vector whose length != the indexed dim must degrade, not crash.

    Before the guard, ``struct.pack(f"{dim}f", *embedding)`` raised an uncaught
    ``struct.error`` (NOT a ``sqlite3.Error``), which propagated through
    ``backend.transaction()`` in the store path and failed the entire write —
    violating the documented "canonical memory write is preserved" contract.
    These run without sqlite-vec because the guard fires before any SQL.
    """

    def test_upsert_vector_skips_wrong_length_embedding_without_raising(self) -> None:
        conn = _RecordingConn()

        # dim=384 table, but a 3-element embedding (e.g. stale config after a
        # model swap). Must NOT raise struct.error and must NOT touch the DB.
        upsert_vector(
            conn,
            threading.Lock(),
            vec_available=True,
            dim=384,
            entry_id="dim-mismatch",
            embedding=[0.1, 0.2, 0.3],
        )

        assert conn.statements == []
        assert conn.commit_called is False
        assert conn.rollback_called is False

    def test_upsert_vector_skips_too_long_embedding_without_raising(self) -> None:
        conn = _RecordingConn()

        upsert_vector(
            conn,
            threading.Lock(),
            vec_available=True,
            dim=4,
            entry_id="too-long",
            embedding=[0.1] * 768,
        )

        assert conn.statements == []

    def test_upsert_vector_does_not_break_outer_transaction_on_mismatch(self) -> None:
        # skip_commit=True is the in-transaction store path. A mismatch must
        # leave the owner's outer transaction intact (no rollback), so the
        # canonical row written alongside still commits.
        conn = _RecordingConn()

        upsert_vector(
            conn,
            threading.Lock(),
            vec_available=True,
            dim=384,
            entry_id="in-txn",
            embedding=[0.5, 0.6],
            skip_commit=True,
        )

        assert conn.rollback_called is False
        assert conn.statements == []

    def test_search_vectors_returns_empty_on_wrong_length_query(self) -> None:
        conn = _RecordingConn()

        results = search_vectors(
            conn,
            threading.Lock(),
            vec_available=True,
            dim=384,
            query_embedding=[0.1, 0.2, 0.3],
        )

        assert results == []
        assert conn.statements == []

    def test_upsert_then_store_row_survives_dimension_mismatch_end_to_end(self, tmp_path: Path) -> None:
        """Integration: a wrong-dim embedding leaves the canonical row intact."""
        pytest.importorskip("sqlite_vec")

        be = SQLiteBackend(tmp_path / "dim.db", dim=384)
        if not be._vec_available:
            pytest.skip("sqlite-vec did not load in this environment")
        entry = make_entry("e2e-dim", "content that must survive")
        be.store(entry)
        # Wrong-length vector (config drift). Must not raise; row must persist.
        be.upsert_vector(entry.id, [0.1] * 768)

        preserved = be.get(entry.id)
        assert preserved is not None
        assert preserved.content == "content that must survive"
        # No vector was stored, so the id is absent from the index.
        assert entry.id not in be.existing_vector_ids()


class TestListEntriesCombinedFilters:
    """FR03: list_entries with combined status + namespace filters."""

    def test_list_entries_combined_filters(self, backend: SQLiteBackend) -> None:
        """list_entries(status=active, namespace='test') applies both filters."""
        backend.store(make_entry("a1", status=MemoryStatus.ACTIVE, namespace="test"))
        backend.store(make_entry("a2", status=MemoryStatus.ACTIVE, namespace="other"))
        backend.store(make_entry("r1", status=MemoryStatus.RESOLVED, namespace="test"))
        backend.store(make_entry("r2", status=MemoryStatus.RESOLVED, namespace="other"))

        results = backend.list_entries(status=MemoryStatus.ACTIVE, namespace="test")
        ids = {entry.id for entry in results}
        assert ids == {"a1"}


class TestDeleteByNamespace:
    """FR03: delete_by_namespace with populated and empty namespaces."""

    def test_delete_by_namespace_populated(self, backend: SQLiteBackend) -> None:
        """delete_by_namespace('test-ns') with 3 entries returns 3."""
        for i in range(3):
            backend.store(make_entry(f"dns-{i}", namespace="test-ns"))
        backend.store(make_entry("other-ns", namespace="keep"))

        deleted = backend.delete_by_namespace("test-ns")
        assert deleted == 3
        assert backend.count("test-ns") == 0
        assert backend.count("keep") == 1

    def test_delete_by_namespace_empty(self, backend: SQLiteBackend) -> None:
        """delete_by_namespace('empty-ns') with 0 entries returns 0."""
        backend.store(make_entry("x", namespace="different"))
        deleted = backend.delete_by_namespace("empty-ns")
        assert deleted == 0


class TestCountWithNamespaceFilter:
    """FR03: count(namespace='specific') returns filtered count."""

    def test_count_with_namespace_filter(self, backend: SQLiteBackend) -> None:
        """count(namespace='specific') returns only that namespace's count."""
        for i in range(4):
            backend.store(make_entry(f"spec-{i}", namespace="specific"))
        for i in range(2):
            backend.store(make_entry(f"other-{i}", namespace="other"))

        assert backend.count(namespace="specific") == 4
        assert backend.count(namespace="other") == 2
        assert backend.count() == 6


class TestSqliteVecExtensionLoadFailure:
    """Regression coverage for sqlite extension load failures."""

    @staticmethod
    def _install_bad_connect(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
        """Patch sqlite3.connect to return a proxy whose enable_load_extension raises."""
        import sqlite3

        original_connect = sqlite3.connect

        class _ConnProxy:
            def __init__(self, conn: sqlite3.Connection, inner_exc: Exception) -> None:
                self._conn = conn
                self._exc = inner_exc

            def enable_load_extension(self, _enabled: bool) -> None:
                raise self._exc

            def __getattr__(self, name: str) -> object:
                return getattr(self._conn, name)

        def connect_proxy(*args: Any, **kwargs: Any) -> object:
            return _ConnProxy(original_connect(*args, **kwargs), exc)

        monkeypatch.setattr(sqlite3, "connect", connect_proxy)

    def test_backend_init_survives_attribute_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Python without load_extension must not crash backend init."""
        self._install_bad_connect(monkeypatch, AttributeError("enable_load_extension not available"))

        sqlite_backend = SQLiteBackend(tmp_path / "noext.db")

        assert sqlite_backend._vec_available is False
        entry = make_entry("no-ext-1", content="should persist without vec")
        sqlite_backend.store(entry)
        got = sqlite_backend.get("no-ext-1")
        assert got is not None
        assert got.content == "should persist without vec"

    def test_backend_init_survives_operational_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Some macOS builds raise OperationalError instead of AttributeError."""
        import sqlite3 as sqlite_module

        self._install_bad_connect(monkeypatch, sqlite_module.OperationalError("not authorized"))

        sqlite_backend = SQLiteBackend(tmp_path / "opfail.db")

        assert sqlite_backend._vec_available is False
        sqlite_backend.store(make_entry("op-1", content="fallback works"))
        assert sqlite_backend.get("op-1") is not None
