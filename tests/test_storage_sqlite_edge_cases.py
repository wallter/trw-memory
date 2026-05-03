"""SQLiteBackend edge-case and integration-adjacent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_storage_sqlite_support import backend, make_entry


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


class TestExistingVectorIds:
    def test_existing_vector_ids_returns_empty_when_vec_unavailable(
        self, backend: SQLiteBackend
    ) -> None:
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

        def connect_proxy(*args: object, **kwargs: object) -> object:
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
