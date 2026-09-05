"""Concurrent access tests for SQLite storage using threading.Barrier.

FR06 (PRD-QUAL-038): Verifies that SQLiteBackend handles concurrent
multi-thread access correctly without corruption, OperationalErrors,
or data loss.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str,
    content: str = "test content",
    *,
    tags: list[str] | None = None,
    namespace: str = "default",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        tags=tags or [],
        namespace=namespace,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_backend(tmp_path: Path) -> SQLiteBackend:  # type: ignore[misc]
    """SQLiteBackend shared across threads."""
    db_path = tmp_path / "concurrent.db"
    backend = SQLiteBackend(db_path)
    yield backend  # type: ignore[misc]
    backend.close()


# ---------------------------------------------------------------------------
# Concurrent tests
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """FR06: Concurrent access tests using threading.Barrier."""

    def test_concurrent_store_five_entries(self, shared_backend: SQLiteBackend) -> None:
        """5 threads store simultaneously; all 5 entries exist afterward."""
        num_threads = 5
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                entry = _make_entry(f"conc-{thread_id}", f"content-{thread_id}")
                shared_backend.store(entry)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert shared_backend.count() == num_threads
        for i in range(num_threads):
            result = shared_backend.get(f"conc-{i}", namespace="default")
            assert result is not None, f"Entry conc-{i} missing"

    def test_concurrent_update_same_entry(self, shared_backend: SQLiteBackend) -> None:
        """3 threads update the same entry's tags; no OperationalError."""
        entry = _make_entry("shared-update", "base content", tags=["original"])
        shared_backend.store(entry)

        num_threads = 3
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                shared_backend.update("shared-update", tags=[f"tag-from-thread-{thread_id}"], namespace="default")
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        result = shared_backend.get("shared-update", namespace="default")
        assert result is not None
        # Tags will be from whichever thread committed last — just verify no crash
        assert isinstance(result.tags, list)

    def test_concurrent_store_and_search(self, shared_backend: SQLiteBackend) -> None:
        """Concurrent store + search operations do not corrupt the database."""
        num_writers = 3
        num_readers = 2
        total = num_writers + num_readers
        barrier = threading.Barrier(total)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        # Pre-seed some entries for search to find
        for i in range(5):
            shared_backend.store(_make_entry(f"seed-{i}", f"python programming item {i}"))

        def writer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(5):
                    shared_backend.store(_make_entry(f"w{thread_id}-{j}", f"python thread {thread_id} item {j}"))
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        def reader(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(5):
                    shared_backend.search("python", top_k=10)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads: list[threading.Thread] = [threading.Thread(target=writer, args=(i,)) for i in range(num_writers)]
        threads.extend(threading.Thread(target=reader, args=(i,)) for i in range(num_readers))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_delete_and_get(self, shared_backend: SQLiteBackend) -> None:
        """Concurrent delete + get on the same entry does not crash."""
        entry = _make_entry("delete-target", "will be deleted")
        shared_backend.store(entry)

        num_threads = 4
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def deleter() -> None:
            try:
                barrier.wait(timeout=5)
                shared_backend.delete("delete-target", namespace="default")
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        def getter() -> None:
            try:
                barrier.wait(timeout=5)
                # May return the entry or None depending on timing — both are valid
                shared_backend.get("delete-target", namespace="default")
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=deleter),
            threading.Thread(target=deleter),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_delete_by_namespace(self, shared_backend: SQLiteBackend) -> None:
        """Concurrent namespace deletes do not corrupt the database."""
        # Create entries in two namespaces
        for i in range(5):
            shared_backend.store(_make_entry(f"ns1-{i}", namespace="ns1"))
            shared_backend.store(_make_entry(f"ns2-{i}", namespace="ns2"))

        barrier = threading.Barrier(2)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def delete_ns(ns: str) -> None:
            try:
                barrier.wait(timeout=5)
                shared_backend.delete_by_namespace(ns)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=delete_ns, args=("ns1",)),
            threading.Thread(target=delete_ns, args=("ns2",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert shared_backend.count() == 0

    def test_concurrent_store_unique_ids(self, shared_backend: SQLiteBackend) -> None:
        """All entries stored concurrently have unique IDs."""
        num_threads = 5
        entries_per_thread = 4
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        errors_lock = threading.Lock()
        stored_ids: list[str] = []
        ids_lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(entries_per_thread):
                    eid = f"unique-{thread_id}-{j}-{uuid.uuid4().hex[:8]}"
                    shared_backend.store(_make_entry(eid, f"data-{eid}"))
                    with ids_lock:
                        stored_ids.append(eid)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # All IDs should be unique
        assert len(stored_ids) == num_threads * entries_per_thread
        assert len(set(stored_ids)) == len(stored_ids), "Duplicate IDs found"
        # All should be retrievable
        assert shared_backend.count() == num_threads * entries_per_thread
