"""E2E SQLite backend tests for trw-memory."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from tests.conftest import make_entry
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


class TestSQLiteBackend:
    """Section 8 of E2E plan: concurrent writes, graceful degradation."""

    def test_concurrent_writes_with_wal(self, tmp_path: Path) -> None:
        """8.1 — Multiple threads writing concurrently succeed under WAL mode."""
        db_path = tmp_path / "concurrent.db"
        backend = SQLiteBackend(db_path)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def writer(prefix: str) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(25):
                    now = datetime.now(timezone.utc)
                    entry = MemoryEntry(
                        id=f"M-{prefix}-{i:03d}",
                        content=f"{prefix} entry {i}",
                        namespace="default",
                        created_at=now,
                        updated_at=now,
                    )
                    backend.store(entry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        backend.close()
        assert errors == [], f"Concurrent write errors: {errors}"

        verify_backend = SQLiteBackend(db_path)
        all_entries = verify_backend.list_entries(namespace="default", limit=200)
        verify_backend.close()
        assert len(all_entries) == 100

    def test_graceful_degradation_without_sqlite_vec(self, tmp_path: Path) -> None:
        """8.3 — SQLiteBackend works without sqlite-vec for metadata operations."""
        db_path = tmp_path / "no_vec.db"
        backend = SQLiteBackend(db_path)

        entry = make_entry(
            entry_id="no-vec-1",
            content="test without vectors",
        )
        backend.store(entry)

        retrieved = backend.get("no-vec-1", namespace="default")
        assert retrieved is not None
        assert retrieved.content == "test without vectors"

        results = backend.search("test without", top_k=10, namespace="default")
        assert len(results) >= 1
        assert results[0].id == "no-vec-1"

        backend.close()
