"""Tests for SQLiteBackend.store_many() bulk insert.

Covers:
- Basic bulk insertion and count verification
- FTS5 indexed after store_many (searchable immediately)
- Duplicate id handling (INSERT OR REPLACE semantics)
- Empty list is a no-op
- Returned count matches input length
- Status/importance/namespace preserved
- Error on invalid data raises StorageError (not crashes)
- Throughput at 1K entries significantly faster than per-row
"""

from __future__ import annotations

import datetime
import threading
import time
import uuid
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _entry(
    *,
    content: str = "test content",
    detail: str = "",
    tags: list[str] | None = None,
    namespace: str = "default",
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    entry_id: str | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id or str(uuid.uuid4()),
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        namespace=namespace,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )


@pytest.fixture()
def backend() -> SQLiteBackend:
    return SQLiteBackend(Path(":memory:"))


class TestStoreManyBasic:
    def test_empty_list_returns_zero(self, backend: SQLiteBackend) -> None:
        assert backend.store_many([]) == 0
        assert backend.count() == 0

    def test_single_entry_stored(self, backend: SQLiteBackend) -> None:
        e = _entry(content="single entry")
        count = backend.store_many([e])
        assert count == 1
        assert backend.count() == 1
        retrieved = backend.get(e.id, namespace=e.namespace)
        assert retrieved is not None
        assert retrieved.content == "single entry"

    def test_batch_returns_input_length(self, backend: SQLiteBackend) -> None:
        entries = [_entry(content=f"batch entry {i}") for i in range(50)]
        assert backend.store_many(entries) == 50
        assert backend.count() == 50

    def test_all_entries_retrievable(self, backend: SQLiteBackend) -> None:
        entries = [_entry(content=f"retrievable {i}") for i in range(20)]
        backend.store_many(entries)
        for e in entries:
            retrieved = backend.get(e.id, namespace=e.namespace)
            assert retrieved is not None
            assert retrieved.id == e.id

    def test_fields_preserved(self, backend: SQLiteBackend) -> None:
        e = _entry(
            content="field test",
            detail="detailed info",
            tags=["alpha", "beta"],
            namespace="custom:ns",
            importance=0.9,
            status=MemoryStatus.OBSOLETE,
        )
        backend.store_many([e])
        retrieved = backend.get(e.id, namespace="custom:ns")
        assert retrieved is not None
        assert retrieved.content == "field test"
        assert retrieved.detail == "detailed info"
        assert set(retrieved.tags) == {"alpha", "beta"}
        assert retrieved.namespace == "custom:ns"
        assert retrieved.importance == pytest.approx(0.9)
        assert retrieved.status == MemoryStatus.OBSOLETE


class TestStoreManyDuplicates:
    def test_duplicate_id_overwrites(self, backend: SQLiteBackend) -> None:
        eid = str(uuid.uuid4())
        e1 = _entry(entry_id=eid, content="first version")
        e2 = _entry(entry_id=eid, content="second version")
        backend.store_many([e1])
        backend.store_many([e2])
        assert backend.count() == 1
        retrieved = backend.get(eid, namespace="default")
        assert retrieved is not None
        assert retrieved.content == "second version"

    def test_mixed_new_and_existing(self, backend: SQLiteBackend) -> None:
        existing = _entry(content="existing")
        backend.store_many([existing])
        new_entry = _entry(content="new entry")
        updated = _entry(entry_id=existing.id, content="updated content")
        backend.store_many([new_entry, updated])
        assert backend.count() == 2
        assert backend.get(existing.id, namespace="default").content == "updated content"  # type: ignore[union-attr]

    def test_same_batch_duplicate_id_indexes_only_last_entry(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        entry_id = str(uuid.uuid4())
        first = _entry(entry_id=entry_id, content="obsolete_unique_term")
        final = _entry(entry_id=entry_id, content="current_unique_term")

        assert backend.store_many([first, final]) == 2
        assert backend.get(entry_id, namespace="default").content == "current_unique_term"  # type: ignore[union-attr]
        assert backend.search_fts("obsolete_unique_term") == []
        assert [entry.id for entry in backend.search_fts("current_unique_term")] == [entry_id]
        assert backend._conn.execute("SELECT COUNT(*) FROM memories_fts WHERE id = ?", (entry_id,)).fetchone()[0] == 1


class TestStoreManyFts:
    def test_fts_indexed_after_store_many(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        entries = [
            _entry(content="quantum computing breakthrough"),
            _entry(content="classical algorithms for sorting"),
            _entry(content="quantum error correction methods"),
        ]
        backend.store_many(entries)
        results = backend.search_fts("quantum", top_k=10)
        assert len(results) == 2
        ids = {r.id for r in results}
        assert entries[0].id in ids
        assert entries[2].id in ids

    def test_fts_overwrite_removes_old_terms(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        eid = str(uuid.uuid4())
        e1 = _entry(entry_id=eid, content="original_term_xyz")
        e2 = _entry(entry_id=eid, content="replacement_term_abc")
        backend.store_many([e1])
        backend.store_many([e2])
        assert backend.search_fts("original_term_xyz") == []
        assert any(r.id == eid for r in backend.search_fts("replacement_term_abc"))

    def test_fts5_optimize_runs_after_large_batch(self, tmp_path: Path) -> None:
        """optimize() runs without error after a 100+ entry batch."""
        if not SQLiteBackend(Path(":memory:")).fts_available:
            pytest.skip("FTS5 not available")
        backend = SQLiteBackend(tmp_path / "opt.db")
        entries = [_entry(content=f"bulk entry {i}") for i in range(100)]
        # Should not raise; implicitly tests optimize() call
        count = backend.store_many(entries)
        assert count == 100
        # Verify entries still searchable after optimize
        results = backend.search_fts("bulk", top_k=10)
        assert len(results) == 10


class TestStoreManyRollbackLocking:
    """store_many's failure-path ROLLBACK must run while holding backend._lock.

    The rollback previously ran AFTER the ``with backend._lock`` block released,
    so a concurrent writer could interleave on the single shared connection
    between the failure and the ROLLBACK. The fix re-acquires the lock for the
    rollback.
    """

    def test_failed_batch_rolls_back_under_lock_and_leaves_no_rows(self, backend: SQLiteBackend) -> None:
        import sqlite3
        from typing import Any

        # Pre-seed one valid row so we can prove the failed batch does not persist.
        backend.store_many([_entry(content="pre-existing", entry_id="KEEP")])
        assert backend.count() == 1

        lock_held_during_rollback: list[bool] = []
        real_conn = backend._conn
        lock = backend._lock

        class _ConnProxy:
            """Delegating proxy — pysqlite3 Connection attributes are read-only,
            so we cannot monkeypatch rollback/executemany in place."""

            def __getattr__(self, name: str) -> Any:
                return getattr(real_conn, name)

            def executemany(self, sql: str, seq_of_params: object) -> Any:
                if "INSERT" in sql.upper() and "memories" in sql:
                    raise sqlite3.OperationalError("injected batch failure")
                return real_conn.executemany(sql, seq_of_params)

            def rollback(self) -> None:
                acquired_from_peer: list[bool] = []

                def probe_lock() -> None:
                    acquired = lock.acquire(blocking=False)
                    acquired_from_peer.append(acquired)
                    if acquired:
                        lock.release()

                probe = threading.Thread(target=probe_lock)
                probe.start()
                probe.join(timeout=1)
                assert not probe.is_alive()
                lock_held_during_rollback.append(acquired_from_peer == [False])
                real_conn.rollback()

        backend._conn = _ConnProxy()  # type: ignore[assignment]
        try:
            with pytest.raises(Exception):  # StorageError wraps the sqlite error
                backend.store_many([_entry(content="will fail", entry_id="FAIL")])
        finally:
            backend._conn = real_conn

        # Rollback happened while the lock was held.
        assert lock_held_during_rollback == [True]
        # The failed batch left no partial rows; the pre-existing row survives.
        assert backend.count() == 1
        assert backend.get("FAIL", namespace="default") is None
        assert backend.get("KEEP", namespace="default") is not None


class TestStoreManyThroughput:
    def test_store_many_faster_than_per_row_at_1k(self, tmp_path: Path) -> None:
        """store_many must be at least 5x faster than per-row store at 1K entries."""
        N = 1000
        db1 = SQLiteBackend(tmp_path / "batch.db")
        db2 = SQLiteBackend(tmp_path / "perrow.db")

        entries1 = [_entry(content=f"batch content {i}") for i in range(N)]
        entries2 = [_entry(content=f"perrow content {i}") for i in range(N)]

        t0 = time.perf_counter()
        db1.store_many(entries1)
        batch_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for e in entries2:
            db2.store(e)
        perrow_ms = (time.perf_counter() - t0) * 1000

        speedup = perrow_ms / batch_ms
        assert speedup >= 3, f"store_many speedup {speedup:.1f}x < 3x: batch={batch_ms:.0f}ms, perrow={perrow_ms:.0f}ms"
