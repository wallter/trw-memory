"""Thread-safety smoke tests for TierManager hot tier.

Uses concurrent threads to perform hot_put and hot_get operations
simultaneously. This is a smoke test — it catches obvious race conditions
like missing locks or corrupted state, but does not prove correctness
under all possible thread interleavings.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.models.memory import MemoryEntry, MemoryStatus


def _make_entry(entry_id: str, content: str = "test") -> MemoryEntry:
    """Create a minimal MemoryEntry for testing."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="",
        tags=[],
        importance=0.5,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


class TestHotTierThreadSafety:
    """Concurrent hot_put + hot_get smoke tests."""

    def test_concurrent_put_no_exceptions(self, tmp_path: Path) -> None:
        """50 threads writing different keys concurrently — no exceptions."""
        mgr = TierManager(base_dir=tmp_path)
        errors: list[Exception] = []
        num_threads = 50

        def worker(i: int) -> None:
            try:
                entry = _make_entry(f"entry-{i}", f"content-{i}")
                mgr.hot_put(f"entry-{i}", entry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent hot_put: {errors}"
        mgr.close()

    def test_concurrent_put_then_get_consistent(self, tmp_path: Path) -> None:
        """After 50 concurrent puts, all entries are retrievable."""
        mgr = TierManager(base_dir=tmp_path)
        num_entries = 50

        def writer(i: int) -> None:
            entry = _make_entry(f"entry-{i}", f"content-{i}")
            mgr.hot_put(f"entry-{i}", entry)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_entries)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Verify all entries exist (up to hot_max_entries capacity)
        found = 0
        for i in range(num_entries):
            entry = mgr.hot_get(f"entry-{i}")
            if entry is not None:
                found += 1
                assert entry.content == f"content-{i}"

        # At least some entries must be present (LRU eviction may drop some)
        assert found > 0, "No entries found after concurrent writes"
        mgr.close()

    def test_concurrent_put_and_get_interleaved(self, tmp_path: Path) -> None:
        """Interleaved put and get operations don't cause crashes."""
        mgr = TierManager(base_dir=tmp_path)
        errors: list[Exception] = []
        num_ops = 50

        def put_worker(i: int) -> None:
            try:
                entry = _make_entry(f"interleaved-{i}")
                mgr.hot_put(f"interleaved-{i}", entry)
            except Exception as exc:
                errors.append(exc)

        def get_worker(i: int) -> None:
            try:
                # May or may not find the entry depending on timing
                mgr.hot_get(f"interleaved-{i}")
            except Exception as exc:
                errors.append(exc)

        threads: list[threading.Thread] = []
        for i in range(num_ops):
            threads.append(threading.Thread(target=put_worker, args=(i,)))
            threads.append(threading.Thread(target=get_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during interleaved ops: {errors}"
        mgr.close()

    def test_concurrent_overwrite_same_key(self, tmp_path: Path) -> None:
        """Multiple threads overwriting the same key — final state is consistent."""
        mgr = TierManager(base_dir=tmp_path)
        errors: list[Exception] = []
        key = "shared-key"
        num_threads = 50

        def writer(i: int) -> None:
            try:
                entry = _make_entry(key, f"content-{i}")
                mgr.hot_put(key, entry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent overwrite: {errors}"
        # The key should exist with SOME valid content
        result = mgr.hot_get(key)
        assert result is not None
        assert result.content.startswith("content-")
        mgr.close()
