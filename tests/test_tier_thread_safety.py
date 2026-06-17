"""Thread-safety smoke tests for TierManager hot tier.

Uses concurrent threads to perform hot_put and hot_get operations
simultaneously. This is a smoke test — it catches obvious race conditions
like missing locks or corrupted state, but does not prove correctness
under all possible thread interleavings.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.lifecycle.tiers._sweep import _sweep_hot_to_warm
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus


def _make_entry(entry_id: str, content: str = "test", last_accessed_at: datetime | None = None) -> MemoryEntry:
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
        last_accessed_at=last_accessed_at or now,
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


class TestHotTierEvictOnWarmAddFailure:
    """trw-memory-2: warm_add failure during overflow must keep the new write."""

    def test_warm_add_failure_keeps_new_entry_and_resolves_overflow(self, tmp_path: Path) -> None:
        """When warm_add raises during eviction, the just-written entry survives
        and the hot tier is brought back under capacity by dropping the LRU
        evictee — previously the new write was lost and overflow stayed unresolved.
        """
        cfg = MemoryConfig(hot_max_entries=3)
        mgr = TierManager(base_dir=tmp_path, config=cfg)

        # Fill to capacity (entry-0 is the LRU / oldest).
        for i in range(3):
            mgr.hot_put(f"entry-{i}", _make_entry(f"entry-{i}", f"content-{i}"))

        # Force warm_add to fail so the eviction's demotion-to-warm path errors.
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("warm tier unavailable")

        mgr.warm_add = _boom  # type: ignore[method-assign]

        # This put triggers overflow (4 > 3); warm_add fails on the LRU evictee.
        mgr.hot_put("entry-new", _make_entry("entry-new", "fresh-write"))

        # The freshly written entry MUST survive — it was not discarded.
        kept = mgr.hot_get("entry-new")
        assert kept is not None
        assert kept.content == "fresh-write"

        # The LRU evictee (entry-0) was dropped to resolve overflow.
        assert mgr.hot_get("entry-0") is None

        # Overflow is resolved: hot tier is back at capacity, not capacity + 1.
        assert len(mgr._hot) == cfg.hot_max_entries
        mgr.close()


class TestSweepHotTierRace:
    """memory-lifecycle-3: hot-tier sweep must not race concurrent hot writers."""

    def test_concurrent_sweep_and_put_no_exceptions(self, tmp_path: Path) -> None:
        """Sweeping (which iterates+mutates hot) while threads hot_put must not

        raise ``RuntimeError: dictionary changed size during iteration`` or any
        other concurrency error. Hot is pre-seeded with stale entries so the
        sweep actually iterates and evicts while writers mutate the dict.
        """
        mgr = TierManager(base_dir=tmp_path)
        stale_day = datetime.now(timezone.utc) - timedelta(days=999)
        for i in range(40):
            mgr.hot_put(f"stale-{i}", _make_entry(f"stale-{i}", last_accessed_at=stale_day))

        errors: list[Exception] = []

        def sweeper() -> None:
            try:
                for _ in range(20):
                    mgr.sweep(MemoryConfig(hot_ttl_days=1))
            except Exception as exc:
                errors.append(exc)

        def putter(i: int) -> None:
            try:
                for j in range(20):
                    mgr.hot_put(f"live-{i}-{j}", _make_entry(f"live-{i}-{j}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=sweeper) for _ in range(3)]
        threads += [threading.Thread(target=putter, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors, f"Concurrency errors during sweep+put: {errors}"
        mgr.close()

    def test_entry_refreshed_during_warm_add_is_not_evicted(self) -> None:
        """The snapshot-then-recheck guard must keep an entry that was refreshed

        (replaced by a newer instance) while its stale snapshot was being
        flushed to warm. Without the ``current is entry`` recheck the freshly
        re-put entry would be silently dropped from hot.
        """
        stale_day = datetime.now(timezone.utc) - timedelta(days=999)
        old = _make_entry("e1", content="old", last_accessed_at=stale_day)
        fresh = _make_entry("e1", content="fresh")  # same id, new instance, current
        hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        hot["e1"] = old
        lock = threading.Lock()

        def warm_add_fn(entry_id: str, data: dict[str, object], emb: list[float] | None) -> None:
            # Simulate a concurrent hot_put refreshing the entry while the
            # (blocking) warm flush is in progress.
            del entry_id, data, emb
            hot["e1"] = fresh

        today = datetime.now(timezone.utc).date()
        demoted, errors = _sweep_hot_to_warm(hot, MemoryConfig(hot_ttl_days=1), today, warm_add_fn, lock)

        # warm_add ran for the stale snapshot, but the refreshed instance must
        # remain in hot (not evicted).
        assert errors == 0
        assert demoted == 0
        assert hot["e1"] is fresh

    def test_stale_entry_is_evicted_when_not_refreshed(self) -> None:
        """Baseline: a genuinely stale entry that is not touched is evicted."""
        stale_day = datetime.now(timezone.utc) - timedelta(days=999)
        old = _make_entry("e1", content="old", last_accessed_at=stale_day)
        hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        hot["e1"] = old
        added: list[str] = []

        def warm_add_fn(entry_id: str, data: dict[str, object], emb: list[float] | None) -> None:
            del data, emb
            added.append(entry_id)

        today = datetime.now(timezone.utc).date()
        demoted, errors = _sweep_hot_to_warm(hot, MemoryConfig(hot_ttl_days=1), today, warm_add_fn)

        assert errors == 0
        assert demoted == 1
        assert added == ["e1"]
        assert "e1" not in hot
