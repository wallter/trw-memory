# ruff: noqa: F811
"""Performance contract tests for lifecycle/tiers.py."""

from __future__ import annotations

import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models.config import MemoryConfig

from ._test_tiers_support import _make_entry, cfg, mem_dir, mgr  # noqa: F401


class TestTierPerformanceContracts:
    def test_hot_tier_latency_p95_under_1ms(self, mgr: TierManager) -> None:
        for index in range(3):
            mgr.hot_put(f"hot-{index}", _make_entry(f"hot-{index}"))

        durations: list[float] = []
        for _ in range(2_000):
            started = time.perf_counter()
            assert mgr.hot_get("hot-1") is not None
            durations.append(time.perf_counter() - started)

        durations.sort()
        assert durations[int(len(durations) * 0.95)] < 0.001

    def test_warm_tier_search_p95_under_50ms(self, mgr: TierManager) -> None:
        for index in range(500):
            mgr.warm_add(
                f"warm-{index}",
                {
                    "id": f"warm-{index}",
                    "content": f"python warm entry {index}" if index % 10 == 0 else f"warm entry {index}",
                    "tags": ["python"] if index % 10 == 0 else ["misc"],
                },
                None,
            )

        durations: list[float] = []
        for _ in range(100):
            started = time.perf_counter()
            results = mgr.warm_search(["python"], None, top_k=25)
            durations.append(time.perf_counter() - started)
            assert results

        durations.sort()
        assert durations[int(len(durations) * 0.95)] < 0.05

    def test_cold_tier_search_p95_under_350ms(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "05"
        cold_partition.mkdir(parents=True, exist_ok=True)
        for index in range(500):
            write_yaml(
                cold_partition / f"cold-{index}.yaml",
                {
                    "id": f"cold-{index}",
                    "content": f"archived python lesson {index}" if index % 10 == 0 else f"archived lesson {index}",
                    "tags": ["python"] if index % 10 == 0 else ["archive"],
                },
            )

        durations: list[float] = []
        for _ in range(25):
            started = time.perf_counter()
            results = mgr.cold_search(["python"])
            durations.append(time.perf_counter() - started)
            assert results

        durations.sort()
        assert durations[int(len(durations) * 0.95)] < 0.35

    def test_hot_tier_memory_budget_under_50mb(self, mgr: TierManager) -> None:
        tracemalloc.start()
        try:
            for index in range(50):
                mgr.hot_put(f"mem-{index}", _make_entry(f"mem-{index}", importance=0.9))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 50 * 1024 * 1024

    def test_sweep_processes_100_entries_under_5_seconds(
        self,
        mgr: TierManager,
        mem_dir: Path,
        cfg: MemoryConfig,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        for index in range(33):
            mgr.hot_put(f"hot-sweep-{index}", _make_entry(f"hot-sweep-{index}", days_old=cfg.hot_ttl_days + 10))

        old_warm_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 20)).isoformat()
        for index in range(33):
            write_yaml(
                entries_dir / f"warm-sweep-{index}.yaml",
                {
                    "id": f"warm-sweep-{index}",
                    "content": f"warm sweep {index}",
                    "importance": 0.05,
                    "status": "active",
                    "last_accessed_at": old_warm_time,
                    "tags": [],
                },
            )

        old_cold_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 20)).isoformat()
        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        for index in range(34):
            write_yaml(
                cold_partition / f"cold-sweep-{index}.yaml",
                {
                    "id": f"cold-sweep-{index}",
                    "content": f"cold sweep {index}",
                    "importance": 0.01,
                    "last_accessed_at": old_cold_time,
                    "tags": [],
                },
            )

        started = time.perf_counter()
        result = mgr.sweep()
        elapsed = time.perf_counter() - started

        assert result.errors == 0
        assert result.demoted >= 36
        assert result.purged >= 34
        assert elapsed < 5.0
