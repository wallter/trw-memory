# ruff: noqa: F811
"""Performance contract tests for lifecycle/tiers.py."""

from __future__ import annotations

import time
import tracemalloc

import pytest

from trw_memory.lifecycle.tiers import TierManager

from ._test_tiers_support import _make_entry, cfg, mem_dir, mgr  # noqa: F401
from ._timing import assert_budget


def _seed_warm_tier(mgr: TierManager) -> None:
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


def _seed_cold_tier(mgr: TierManager) -> None:
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


class TestTierPerformanceContracts:
    def test_warm_tier_search_p95_under_50ms(self, mgr: TierManager) -> None:
        _seed_warm_tier(mgr)
        assert mgr.warm_search(["python"], None, top_k=25)

    @pytest.mark.requires_local_timing
    def test_warm_tier_search_p95_under_50ms_budget(self, mgr: TierManager) -> None:
        _seed_warm_tier(mgr)

        durations: list[float] = []
        for _ in range(100):
            started = time.perf_counter()
            mgr.warm_search(["python"], None, top_k=25)
            durations.append(time.perf_counter() - started)

        durations.sort()
        assert_budget("warm_tier_search_p95", durations[int(len(durations) * 0.95)], 0.05, "s")

    def test_cold_tier_search_p95_under_350ms(self, mgr: TierManager) -> None:
        _seed_cold_tier(mgr)
        assert mgr.cold_search(["python"])

    @pytest.mark.requires_local_timing
    def test_cold_tier_search_p95_under_350ms_budget(self, mgr: TierManager) -> None:
        _seed_cold_tier(mgr)

        durations: list[float] = []
        for _ in range(25):
            started = time.perf_counter()
            mgr.cold_search(["python"])
            durations.append(time.perf_counter() - started)

        durations.sort()
        assert_budget("cold_tier_search_p95", durations[int(len(durations) * 0.95)], 0.35, "s")

    def test_hot_tier_memory_budget_under_50mb(self, mgr: TierManager) -> None:
        tracemalloc.start()
        try:
            for index in range(50):
                mgr.hot_put(f"mem-{index}", _make_entry(f"mem-{index}", importance=0.9))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 50 * 1024 * 1024
