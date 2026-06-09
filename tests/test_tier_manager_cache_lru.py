"""Bounded-LRU tests for the tier-manager cache (memory-lifecycle-5).

``_TIER_MANAGER_CACHE`` previously grew without bound, holding one open SQLite
connection per (storage_path, backend, namespace) key forever. It is now a
bounded LRU that closes the evicted manager before dropping it.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import trw_memory.lifecycle.tiers._runtime as runtime
from trw_memory.lifecycle.tiers._runtime import get_tier_manager
from trw_memory.models.config import MemoryConfig

if TYPE_CHECKING:
    from trw_memory.lifecycle.tiers._manager import TierManager


@pytest.fixture
def isolated_cache() -> object:
    """Swap in a fresh empty cache + small cap, restore afterwards."""
    saved_cache = runtime._TIER_MANAGER_CACHE
    saved_max = runtime._TIER_MANAGER_CACHE_MAX
    runtime._TIER_MANAGER_CACHE = OrderedDict()
    runtime._TIER_MANAGER_CACHE_MAX = 3
    try:
        yield
    finally:
        for mgr in runtime._TIER_MANAGER_CACHE.values():
            mgr.close()
        runtime._TIER_MANAGER_CACHE = saved_cache
        runtime._TIER_MANAGER_CACHE_MAX = saved_max


class TestTierManagerCacheLRU:
    def test_cache_is_bounded_and_closes_evicted_manager(self, isolated_cache: object, tmp_path: Path) -> None:
        config = MemoryConfig(storage_path=str(tmp_path), storage_backend="sqlite")

        # Fill the cache to capacity (3 distinct namespaces).
        first: TierManager = get_tier_manager(config, "ns-0")
        get_tier_manager(config, "ns-1")
        get_tier_manager(config, "ns-2")
        assert len(runtime._TIER_MANAGER_CACHE) == 3

        closed: list[bool] = []
        original_close = first.close

        def _tracking_close() -> None:
            closed.append(True)
            original_close()

        first.close = _tracking_close  # type: ignore[method-assign]

        # A 4th distinct namespace overflows the cap; the LRU (ns-0 / first) is
        # evicted AND closed (releasing its SQLite connection).
        get_tier_manager(config, "ns-3")

        assert len(runtime._TIER_MANAGER_CACHE) == 3
        assert closed == [True], "evicted manager was not close()d (connection leak)"
        # The evicted key is gone; the survivors remain.
        keys = {k[2] for k in runtime._TIER_MANAGER_CACHE}
        assert "ns-0" not in keys
        assert {"ns-1", "ns-2", "ns-3"} <= keys

    def test_recent_access_protects_from_eviction(self, isolated_cache: object, tmp_path: Path) -> None:
        config = MemoryConfig(storage_path=str(tmp_path), storage_backend="sqlite")

        get_tier_manager(config, "ns-0")
        get_tier_manager(config, "ns-1")
        get_tier_manager(config, "ns-2")

        # Touch ns-0 so it becomes most-recently-used; ns-1 is now the LRU.
        get_tier_manager(config, "ns-0")
        # Overflow: ns-1 (now LRU) must be evicted, ns-0 must survive.
        get_tier_manager(config, "ns-3")

        keys = {k[2] for k in runtime._TIER_MANAGER_CACHE}
        assert "ns-0" in keys, "recently-accessed manager was wrongly evicted"
        assert "ns-1" not in keys

    def test_same_namespace_returns_cached_instance(self, isolated_cache: object, tmp_path: Path) -> None:
        config = MemoryConfig(storage_path=str(tmp_path), storage_backend="sqlite")
        a = get_tier_manager(config, "ns-0")
        b = get_tier_manager(config, "ns-0")
        assert a is b
        assert len(runtime._TIER_MANAGER_CACHE) == 1
