"""Cold-tier search cache is a bounded LRU (trw-memory-15).

`ColdTierStore._cached_search_entry` inserts one entry per distinct cold YAML
file searched. Before trw-memory-15 the backing cache was an unbounded dict, so
a long-lived process searching across a large cold archive leaked RAM without
limit. These tests pin the bound and the LRU eviction order as behaviour.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.tiers._cold import ColdTierStore
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.storage.persistence import write_yaml


def _make_store(tmp_path: Path, cap: int) -> ColdTierStore:
    return ColdTierStore(tmp_path, WarmTierStore(tmp_path), search_cache_max=cap)


def _seed(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    write_yaml(path, {"content": f"entry {name}", "tags": [name]})
    return path


def test_cache_never_exceeds_bound_and_keeps_newest(tmp_path: Path) -> None:
    store = _make_store(tmp_path, cap=5)
    files = [_seed(tmp_path, f"e{i}") for i in range(20)]
    for path in files:
        assert store._cached_search_entry(path) is not None

    # Bound is enforced regardless of how many distinct files were searched.
    assert len(store._search_cache) == 5
    # Pure insertion order ⇒ the 5 most-recently inserted survive.
    assert set(store._search_cache) == {str(p) for p in files[-5:]}


def test_recent_access_protects_from_lru_eviction(tmp_path: Path) -> None:
    store = _make_store(tmp_path, cap=3)
    f0, f1, f2 = (_seed(tmp_path, f"e{i}") for i in range(3))
    for path in (f0, f1, f2):
        store._cached_search_entry(path)

    # Touch f0 (a cache hit) → it becomes most-recently-used, so f1 is now LRU.
    store._cached_search_entry(f0)

    # Insert a 4th file → exactly one eviction, and it must be f1 (the LRU), not f0.
    f3 = _seed(tmp_path, "e3")
    store._cached_search_entry(f3)

    survivors = set(store._search_cache)
    assert len(survivors) == 3
    assert str(f0) in survivors  # protected by the recent touch
    assert str(f1) not in survivors  # evicted as least-recently-used
    assert {str(f2), str(f3)} <= survivors


def test_cap_of_one_keeps_only_the_last_searched(tmp_path: Path) -> None:
    store = _make_store(tmp_path, cap=1)
    last = None
    for i in range(4):
        last = _seed(tmp_path, f"e{i}")
        store._cached_search_entry(last)
    assert set(store._search_cache) == {str(last)}


def test_manager_passes_configured_bound(tmp_path: Path) -> None:
    # The tier manager must thread the configured knob into the cold store,
    # otherwise the config field is dead (the gap trw-memory-15 originally left).
    from trw_memory.lifecycle.tiers._manager import TierManager
    from trw_memory.models.config import MemoryConfig

    mgr = TierManager(tmp_path, config=MemoryConfig(cold_search_cache_max=7))
    assert mgr._cold_store._search_cache_max == 7
