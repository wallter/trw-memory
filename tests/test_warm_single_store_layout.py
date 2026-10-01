"""FR: the tier runtime roots per-namespace tier files at the single store's OWN
directory, not ``storage_path`` (learning L-LhQe).

Before this fix, ``namespace_storage_dir`` always rooted a namespace's tier
directory at ``resolve_storage_root(config)``, even when a set
``memory_single_store_path`` pointed the canonical backend somewhere else. A
project moved onto a single store without also moving ``storage_path`` left
its ``warm.db`` behind at the OLD root while the canonical file lived at the
NEW one.

An intermediate fix collapsed every namespace onto ONE shared tier directory
(the single store's parent) -- codex r1 caught that this broke namespace
isolation: :class:`~trw_memory.lifecycle.tiers._warm.WarmTierStore` is
single-tenant (its rows are never partitioned by namespace), so two namespaces
sharing one ``warm.db`` cross-contaminate. The correct fix keeps ONE
tier directory PER namespace, rooted at the single store's own directory
rather than at ``storage_path``.

There is no automatic copy of an already-orphaned pre-fix ``warm.db``: the
warm tier simply rebuilds itself from the canonical backend on next use
(see ``warmup_tier_manager``), so an orphan is unused, not lost data.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

import trw_memory.lifecycle.tiers._runtime as runtime
from tests.conftest import make_entry
from trw_memory.lifecycle.tiers._runtime import namespace_storage_dir, remember_entry_in_tiers
from trw_memory.models.config import MemoryConfig

_NAMESPACE = "default"


@pytest.fixture
def isolated_cache() -> object:
    """Swap in a fresh empty tier-manager cache, closing everything afterwards."""
    saved_cache = runtime._TIER_MANAGER_CACHE
    runtime._TIER_MANAGER_CACHE = OrderedDict()
    try:
        yield
    finally:
        for mgr in runtime._TIER_MANAGER_CACHE.values():
            mgr.close()
        runtime._TIER_MANAGER_CACHE = saved_cache


def _single_store_config(tmp_path: Path) -> MemoryConfig:
    store = tmp_path / "memory.db"
    return MemoryConfig(storage_path=str(tmp_path), memory_single_store_path=str(store))


def _diverged_config(tmp_path: Path) -> MemoryConfig:
    """A single store whose directory DIFFERS from ``storage_path`` -- the exact
    shape that leaves a namespace's tier files at the wrong root before this fix."""
    old_root = tmp_path / "old-per-namespace-root"
    new_root = tmp_path / "single-store-dir"
    return MemoryConfig(storage_path=str(old_root), memory_single_store_path=str(new_root / "memory.db"))


class TestNamespaceStorageDirHonoursSingleStore:
    def test_tier_dir_is_per_namespace_under_the_single_stores_own_directory(self, tmp_path: Path) -> None:
        config = _diverged_config(tmp_path)
        tier_dir = namespace_storage_dir(config, _NAMESPACE)
        assert tier_dir == (tmp_path / "single-store-dir" / _NAMESPACE).resolve()
        assert tier_dir != (tmp_path / "old-per-namespace-root" / _NAMESPACE).resolve()

    def test_storing_through_the_tier_runtime_writes_no_orphan_at_the_old_root(
        self, isolated_cache: object, tmp_path: Path
    ) -> None:
        config = _diverged_config(tmp_path)
        entry = make_entry(entry_id="M-001", namespace=_NAMESPACE)

        # warm.db (the sqlite-vec sidecar) is only opened when an embedding is
        # written -- an entry with none only touches the JSONL sidecar. A real
        # store computes one, so this is the path the bug actually hit.
        remember_entry_in_tiers(config, _NAMESPACE, entry, embedding=[0.1] * 8)

        old_root_orphan = tmp_path / "old-per-namespace-root" / _NAMESPACE / "memory" / "warm.db"
        correct = tmp_path / "single-store-dir" / _NAMESPACE / "memory" / "warm.db"
        assert not old_root_orphan.exists(), f"nothing should land at the old, wrong root: {old_root_orphan}"
        assert correct.exists(), f"warm.db must sit under the single store's own directory: {correct}"

    def test_two_namespaces_get_different_warm_dbs_and_entries_are_isolated(
        self, isolated_cache: object, tmp_path: Path
    ) -> None:
        """Codex r1 BLOCK: sharing one tier directory across namespaces cross-contaminates
        WarmTierStore, which is single-tenant. Fails against a shared-directory regression."""
        config = _single_store_config(tmp_path)
        remember_entry_in_tiers(
            config,
            "alpha",
            make_entry(entry_id="M-a", namespace="alpha", content="alpha-only-marker"),
            embedding=[0.1] * 8,
        )
        remember_entry_in_tiers(
            config,
            "beta",
            make_entry(entry_id="M-b", namespace="beta", content="beta-only-marker"),
            embedding=[0.2] * 8,
        )

        alpha_dir = namespace_storage_dir(config, "alpha")
        beta_dir = namespace_storage_dir(config, "beta")
        assert alpha_dir != beta_dir, "two namespaces must not share one tier directory"

        alpha_warm = alpha_dir / "memory" / "warm.db"
        beta_warm = beta_dir / "memory" / "warm.db"
        assert alpha_warm.exists()
        assert beta_warm.exists()
        assert alpha_warm != beta_warm, "two namespaces must not share one warm.db"

        # The warm-tier keyword sidecar is the observable proof: a shared file
        # would contain BOTH namespaces' entries.
        alpha_sidecar_text = alpha_warm.with_suffix(".jsonl").read_text(encoding="utf-8")
        beta_sidecar_text = beta_warm.with_suffix(".jsonl").read_text(encoding="utf-8")
        assert "M-b" not in alpha_sidecar_text, "beta's entry must not be visible from alpha's tier files"
        assert "M-a" not in beta_sidecar_text, "alpha's entry must not be visible from beta's tier files"

    def test_without_a_single_store_the_per_namespace_layout_is_unchanged(self, tmp_path: Path) -> None:
        config = MemoryConfig(storage_path=str(tmp_path))
        tier_dir = namespace_storage_dir(config, "ns-a")
        assert tier_dir == (tmp_path / "ns-a").resolve()

    def test_the_warm_cache_rebuilds_automatically_from_the_canonical_store(
        self, isolated_cache: object, tmp_path: Path
    ) -> None:
        """The claim the doctor row's remedy text depends on: a namespace whose warm
        tier is empty (a fresh dir, or an unused legacy orphan left behind) is not
        missing data -- a search through the tier manager rebuilds it from the
        canonical backend (``warmup_tier_manager``), through the public
        ``tier_candidates`` chokepoint every recall goes through."""
        from trw_memory.integrations._backend import create_backend_from_config
        from trw_memory.lifecycle.tiers._runtime import tier_candidates

        config = _single_store_config(tmp_path)
        with create_backend_from_config(config, _NAMESPACE) as backend:
            backend.store(make_entry(entry_id="M-canonical", namespace=_NAMESPACE, content="canonical only marker"))

            tier_dir = namespace_storage_dir(config, _NAMESPACE)
            assert not (tier_dir / "memory" / "warm.db").exists(), "nothing has touched the tier yet"

            found = tier_candidates(config, _NAMESPACE, backend, query="canonical only marker", tags=None, limit=10)

        assert any(row.get("id") == "M-canonical" for row in found), (
            f"expected the canonical-only entry to surface once the tier rebuilt from the backend: {found}"
        )
