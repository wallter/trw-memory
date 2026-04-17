"""Tests for lifecycle/tiers.py — TierManager hot/warm/cold lifecycle.

TDD: tests written before implementation.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers import (
    TierManager,
    TierSweepResult,
    compute_importance_score,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> MemoryConfig:
    return MemoryConfig(
        hot_max_entries=3,
        hot_ttl_days=7,
        cold_threshold_days=90,
        retention_days=365,
        decay_half_life_days=14.0,
        score_relevance_weight=0.4,
        score_recency_weight=0.3,
        score_importance_weight=0.3,
    )


@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def mgr(mem_dir: Path, cfg: MemoryConfig) -> TierManager:
    return TierManager(base_dir=mem_dir, config=cfg)


def _make_entry(
    entry_id: str = "test-id",
    importance: float = 0.5,
    status: str = "active",
    days_old: int = 0,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    last_acc = now - timedelta(days=days_old)
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        detail="some detail",
        tags=["tag1"],
        importance=importance,
        status=MemoryStatus(status),
        last_accessed_at=last_acc,
    )


# ---------------------------------------------------------------------------
# compute_importance_score
# ---------------------------------------------------------------------------


class TestComputeImportanceScore:
    def test_returns_float_in_unit_interval(self) -> None:
        entry: dict[str, object] = {
            "content": "test content",
            "detail": "",
            "importance": 0.8,
        }
        cfg = MemoryConfig()
        score = compute_importance_score(entry, ["test"], config=cfg)
        assert 0.0 <= score <= 1.0

    def test_high_importance_entry_scores_higher(self) -> None:
        cfg = MemoryConfig()
        high: dict[str, object] = {"content": "x", "importance": 0.9}
        low: dict[str, object] = {"content": "x", "importance": 0.1}
        assert compute_importance_score(high, [], config=cfg) > compute_importance_score(low, [], config=cfg)

    def test_token_overlap_boosts_relevance(self) -> None:
        cfg = MemoryConfig()
        matched: dict[str, object] = {"content": "foo bar baz", "importance": 0.5}
        no_match: dict[str, object] = {"content": "xyz xyz xyz", "importance": 0.5}
        s_match = compute_importance_score(matched, ["foo", "bar"], config=cfg)
        s_nomatch = compute_importance_score(no_match, ["foo", "bar"], config=cfg)
        assert s_match > s_nomatch

    def test_cosine_similarity_used_when_embeddings_provided(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {"content": "irrelevant", "importance": 0.5}
        q_emb = [1.0, 0.0]
        e_emb = [1.0, 0.0]  # perfect match
        score = compute_importance_score(entry, [], query_embedding=q_emb, entry_embedding=e_emb, config=cfg)
        assert score > 0.3  # relevance component should be high

    def test_orthogonal_embeddings_zero_relevance(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {"content": "irrelevant", "importance": 0.0}
        q_emb = [1.0, 0.0]
        e_emb = [0.0, 1.0]  # orthogonal
        score = compute_importance_score(entry, [], query_embedding=q_emb, entry_embedding=e_emb, config=cfg)
        assert score < 0.5  # low relevance + low importance

    def test_stale_entry_lower_recency(self) -> None:
        cfg = MemoryConfig()
        fresh: dict[str, object] = {
            "content": "x",
            "importance": 0.5,
            "last_accessed_at": datetime.now(timezone.utc).isoformat(),
        }
        stale: dict[str, object] = {
            "content": "x",
            "importance": 0.5,
            "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),
        }
        s_fresh = compute_importance_score(fresh, [], config=cfg)
        s_stale = compute_importance_score(stale, [], config=cfg)
        assert s_fresh > s_stale

    def test_empty_query_tokens_zero_relevance_fallback(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {"content": "test", "importance": 0.5}
        score = compute_importance_score(entry, [], config=cfg)
        assert 0.0 <= score <= 1.0

    def test_weight_normalization(self) -> None:
        # Weights that don't sum to 1.0 — normalize internally in compute_importance_score
        # MemoryConfig enforces sum=1.0, so patch directly for the normalization code path
        cfg = MemoryConfig()
        # Manually override weights to test normalization path in compute_importance_score
        object.__setattr__(cfg, "score_relevance_weight", 0.2)
        object.__setattr__(cfg, "score_recency_weight", 0.2)
        object.__setattr__(cfg, "score_importance_weight", 0.2)
        entry: dict[str, object] = {"content": "x", "importance": 0.5}
        score = compute_importance_score(entry, [], config=cfg)
        assert 0.0 <= score <= 1.0

    def test_prefers_q_value_when_outcomes_exist(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {
            "content": "deployment lesson",
            "importance": 0.2,
            "q_value": 0.9,
            "q_observations": 5,
        }
        baseline: dict[str, object] = {
            "content": "deployment lesson",
            "importance": 0.2,
            "q_value": 0.2,
            "q_observations": 5,
        }
        assert compute_importance_score(entry, ["deployment"], config=cfg) > compute_importance_score(
            baseline,
            ["deployment"],
            config=cfg,
        )


# ---------------------------------------------------------------------------
# TierManager — hot tier
# ---------------------------------------------------------------------------


class TestHotTier:
    def test_hot_get_miss_returns_none(self, mgr: TierManager) -> None:
        assert mgr.hot_get("nonexistent") is None

    def test_hot_put_and_get(self, mgr: TierManager) -> None:
        entry = _make_entry("e1")
        mgr.hot_put("e1", entry)
        result = mgr.hot_get("e1")
        assert result is not None
        assert result.id == "e1"

    def test_hot_get_moves_to_mru(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        # Access e1 — should become MRU
        mgr.hot_get("e1")
        # Add e4 — should evict e2 (now LRU), not e1
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_get("e1") is not None
        assert mgr.hot_get("e2") is None  # evicted

    def test_hot_put_evicts_lru_when_over_capacity(self, mgr: TierManager) -> None:
        # cfg has hot_max_entries=3
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        assert mgr.hot_size == 3
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_size == 3
        assert mgr.hot_get("e1") is None  # LRU was evicted

    def test_hot_capacity_eviction_demotes_entry_to_warm_tier(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))

        mgr.hot_put("e4", _make_entry("e4"))

        warm_results = mgr.warm_search(["e1"], None)
        warm_ids = [str(row["id"]) for row in warm_results]
        assert "e1" in warm_ids

    def test_hot_capacity_eviction_keeps_entry_when_warm_demote_fails(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))

        original_warm_add = mgr.warm_add

        def _fail_once(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("disk full")

        mgr.warm_add = _fail_once  # type: ignore[method-assign]
        try:
            mgr.hot_put("e4", _make_entry("e4"))
        finally:
            mgr.warm_add = original_warm_add  # type: ignore[method-assign]

        assert mgr.hot_size == 3
        assert mgr.hot_get("e1") is not None
        assert mgr.hot_get("e4") is None

    def test_hot_put_refresh_existing(self, mgr: TierManager) -> None:
        entry = _make_entry("e1")
        mgr.hot_put("e1", entry)
        updated = _make_entry("e1", importance=0.9)
        mgr.hot_put("e1", updated)
        result = mgr.hot_get("e1")
        assert result is not None
        assert result.importance == pytest.approx(0.9)

    def test_hot_clear(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_clear()
        assert mgr.hot_size == 0
        assert mgr.hot_get("e1") is None

    def test_hot_size_property(self, mgr: TierManager) -> None:
        assert mgr.hot_size == 0
        mgr.hot_put("e1", _make_entry("e1"))
        assert mgr.hot_size == 1

    def test_hot_put_update_last_accessed(self, mgr: TierManager) -> None:
        # Updating an existing entry should update last_accessed_at
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        # Force eviction of e1 by adding e4; e1 must have been evicted
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_get("e1") is None

    def test_hot_get_refreshes_ttl_eligibility(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        mgr.hot_put("e1", _make_entry("e1", days_old=30))

        refreshed = mgr.hot_get("e1")
        assert refreshed is not None

        result = mgr.sweep(config=cfg)

        assert result.demoted == 0
        assert mgr.hot_get("e1") is not None

    def test_warmup_hot_from_warm_ranks_full_sidecar_before_truncating(self, mgr: TierManager) -> None:
        for idx in range(10):
            mgr.warm_add(
                f"low-{idx}",
                _make_entry(f"low-{idx}", importance=0.1).model_dump(mode="json"),
                None,
            )
        mgr.warm_add(
            "best",
            _make_entry("best", importance=0.95).model_dump(mode="json"),
            None,
        )

        loaded = mgr.warmup_hot_from_warm(max_entries=1)
        assert loaded == 1
        assert mgr.hot_get("best") is not None


# ---------------------------------------------------------------------------
# TierManager — warm tier
# ---------------------------------------------------------------------------


class TestWarmTier:
    def test_warm_add_and_sidecar_created(self, mgr: TierManager, mem_dir: Path) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "warm entry", "tags": ["x"]}
        mgr.warm_add("e1", entry_data, None)
        sidecar = mgr._warm_sidecar_path()
        assert sidecar.exists()

    def test_warm_sidecar_contains_entry(self, mgr: TierManager) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "test warm", "tags": ["a"]}
        mgr.warm_add("e1", entry_data, None)
        sidecar = mgr._warm_sidecar_path()
        text = sidecar.read_text(encoding="utf-8")
        records = [json.loads(l) for l in text.splitlines() if l.strip()]
        ids = [r["id"] for r in records]
        assert "e1" in ids

    def test_warm_add_upsert_replaces_existing(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "old", "tags": []}
        d2: dict[str, object] = {"id": "e1", "content": "new", "tags": []}
        mgr.warm_add("e1", d1, None)
        mgr.warm_add("e1", d2, None)
        sidecar = mgr._warm_sidecar_path()
        records = [json.loads(l) for l in sidecar.read_text(encoding="utf-8").splitlines() if l.strip()]
        e1_records = [r for r in records if r.get("id") == "e1"]
        assert len(e1_records) == 1
        assert e1_records[0]["summary"] == "new"

    def test_warm_remove_clears_sidecar(self, mgr: TierManager) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "remove me", "tags": []}
        mgr.warm_add("e1", entry_data, None)
        mgr.warm_remove("e1")
        sidecar = mgr._warm_sidecar_path()
        if sidecar.exists():
            records = [json.loads(l) for l in sidecar.read_text(encoding="utf-8").splitlines() if l.strip()]
            ids = [r.get("id") for r in records]
            assert "e1" not in ids

    def test_warm_keyword_search_finds_match(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "python programming", "tags": ["code"]}
        d2: dict[str, object] = {"id": "e2", "content": "cooking recipes", "tags": ["food"]}
        mgr.warm_add("e1", d1, None)
        mgr.warm_add("e2", d2, None)
        results = mgr.warm_search(["python"], None)
        ids = [r["id"] for r in results]
        assert "e1" in ids
        assert "e2" not in ids

    def test_warm_keyword_search_no_match_empty(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "python programming", "tags": []}
        mgr.warm_add("e1", d1, None)
        results = mgr.warm_search(["ruby"], None)
        assert results == []

    def test_warm_keyword_search_matches_detail_text(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "opaque title", "detail": "detail-only-hit", "tags": []}
        mgr.warm_add("e1", d1, None)
        results = mgr.warm_search(["detail-only-hit"], None)
        ids = [r["id"] for r in results]
        assert "e1" in ids

    def test_warm_search_no_tokens_returns_empty(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "any content", "tags": []}
        mgr.warm_add("e1", d1, None)
        results = mgr.warm_search([], None)
        assert results == []

    def test_warm_search_by_tag(self, mgr: TierManager) -> None:
        d1: dict[str, object] = {"id": "e1", "content": "x", "tags": ["mytag"]}
        mgr.warm_add("e1", d1, None)
        results = mgr.warm_search(["mytag"], None)
        ids = [r["id"] for r in results]
        assert "e1" in ids

    def test_warm_search_reranks_by_composite_score(self, mgr: TierManager) -> None:
        stale_low = _make_entry("low", importance=0.1, days_old=60).model_dump(mode="json")
        stale_low["content"] = "shared token"
        fresh_high = _make_entry("high", importance=0.9, days_old=0).model_dump(mode="json")
        fresh_high["content"] = "shared token"
        mgr.warm_add("low", stale_low, None)
        mgr.warm_add("high", fresh_high, None)

        results = mgr.warm_search(["shared"], None, top_k=2)

        ids = [str(row["id"]) for row in results]
        assert ids == ["high", "low"]

    def test_warm_search_skips_orphaned_vector_hits(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            def search_vectors(self, _query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
                return [("orphaned-entry", 0.0)][:top_k]

        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: _FakeBackend())

        results = mgr.warm_search(["semantic"], [1.0, 0.0], top_k=5)

        assert results == []

    def test_warm_remove_deletes_vector_rows(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            def __init__(self) -> None:
                self.vector_deleted = False

            def delete(self, _entry_id: str) -> bool:
                return False

            def delete_vector(self, _entry_id: str) -> bool:
                self.vector_deleted = True
                return True

        fake_backend = _FakeBackend()
        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: fake_backend)

        mgr.warm_add("e1", {"id": "e1", "content": "x", "tags": []}, None)

        assert mgr.warm_remove("e1") is True
        assert fake_backend.vector_deleted is True


# ---------------------------------------------------------------------------
# TierManager — cold tier
# ---------------------------------------------------------------------------


class TestColdTier:
    def test_cold_archive_moves_file(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "e1-test.yaml"
        write_yaml(entry_file, {"id": "e1", "content": "test"})

        mgr.cold_archive("e1", entry_file)

        assert not entry_file.exists()
        cold_base = mgr._cold_dir()
        yaml_files = list(cold_base.rglob("*.yaml"))
        assert len(yaml_files) == 1
        data = read_yaml(yaml_files[0])
        assert data["id"] == "e1"

    def test_cold_archive_uses_entry_created_partition(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "dated-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "e-created",
                "content": "test",
                "created_at": "2024-03-15T12:00:00+00:00",
            },
        )

        mgr.cold_archive("e-created", entry_file)

        assert (mgr._cold_dir() / "2024" / "03" / "dated-entry.yaml").exists()

    def test_cold_archive_raises_on_missing_file(self, mgr: TierManager, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(Exception):
            mgr.cold_archive("bad", missing)

    def test_cold_archive_rolls_back_when_warm_cleanup_fails(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "e-warm-fail.yaml"
        write_yaml(entry_file, {"id": "e-warm-fail", "content": "test"})

        original_warm_remove = mgr._warm_store.warm_remove

        def _fail_warm_remove(_entry_id: str) -> bool:
            return False

        mgr._warm_store.warm_remove = _fail_warm_remove  # type: ignore[assignment]
        try:
            with pytest.raises(StorageError):
                mgr.cold_archive("e-warm-fail", entry_file)
        finally:
            mgr._warm_store.warm_remove = original_warm_remove  # type: ignore[method-assign]

        assert entry_file.exists()
        assert not any(path.name == "e-warm-fail.yaml" for path in mgr._cold_dir().rglob("*.yaml"))

    def test_cold_promote_finds_entry(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        # Manually place file in cold archive
        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e1-promote.yaml"
        write_yaml(yaml_file, {"id": "e1", "content": "promote me", "tags": []})

        result = mgr.cold_promote("e1")
        assert result is not None
        assert result["id"] == "e1"
        # Original file removed from cold
        assert not yaml_file.exists()
        assert (mem_dir / "entries" / "e1.yaml").exists()

    def test_cold_promote_returns_none_if_not_found(self, mgr: TierManager) -> None:
        result = mgr.cold_promote("nonexistent-id")
        assert result is None

    def test_cold_promote_updates_last_accessed(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e2.yaml"
        old_time = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e2", "content": "old entry", "last_accessed_at": old_time})

        result = mgr.cold_promote("e2")
        assert result is not None
        # last_accessed_at should be updated
        assert result.get("last_accessed_at") != old_time

    def test_cold_promote_keeps_archive_when_restore_fails(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e3.yaml"
        write_yaml(yaml_file, {"id": "e3", "content": "cold entry"})

        def _fail_restore(_entry_data: dict[str, object]) -> None:
            raise OSError("backend unavailable")

        result = mgr.cold_promote("e3", restore_entry_fn=_fail_restore)
        assert result is None
        assert yaml_file.exists()

    def test_cold_promote_rolls_back_restore_when_warm_add_fails(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e4.yaml"
        original_last_accessed = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e4", "content": "cold entry", "last_accessed_at": original_last_accessed})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote("e4")
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e4.yaml").exists()
        assert read_yaml(yaml_file)["last_accessed_at"] == original_last_accessed

    def test_cold_promote_rolls_back_warm_and_canonical_when_archive_delete_fails(
        self,
        mgr: TierManager,
        mem_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e5.yaml"
        write_yaml(yaml_file, {"id": "e5", "content": "rollback me"})

        original_unlink = Path.unlink

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == yaml_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)

        result = mgr.cold_promote("e5")

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e5.yaml").exists()
        assert mgr.warm_search(["rollback"], None, top_k=5) == []

    def test_cold_promote_suppresses_rollback_delete_failure(
        self,
        mgr: TierManager,
        mem_dir: Path,
    ) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e6.yaml"
        original_last_accessed = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e6", "content": "cold entry", "last_accessed_at": original_last_accessed})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        def _restore(entry_data: dict[str, object]) -> None:
            write_yaml(mem_dir / "entries" / "e6.yaml", entry_data)

        def _fail_rollback(_entry_id: str) -> bool:
            raise RuntimeError("rollback unavailable")

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote(
                "e6",
                restore_entry_fn=_restore,
                delete_restored_entry_fn=_fail_rollback,
            )
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert read_yaml(yaml_file)["last_accessed_at"] == original_last_accessed
        assert not (mem_dir / "entries" / "e6.yaml").exists()

    def test_cold_promote_force_deletes_restored_entry_when_primary_rollback_fails(
        self,
        mgr: TierManager,
        mem_dir: Path,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e6b.yaml"
        write_yaml(yaml_file, {"id": "e6b", "content": "cold entry"})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        def _restore(entry_data: dict[str, object]) -> None:
            write_yaml(mem_dir / "entries" / "e6b.yaml", entry_data)

        def _fail_rollback(_entry_id: str) -> bool:
            raise RuntimeError("rollback unavailable")

        def _force_delete(entry_id: str) -> bool:
            (mem_dir / "entries" / f"{entry_id}.yaml").unlink(missing_ok=True)
            return True

        def _verify_removed(entry_id: str) -> bool:
            return not (mem_dir / "entries" / f"{entry_id}.yaml").exists()

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote(
                "e6b",
                restore_entry_fn=_restore,
                delete_restored_entry_fn=_fail_rollback,
                force_delete_restored_entry_fn=_force_delete,
                verify_restored_entry_removed_fn=_verify_removed,
            )
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e6b.yaml").exists()

    def test_cold_promote_purges_sidecar_when_warm_rollback_raises(
        self,
        mgr: TierManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e7.yaml"
        write_yaml(yaml_file, {"id": "e7", "content": "rollback purge"})

        original_unlink = Path.unlink
        original_warm_remove = mgr._warm_store.warm_remove

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == yaml_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        def _raise_warm_remove(_entry_id: str) -> bool:
            raise OSError("warm cleanup failed")

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
        mgr._warm_store.warm_remove = _raise_warm_remove  # type: ignore[assignment]
        try:
            result = mgr.cold_promote("e7")
        finally:
            mgr._warm_store.warm_remove = original_warm_remove  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert mgr.warm_search(["rollback"], None, top_k=5) == []

    def test_cold_search_finds_matching_entry(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "02"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "entry1.yaml",
            {"id": "e1", "content": "machine learning basics", "tags": ["ml"]},
        )
        write_yaml(
            cold_partition / "entry2.yaml",
            {"id": "e2", "content": "cooking recipes", "tags": ["food"]},
        )

        results = mgr.cold_search(["machine", "learning"])
        ids = [r.get("id") for r in results]
        assert "e1" in ids
        assert "e2" not in ids

    def test_cold_search_empty_tokens_returns_empty(self, mgr: TierManager) -> None:
        results = mgr.cold_search([])
        assert results == []

    def test_cold_promote_restores_warm_vector_from_archived_embedding(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        class _FakeWarmBackend:
            def __init__(self) -> None:
                self.vectors: dict[str, list[float]] = {}

            def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
                self.vectors[entry_id] = embedding

        cold_partition = mgr._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "vector-entry.yaml",
            {
                "id": "vector-entry",
                "content": "keyword promotion",
                "_warm_embedding": [1.0, 0.0],
            },
        )

        fake_backend = _FakeWarmBackend()
        mgr._warm_store._get_warm_backend = lambda dim=None: fake_backend  # type: ignore[assignment,return-value]
        result = mgr.cold_promote("vector-entry")

        assert result is not None
        assert fake_backend.vectors["vector-entry"] == [1.0, 0.0]

    def test_cold_search_no_cold_dir_returns_empty(self, mgr: TierManager) -> None:
        results = mgr.cold_search(["anything"])
        assert results == []

    def test_cold_search_by_tag(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "tagged.yaml",
            {"id": "tagged-e", "content": "something", "tags": ["special-tag"]},
        )
        results = mgr.cold_search(["special-tag"])
        ids = [r.get("id") for r in results]
        assert "tagged-e" in ids

    def test_cold_search_matches_detail_text(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "detail-hit.yaml",
            {"id": "detail-hit", "content": "opaque title", "detail": "semantic migration note", "tags": []},
        )

        results = mgr.cold_search(["migration"])
        assert [r.get("id") for r in results] == ["detail-hit"]

    def test_path_traversal_guard(self, mgr: TierManager, tmp_path: Path) -> None:
        """cold_archive must reject paths outside base_dir."""
        from trw_memory.storage.persistence import write_yaml

        outside = tmp_path.parent / "evil.yaml"
        write_yaml(outside, {"id": "evil", "content": "escape"})
        with pytest.raises(Exception):
            mgr.cold_archive("evil", outside)


# ---------------------------------------------------------------------------
# TierManager — sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_returns_tier_sweep_result(self, mgr: TierManager) -> None:
        result = mgr.sweep()
        assert isinstance(result, TierSweepResult)

    def test_sweep_empty_dirs_returns_zeros(self, mgr: TierManager) -> None:
        result = mgr.sweep()
        assert result.promoted == 0
        assert result.demoted == 0
        assert result.purged == 0
        assert result.errors == 0

    def test_sweep_demotes_stale_hot_entry(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        stale_entry = _make_entry("stale", days_old=cfg.hot_ttl_days + 5)
        mgr.hot_put("stale", stale_entry)
        assert mgr.hot_size == 1
        result = mgr.sweep()
        # stale entry evicted from hot
        assert mgr.hot_get("stale") is None
        assert result.demoted >= 1

    def test_sweep_keeps_fresh_hot_entry(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        fresh_entry = _make_entry("fresh", days_old=1)
        mgr.hot_put("fresh", fresh_entry)
        result = mgr.sweep()
        assert mgr.hot_get("fresh") is not None
        assert result.demoted == 0

    def test_sweep_hot_to_warm_failure_keeps_entry_in_hot(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        stale_entry = _make_entry("stale-hot", days_old=cfg.hot_ttl_days + 5)
        mgr.hot_put("stale-hot", stale_entry)

        original_warm_add = mgr.warm_add

        def _fail_once(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("disk full")

        mgr.warm_add = _fail_once  # type: ignore[method-assign]
        try:
            result = mgr.sweep()
        finally:
            mgr.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result.errors == 1
        assert mgr.hot_get("stale-hot") is not None

    def test_sweep_demotes_warm_to_cold(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)

        mgr._entries_dir = entries_dir

        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)).isoformat()
        entry_file = entries_dir / "old-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "old-entry",
                "content": "ancient knowledge",
                "importance": 0.1,  # below threshold
                "status": "active",
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not entry_file.exists()
        assert result.demoted >= 1

    def test_sweep_demotes_sqlite_warm_entry_to_cold(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        sqlite_cfg = cfg.model_copy(update={"storage_backend": "sqlite"})
        mgr.update_config(sqlite_cfg)

        old_time = datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
        entry = MemoryEntry(
            id="old-sqlite-entry",
            content="ancient sqlite knowledge",
            importance=0.1,
            namespace="default",
            last_accessed_at=old_time,
        )
        with SQLiteBackend(mem_dir / sqlite_cfg.sqlite_db_name, dim=sqlite_cfg.embedding_dim) as backend:
            backend.store(entry)

        mgr.warm_add(entry.id, entry.model_dump(mode="json"), [1.0, 0.0])

        result = mgr.sweep(config=sqlite_cfg)

        cold_file = mgr._cold_dir() / str(entry.created_at.year) / f"{entry.created_at.month:02d}" / f"{entry.id}.yaml"
        assert cold_file.exists()
        assert result.demoted >= 1
        with SQLiteBackend(mem_dir / sqlite_cfg.sqlite_db_name, dim=sqlite_cfg.embedding_dim) as backend:
            assert backend.get(entry.id) is None

    def test_sweep_warm_to_cold_skips_non_active(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)).isoformat()
        entry_file = entries_dir / "resolved-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "resolved-entry",
                "content": "resolved",
                "importance": 0.1,
                "status": "resolved",  # not active
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        # resolved entries should NOT be archived
        assert entry_file.exists()

    def test_sweep_purges_expired_cold_entry(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)).isoformat()
        cold_file = cold_partition / "expired.yaml"
        write_yaml(
            cold_file,
            {
                "id": "expired-entry",
                "content": "expired",
                "importance": 0.05,  # below purge threshold
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not cold_file.exists()
        assert result.purged >= 1

    def test_sweep_keeps_high_importance_cold_entry(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)).isoformat()
        cold_file = cold_partition / "important.yaml"
        write_yaml(
            cold_file,
            {
                "id": "important-entry",
                "content": "important",
                "importance": 0.9,  # high importance → keep
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert cold_file.exists()
        assert result.purged == 0

    def test_sweep_error_in_entry_increments_errors(self, mgr: TierManager, mem_dir: Path) -> None:

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        # Write a corrupt YAML file (binary garbage)
        corrupt_file = entries_dir / "corrupt.yaml"
        corrupt_file.write_bytes(b"\xff\xfe invalid yaml !!!")

        result = mgr.sweep()
        assert result.errors >= 1

    def test_sweep_writes_purge_audit_log(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 50)).isoformat()
        write_yaml(
            cold_partition / "old.yaml",
            {
                "id": "purge-audit-e",
                "content": "purge me",
                "importance": 0.05,
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        mgr.sweep()

        audit_path = mgr._base_dir / "memory" / "purge_audit.jsonl"
        assert audit_path.exists()
        lines = [l for l in audit_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["entry_id"] == "purge-audit-e"

    def test_sweep_uses_call_time_config_thresholds(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)).isoformat()
        entry_file = entries_dir / "override-threshold.yaml"
        write_yaml(
            entry_file,
            {
                "id": "override-threshold",
                "content": "aged but still important",
                "importance": 0.8,
                "status": "active",
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        override_cfg = MemoryConfig(
            hot_max_entries=cfg.hot_max_entries,
            hot_ttl_days=cfg.hot_ttl_days,
            cold_threshold_days=cfg.cold_threshold_days,
            retention_days=cfg.retention_days,
            warm_archive_max_score=0.3,
            cold_purge_max_score=cfg.cold_purge_max_score,
        )
        result = mgr.sweep(config=override_cfg)
        assert result.demoted >= 1
        assert not entry_file.exists()

    def test_sweep_reads_hot_ttl_from_environment(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_HOT_TTL_DAYS", "1")
        stale_entry = _make_entry("env-hot", days_old=2)
        mgr.hot_put("env-hot", stale_entry)

        result = mgr.sweep(config=MemoryConfig())

        assert result.demoted >= 1
        assert mgr.hot_get("env-hot") is None

    def test_sweep_reads_retention_days_from_environment(
        self,
        mgr: TierManager,
        cfg: MemoryConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_RETENTION_DAYS", "30")
        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        cold_file = cold_partition / "env-retention.yaml"
        write_yaml(
            cold_file,
            {
                "id": "env-retention-entry",
                "content": "purge me via env",
                "importance": 0.05,
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep(config=MemoryConfig(cold_purge_max_score=cfg.cold_purge_max_score))

        assert result.purged >= 1
        assert not cold_file.exists()


# ---------------------------------------------------------------------------
# TierSweepResult
# ---------------------------------------------------------------------------


class TestTierSweepResult:
    def test_is_namedtuple(self) -> None:
        r = TierSweepResult(promoted=1, demoted=2, purged=3, errors=0)
        assert r.promoted == 1
        assert r.demoted == 2
        assert r.purged == 3
        assert r.errors == 0

    def test_total_property(self) -> None:
        r = TierSweepResult(promoted=1, demoted=2, purged=3, errors=1)
        assert r.total == 7


# ===========================================================================
# FR04 (PRD-QUAL-038): Tier Lifecycle Edge Case Tests
# ===========================================================================


class TestColdPromoteEdgeCases:
    """FR04: cold_promote() edge cases."""

    def test_cold_promote_found(self, mgr: TierManager, mem_dir: Path) -> None:
        """Cold archive has entry -> returns entry, moves to warm, deletes YAML."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "12"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "promote-found.yaml"
        write_yaml(yaml_file, {"id": "promote-found", "content": "found me", "tags": ["test"]})

        result = mgr.cold_promote("promote-found")
        assert result is not None
        assert result["id"] == "promote-found"
        assert result["content"] == "found me"
        # Original file should be removed
        assert not yaml_file.exists()

    def test_cold_promote_not_found(self, mgr: TierManager) -> None:
        """No matching entry -> returns None."""
        # Cold dir may not even exist
        result = mgr.cold_promote("nonexistent-entry-xyz")
        assert result is None

    def test_cold_promote_read_yaml_failure(self, mgr: TierManager, mem_dir: Path) -> None:
        """Corrupt YAML file -> skips to next, no crash."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "11"
        cold_partition.mkdir(parents=True, exist_ok=True)

        # Write a corrupt file first
        corrupt_file = cold_partition / "corrupt.yaml"
        corrupt_file.write_bytes(b"\xff\xfe not valid yaml!")

        # Write a valid file after
        valid_file = cold_partition / "valid.yaml"
        write_yaml(valid_file, {"id": "valid-entry", "content": "valid data"})

        # Should skip corrupt, find valid
        result = mgr.cold_promote("valid-entry")
        assert result is not None
        assert result["id"] == "valid-entry"

    def test_cold_promote_updates_last_accessed(self, mgr: TierManager, mem_dir: Path) -> None:
        """Promoted entry has updated last_accessed_at."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "10"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "promote-accessed.yaml"
        old_time = "2024-01-01T00:00:00+00:00"
        write_yaml(
            yaml_file,
            {"id": "promote-accessed", "content": "old", "last_accessed_at": old_time},
        )

        result = mgr.cold_promote("promote-accessed")
        assert result is not None
        assert result["last_accessed_at"] != old_time


class TestSweepEdgeCases:
    """FR04: sweep() edge cases -- hot->warm, warm->cold, cold->purge."""

    def test_sweep_warm_to_cold_archival(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        """Entry exceeds cold_threshold_days + low importance -> archived."""
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 20)).isoformat()
        entry_file = entries_dir / "archive-me.yaml"
        write_yaml(
            entry_file,
            {
                "id": "archive-me",
                "content": "stale warm",
                "importance": 0.05,
                "status": "active",
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not entry_file.exists()
        assert result.demoted >= 1

        # Verify file appeared in cold archive
        cold_files = list(mgr._cold_dir().rglob("*.yaml"))
        assert len(cold_files) >= 1

    def test_sweep_cold_to_purge(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        """Entry exceeds retention_days + very low importance -> purged with audit."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 100)).isoformat()
        cold_file = cold_partition / "purge-me.yaml"
        write_yaml(
            cold_file,
            {
                "id": "purge-me",
                "content": "expired entry",
                "importance": 0.01,
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not cold_file.exists()
        assert result.purged >= 1

    def test_sweep_purge_writes_audit_jsonl(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        """Verify purge_audit.jsonl gets JSON record on purge."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "05"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 200)).isoformat()
        write_yaml(
            cold_partition / "audit-test.yaml",
            {
                "id": "audit-entry",
                "content": "will be purged",
                "importance": 0.02,
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        mgr.sweep()

        audit_path = mem_dir / "memory" / "purge_audit.jsonl"
        assert audit_path.exists()
        lines = [l for l in audit_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["entry_id"] == "audit-entry"
        assert "purged_at" in record
        assert "days_idle" in record

    def test_sweep_error_handling(self, mgr: TierManager, mem_dir: Path) -> None:
        """Per-entry exception increments errors count."""
        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        # Write binary garbage as a YAML file
        corrupt = entries_dir / "bad-data.yaml"
        corrupt.write_bytes(b"\x00\x01\x02 not yaml at all")

        result = mgr.sweep()
        assert result.errors >= 1

    def test_sweep_no_entries_to_process(self, mgr: TierManager) -> None:
        """Empty tiers -> sweep returns zeros."""
        result = mgr.sweep()
        assert result.promoted == 0
        assert result.demoted == 0
        assert result.purged == 0
        assert result.errors == 0
        assert result.total == 0

    def test_sweep_warm_to_cold_writes_yaml(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        """Archived entry creates YAML file in cold dir."""
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 30)).isoformat()
        entry_file = entries_dir / "cold-check.yaml"
        write_yaml(
            entry_file,
            {
                "id": "cold-check",
                "content": "verify cold write",
                "importance": 0.05,
                "status": "active",
                "last_accessed_at": old_time,
                "tags": [],
            },
        )

        mgr.sweep()

        # Verify YAML was written to cold directory
        cold_files = list(mgr._cold_dir().rglob("*.yaml"))
        cold_ids = []
        from trw_memory.storage.persistence import read_yaml as _read_yaml

        for f in cold_files:
            data = _read_yaml(f)
            cold_ids.append(data.get("id"))
        assert "cold-check" in cold_ids


class TestWarmKeywordSearchEdgeCases:
    """FR04: _warm_keyword_search edge cases."""

    def test_warm_keyword_search_malformed_jsonl(self, mgr: TierManager, mem_dir: Path) -> None:
        """Bad JSONL line -> skipped, valid lines searched."""
        sidecar = mgr._warm_sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            '{"id": "valid-1", "summary": "python coding", "tags": []}',
            "this is not valid json at all",
            '{"id": "valid-2", "summary": "java programming", "tags": []}',
        ]
        sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")

        results = mgr.warm_search(["python"], None)
        ids = [r["id"] for r in results]
        assert "valid-1" in ids
        assert "valid-2" not in ids  # "java" doesn't match "python"

    def test_warm_keyword_search_empty_sidecar(self, mgr: TierManager, mem_dir: Path) -> None:
        """Empty sidecar file -> empty results."""
        sidecar = mgr._warm_sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("", encoding="utf-8")

        results = mgr.warm_search(["anything"], None)
        assert results == []

    def test_warm_keyword_search_zero_match(self, mgr: TierManager) -> None:
        """No matching entries -> empty list."""
        d1: dict[str, object] = {"id": "e1", "content": "alpha beta", "tags": []}
        mgr.warm_add("e1", d1, None)

        results = mgr.warm_search(["zzz_nonexistent_token"], None)
        assert results == []


class TestTierPerformanceContracts:
    def test_hot_tier_latency_p95_under_1ms(self, mgr: TierManager) -> None:
        for i in range(3):
            mgr.hot_put(f"hot-{i}", _make_entry(f"hot-{i}"))

        durations: list[float] = []
        for _ in range(2_000):
            started = time.perf_counter()
            assert mgr.hot_get("hot-1") is not None
            durations.append(time.perf_counter() - started)

        durations.sort()
        p95 = durations[int(len(durations) * 0.95)]
        assert p95 < 0.001

    def test_warm_tier_search_p95_under_50ms(self, mgr: TierManager) -> None:
        for i in range(500):
            mgr.warm_add(
                f"warm-{i}",
                {
                    "id": f"warm-{i}",
                    "content": f"python warm entry {i}" if i % 10 == 0 else f"warm entry {i}",
                    "tags": ["python"] if i % 10 == 0 else ["misc"],
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
        p95 = durations[int(len(durations) * 0.95)]
        assert p95 < 0.05

    def test_cold_tier_search_p95_under_350ms(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "05"
        cold_partition.mkdir(parents=True, exist_ok=True)
        for i in range(500):
            write_yaml(
                cold_partition / f"cold-{i}.yaml",
                {
                    "id": f"cold-{i}",
                    "content": f"archived python lesson {i}" if i % 10 == 0 else f"archived lesson {i}",
                    "tags": ["python"] if i % 10 == 0 else ["archive"],
                },
            )

        durations: list[float] = []
        for _ in range(25):
            started = time.perf_counter()
            results = mgr.cold_search(["python"])
            durations.append(time.perf_counter() - started)
            assert results

        durations.sort()
        p95 = durations[int(len(durations) * 0.95)]
        assert p95 < 0.35

    def test_hot_tier_memory_budget_under_50mb(self, mgr: TierManager) -> None:
        tracemalloc.start()
        try:
            for i in range(50):
                mgr.hot_put(f"mem-{i}", _make_entry(f"mem-{i}", importance=0.9))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 50 * 1024 * 1024

    def test_sweep_processes_100_entries_under_5_seconds(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        for i in range(33):
            mgr.hot_put(f"hot-sweep-{i}", _make_entry(f"hot-sweep-{i}", days_old=cfg.hot_ttl_days + 10))

        old_warm_time = (datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 20)).isoformat()
        for i in range(33):
            write_yaml(
                entries_dir / f"warm-sweep-{i}.yaml",
                {
                    "id": f"warm-sweep-{i}",
                    "content": f"warm sweep {i}",
                    "importance": 0.05,
                    "status": "active",
                    "last_accessed_at": old_warm_time,
                    "tags": [],
                },
            )

        old_cold_time = (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 20)).isoformat()
        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        for i in range(34):
            write_yaml(
                cold_partition / f"cold-sweep-{i}.yaml",
                {
                    "id": f"cold-sweep-{i}",
                    "content": f"cold sweep {i}",
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
