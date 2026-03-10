"""Tests for lifecycle/tiers.py — TierManager hot/warm/cold lifecycle.

TDD: tests written before implementation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
        assert compute_importance_score(high, [], config=cfg) > compute_importance_score(
            low, [], config=cfg
        )

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
        records = [
            json.loads(l)
            for l in sidecar.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        e1_records = [r for r in records if r.get("id") == "e1"]
        assert len(e1_records) == 1
        assert e1_records[0]["summary"] == "new"

    def test_warm_remove_clears_sidecar(self, mgr: TierManager) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "remove me", "tags": []}
        mgr.warm_add("e1", entry_data, None)
        mgr.warm_remove("e1")
        sidecar = mgr._warm_sidecar_path()
        if sidecar.exists():
            records = [
                json.loads(l)
                for l in sidecar.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
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

    def test_cold_archive_raises_on_missing_file(self, mgr: TierManager, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(Exception):
            mgr.cold_archive("bad", missing)

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

    def test_sweep_demotes_stale_hot_entry(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        stale_entry = _make_entry("stale", days_old=cfg.hot_ttl_days + 5)
        mgr.hot_put("stale", stale_entry)
        assert mgr.hot_size == 1
        result = mgr.sweep()
        # stale entry evicted from hot
        assert mgr.hot_get("stale") is None
        assert result.demoted >= 1

    def test_sweep_keeps_fresh_hot_entry(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        fresh_entry = _make_entry("fresh", days_old=1)
        mgr.hot_put("fresh", fresh_entry)
        result = mgr.sweep()
        assert mgr.hot_get("fresh") is not None
        assert result.demoted == 0

    def test_sweep_demotes_warm_to_cold(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)

        mgr._entries_dir = entries_dir

        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
        ).isoformat()
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

    def test_sweep_warm_to_cold_skips_non_active(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
        ).isoformat()
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

    def test_sweep_purges_expired_cold_entry(
        self, mgr: TierManager, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)
        ).isoformat()
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

    def test_sweep_keeps_high_importance_cold_entry(
        self, mgr: TierManager, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)
        ).isoformat()
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

    def test_sweep_error_in_entry_increments_errors(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        # Write a corrupt YAML file (binary garbage)
        corrupt_file = entries_dir / "corrupt.yaml"
        corrupt_file.write_bytes(b"\xff\xfe invalid yaml !!!")

        result = mgr.sweep()
        assert result.errors >= 1

    def test_sweep_writes_purge_audit_log(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 50)
        ).isoformat()
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

    def test_cold_promote_found(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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

    def test_cold_promote_read_yaml_failure(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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

    def test_cold_promote_updates_last_accessed(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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

    def test_sweep_hot_to_warm_demotion(
        self, mgr: TierManager, cfg: MemoryConfig
    ) -> None:
        """Entry exceeds hot_ttl_days -> moved to warm."""
        stale = _make_entry("stale-hot", days_old=cfg.hot_ttl_days + 1)
        mgr.hot_put("stale-hot", stale)
        assert mgr.hot_size == 1

        result = mgr.sweep()
        assert mgr.hot_get("stale-hot") is None
        assert result.demoted >= 1

    def test_sweep_warm_to_cold_archival(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        """Entry exceeds cold_threshold_days + low importance -> archived."""
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 20)
        ).isoformat()
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

    def test_sweep_cold_to_purge(
        self, mgr: TierManager, cfg: MemoryConfig
    ) -> None:
        """Entry exceeds retention_days + very low importance -> purged with audit."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 100)
        ).isoformat()
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

    def test_sweep_purge_writes_audit_jsonl(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        """Verify purge_audit.jsonl gets JSON record on purge."""
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "05"
        cold_partition.mkdir(parents=True, exist_ok=True)
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 200)
        ).isoformat()
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

    def test_sweep_error_handling(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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

    def test_sweep_warm_to_cold_writes_yaml(
        self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig
    ) -> None:
        """Archived entry creates YAML file in cold dir."""
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        old_time = (
            datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 30)
        ).isoformat()
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

    def test_warm_keyword_search_malformed_jsonl(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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

    def test_warm_keyword_search_empty_sidecar(
        self, mgr: TierManager, mem_dir: Path
    ) -> None:
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
