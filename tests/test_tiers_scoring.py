"""Tests for lifecycle/tiers.py scoring utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trw_memory.lifecycle.tiers import TierSweepResult, compute_importance_score
from trw_memory.models.config import MemoryConfig


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
        s_no_match = compute_importance_score(no_match, ["foo", "bar"], config=cfg)
        assert s_match > s_no_match

    def test_cosine_similarity_used_when_embeddings_provided(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {"content": "irrelevant", "importance": 0.5}
        score = compute_importance_score(
            entry,
            [],
            query_embedding=[1.0, 0.0],
            entry_embedding=[1.0, 0.0],
            config=cfg,
        )
        assert score > 0.3

    def test_orthogonal_embeddings_zero_relevance(self) -> None:
        cfg = MemoryConfig()
        entry: dict[str, object] = {"content": "irrelevant", "importance": 0.0}
        score = compute_importance_score(
            entry,
            [],
            query_embedding=[1.0, 0.0],
            entry_embedding=[0.0, 1.0],
            config=cfg,
        )
        assert score < 0.5

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
        cfg = MemoryConfig()
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


class TestTierSweepResult:
    def test_is_namedtuple(self) -> None:
        result = TierSweepResult(promoted=1, demoted=2, purged=3, errors=0)
        assert result.promoted == 1
        assert result.demoted == 2
        assert result.purged == 3
        assert result.errors == 0

    def test_total_property(self) -> None:
        result = TierSweepResult(promoted=1, demoted=2, purged=3, errors=1)
        assert result.total == 7
