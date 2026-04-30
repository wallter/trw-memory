"""E2E lifecycle and scoring tests for trw-memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import make_entry_dict


class TestDecayScoring:
    """Section 2 of E2E plan: time decay, Q-learning, composite utility."""

    def test_time_decay_floor_at_03(self) -> None:
        """2.1 — apply_time_decay never goes below 0.3 floor."""
        from trw_memory.lifecycle.scoring import apply_time_decay

        now = datetime.now(timezone.utc)
        decay_fresh = apply_time_decay(1.0, now)
        assert decay_fresh >= 0.95

        year_old = now - timedelta(days=365)
        decay_year = apply_time_decay(1.0, year_old)
        assert decay_year >= 0.3
        assert 0.65 <= decay_year <= 0.75

        very_old = now - timedelta(days=1000)
        decay_old = apply_time_decay(1.0, very_old)
        assert decay_old >= 0.3
        assert decay_old == pytest.approx(0.3, abs=0.01)

    def test_q_learning_convergence(self) -> None:
        """2.2 — Q-learning updates converge toward target reward."""
        from trw_memory.lifecycle.scoring import update_q_value

        q_new = update_q_value(q_old=0.5, reward=1.0, alpha=0.15)
        assert q_new == pytest.approx(0.575, abs=0.01)

        q_neg = update_q_value(q_old=0.5, reward=0.0, alpha=0.15)
        assert q_neg == pytest.approx(0.425, abs=0.01)

        q_value = 0.5
        for _ in range(50):
            q_value = update_q_value(q_value, reward=0.8, alpha=0.15)
        assert abs(q_value - 0.8) < 0.05

    def test_composite_utility_score_ordering(self) -> None:
        """2.3 — High-impact recent entries score higher than low-impact old ones."""
        from trw_memory.lifecycle.scoring import compute_utility_score

        score_high = compute_utility_score(
            q_value=0.85,
            days_since_last_access=1,
            recurrence_count=10,
            base_impact=0.9,
            q_observations=10,
            access_count=10,
            source_type="human",
            half_life_days=14.0,
        )
        score_low = compute_utility_score(
            q_value=0.1,
            days_since_last_access=300,
            recurrence_count=1,
            base_impact=0.2,
            q_observations=2,
            access_count=0,
            source_type="agent",
            half_life_days=14.0,
        )

        assert score_high > score_low
        assert score_high > 0.5
        assert score_low < 0.3


class TestThreeTierLifecycle:
    """Section 5 of E2E plan: hot tier LRU, sweep transitions."""

    def test_prune_candidates_tier_classification(self) -> None:
        """2.6 — utility_based_prune_candidates classifies entries into tiers."""
        from trw_memory.lifecycle.scoring import utility_based_prune_candidates

        now = datetime.now(timezone.utc)
        entries = [
            make_entry_dict(
                entry_id="resolved-1",
                content="resolved entry",
                importance=0.5,
                status="resolved",
                created_at=now - timedelta(days=10),
            ),
            make_entry_dict(
                entry_id="low-util-1",
                content="very low utility",
                importance=0.01,
                status="active",
                created_at=now - timedelta(days=200),
                last_accessed_at=now - timedelta(days=200),
            ),
            make_entry_dict(
                entry_id="high-imp-1",
                content="high importance recent",
                importance=0.9,
                status="active",
                created_at=now - timedelta(days=2),
            ),
        ]

        candidates = utility_based_prune_candidates(entries)
        candidate_ids = {str(candidate["id"]) for candidate in candidates}
        assert "resolved-1" in candidate_ids
        assert "low-util-1" in candidate_ids
        assert "high-imp-1" not in candidate_ids

    def test_sweep_resolved_entries_are_candidates(self) -> None:
        """2.7 — Resolved entries are always prune candidates regardless of age."""
        from trw_memory.lifecycle.scoring import utility_based_prune_candidates

        now = datetime.now(timezone.utc)
        entries = [
            make_entry_dict(
                entry_id="resolved-recent",
                content="just resolved",
                importance=0.9,
                status="resolved",
                created_at=now - timedelta(days=1),
            ),
        ]
        candidates = utility_based_prune_candidates(entries)
        assert len(candidates) == 1
        assert candidates[0]["id"] == "resolved-recent"


class TestConfigValidation:
    """Config edge cases from section 10 of the E2E plan."""

    def test_score_weights_must_sum_to_one(self) -> None:
        """10.1 — Score weights summing to != 1.0 raises validation error."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            MemoryConfig(
                score_relevance_weight=0.5,
                score_recency_weight=0.5,
                score_importance_weight=0.5,
            )

    def test_decay_half_life_must_be_positive(self) -> None:
        """10.2 — Negative or zero decay_half_life_days raises validation error."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError):
            MemoryConfig(decay_half_life_days=-1.0)


class TestHumanSourceBoost:
    """Decay scoring: human source entries get utility boost."""

    def test_human_source_scores_higher(self) -> None:
        """2.5 — Human-sourced entries get +0.1 utility boost."""
        from trw_memory.lifecycle.scoring import compute_utility_score

        score_human = compute_utility_score(
            q_value=0.5,
            days_since_last_access=30,
            recurrence_count=1,
            base_impact=0.5,
            q_observations=3,
            access_count=1,
            source_type="human",
            half_life_days=14.0,
        )
        score_agent = compute_utility_score(
            q_value=0.5,
            days_since_last_access=30,
            recurrence_count=1,
            base_impact=0.5,
            q_observations=3,
            access_count=1,
            source_type="agent",
            half_life_days=14.0,
        )

        assert score_human > score_agent
