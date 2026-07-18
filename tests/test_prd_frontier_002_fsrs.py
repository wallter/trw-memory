"""Tests for FSRS-inspired spaced repetition scoring (frontier-002)."""

from __future__ import annotations

import pytest

from trw_memory.lifecycle.scoring import (
    compute_fsrs_utility_score,
    fsrs_retrievability,
)


class TestFsrsRetrieval:
    def test_retrievability_at_zero_days_is_one(self):
        """At t=0, R=1.0 (perfect recall)."""
        r = fsrs_retrievability(0.0, stability=14.0)
        assert r == pytest.approx(1.0)

    def test_retrievability_at_stability_is_approx_0_9(self):
        """At t=S, R~=0.9 (90% target)."""
        r = fsrs_retrievability(14.0, stability=14.0)
        assert r == pytest.approx(0.9, abs=0.01)

    def test_retrievability_decreases_with_time(self):
        """R decreases monotonically with elapsed time."""
        stab = 7.0
        r0 = fsrs_retrievability(0, stab)
        r7 = fsrs_retrievability(7, stab)
        r30 = fsrs_retrievability(30, stab)
        assert r0 > r7 > r30

    def test_retrievability_negative_days_clamped(self):
        """Negative elapsed_days treated as 0."""
        r = fsrs_retrievability(-5.0, stability=10.0)
        assert r == pytest.approx(1.0)

    def test_zero_stability_uses_default(self):
        """stability=0 uses default without ZeroDivisionError."""
        r = fsrs_retrievability(1.0, stability=0.0)
        assert 0.0 < r <= 1.0


class TestComputeFsrsUtility:
    def test_recent_high_importance_entry_scores_high(self):
        """Entry recalled today with high importance -> high utility."""
        score = compute_fsrs_utility_score(importance=0.9, elapsed_days=0.0, recurrence=5)
        assert score > 0.8

    def test_old_low_importance_entry_scores_low(self):
        """Old, rarely-recalled, low-importance entry -> low utility."""
        score = compute_fsrs_utility_score(importance=0.1, elapsed_days=365.0, recurrence=1)
        assert score < 0.3

    def test_utility_in_unit_range(self):
        """Utility score always in [0, 1]."""
        for imp in [0.0, 0.5, 1.0]:
            for days in [0.0, 7.0, 30.0, 365.0]:
                for rec in [1, 5, 20]:
                    score = compute_fsrs_utility_score(imp, days, rec)
                    assert 0.0 <= score <= 1.0, f"imp={imp} days={days} rec={rec} -> {score}"

    def test_more_recalls_higher_utility(self):
        """Same entry recalled more often has higher utility."""
        s1 = compute_fsrs_utility_score(0.7, 7.0, recurrence=1)
        s5 = compute_fsrs_utility_score(0.7, 7.0, recurrence=5)
        assert s5 > s1

    def test_explicit_stability_used(self):
        """When stability is supplied, it's used directly."""
        r_stable = compute_fsrs_utility_score(0.7, elapsed_days=30.0, stability=60.0)
        r_short = compute_fsrs_utility_score(0.7, elapsed_days=30.0, stability=5.0)
        assert r_stable > r_short  # longer stability = better retention at 30 days
