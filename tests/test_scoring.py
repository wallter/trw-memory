"""Tests for lifecycle/scoring.py.

Covers:
- clamp01, ensure_utc helpers
- update_q_value: EMA formula, clamping, recurrence bonus
- compute_utility_score: cold-start blending, Ebbinghaus decay, access boost,
  source boost, clamp behaviour
- apply_time_decay: linear decay, 0.3 floor, naive/aware datetime handling
- bayesian_calibrate: weighted average, org_weight cap
- enforce_tier_distribution: demotion logic, small-set no-op
- rank_by_utility: relevance+utility blending, wildcard query
- entry_utility: field extraction from MemoryEntry dict
- utility_based_prune_candidates: three tiers
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import pairwise

import pytest

from trw_memory.lifecycle.scoring import (
    apply_time_decay,
    bayesian_calibrate,
    compute_calibration_accuracy,
    compute_utility_score,
    enforce_tier_distribution,
    entry_utility,
    rank_by_utility,
    update_q_value,
    utility_based_prune_candidates,
)
from trw_memory.models.config import MemoryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _entry(
    *,
    importance: float = 0.5,
    q_value: float = 0.5,
    q_observations: int = 0,
    recurrence: int = 1,
    access_count: int = 0,
    source: str = "agent",
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    status: str = "active",
    entry_id: str = "M-001",
) -> dict[str, object]:
    """Build a minimal entry dict matching MemoryEntry serialised shape."""
    now = _now_utc()
    return {
        "id": entry_id,
        "content": "test content",
        "importance": importance,
        "q_value": q_value,
        "q_observations": q_observations,
        "recurrence": recurrence,
        "access_count": access_count,
        "source": source,
        "created_at": (created_at or now).isoformat(),
        "last_accessed_at": last_accessed_at.isoformat() if last_accessed_at else None,
        "status": status,
    }


# ---------------------------------------------------------------------------
# update_q_value
# ---------------------------------------------------------------------------


def test_update_q_value_basic() -> None:
    q = update_q_value(0.5, 0.8, alpha=0.15)
    expected = 0.5 + 0.15 * (0.8 - 0.5)
    assert abs(q - expected) < 1e-9


def test_update_q_value_clamp_upper() -> None:
    # 0.0 + 0.9*(1.0 - 0.0) = 0.9 — use q_old=0.0 to guarantee hitting the cap path
    # Test clamping via recurrence_bonus pushing above 1.0
    q = update_q_value(0.99, 1.0, alpha=1.0)
    assert q == 1.0


def test_update_q_value_clamp_lower() -> None:
    q = update_q_value(0.01, -1.0, alpha=0.9)
    assert q == 0.0


def test_update_q_value_recurrence_bonus() -> None:
    q = update_q_value(0.5, 0.5, alpha=0.15, recurrence_bonus=0.05)
    # No shift from alpha (reward == q_old), but bonus adds
    assert abs(q - 0.55) < 1e-9


def test_update_q_value_zero_alpha() -> None:
    q = update_q_value(0.5, 0.8, alpha=0.0)
    assert q == 0.5


# ---------------------------------------------------------------------------
# apply_time_decay
# ---------------------------------------------------------------------------


def test_apply_time_decay_brand_new() -> None:
    now = _now_utc()
    result = apply_time_decay(0.8, now)
    # 0 days → decay_factor = 1.0
    assert abs(result - 0.8) < 0.01


def test_apply_time_decay_one_year() -> None:
    old = _now_utc() - timedelta(days=365)
    result = apply_time_decay(1.0, old)
    # decay_factor = max(0.3, 1 - 1.0 * 0.3) = 0.7
    assert abs(result - 0.7) < 0.02


def test_apply_time_decay_floor() -> None:
    very_old = _now_utc() - timedelta(days=3650)
    result = apply_time_decay(1.0, very_old)
    # decay_factor floored at 0.3
    assert result >= 0.3


def test_apply_time_decay_naive_datetime() -> None:
    naive = datetime.now(tz=timezone.utc).replace(tzinfo=None)  # no tzinfo
    result = apply_time_decay(0.9, naive)
    assert 0.0 <= result <= 1.0


def test_apply_time_decay_clamps_output() -> None:
    now = _now_utc()
    result = apply_time_decay(0.0, now)
    assert result == 0.0


# ---------------------------------------------------------------------------
# compute_utility_score
# ---------------------------------------------------------------------------


def test_compute_utility_score_cold_start_uses_impact() -> None:
    # q_observations=0 → fully trust base_impact (w=0)
    score = compute_utility_score(
        q_value=0.0,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.8,
        q_observations=0,
        cold_start_threshold=3,
    )
    # With 0 days decay and recurrence 1, retention ~1.0
    assert score > 0.5


def test_compute_utility_score_high_q_after_cold_start() -> None:
    score = compute_utility_score(
        q_value=0.9,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.5,
        q_observations=5,
        cold_start_threshold=3,
    )
    assert score > 0.7


def test_compute_utility_score_decay_reduces_score() -> None:
    fresh = compute_utility_score(
        q_value=0.8,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.8,
        q_observations=10,
    )
    old = compute_utility_score(
        q_value=0.8,
        days_since_last_access=100,
        recurrence_count=1,
        base_impact=0.8,
        q_observations=10,
    )
    assert fresh > old


def test_compute_utility_score_access_boost() -> None:
    without_boost = compute_utility_score(
        q_value=0.5,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.5,
        q_observations=5,
        access_count=0,
    )
    with_boost = compute_utility_score(
        q_value=0.5,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.5,
        q_observations=5,
        access_count=100,
    )
    assert with_boost > without_boost


def test_compute_utility_score_human_source_boost() -> None:
    agent_score = compute_utility_score(
        q_value=0.5,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.5,
        q_observations=5,
        source_type="agent",
    )
    human_score = compute_utility_score(
        q_value=0.5,
        days_since_last_access=0,
        recurrence_count=1,
        base_impact=0.5,
        q_observations=5,
        source_type="human",
    )
    assert human_score > agent_score


def test_compute_utility_score_clamped() -> None:
    score = compute_utility_score(
        q_value=1.0,
        days_since_last_access=0,
        recurrence_count=100,
        base_impact=1.0,
        q_observations=100,
        access_count=1000,
        source_type="human",
    )
    assert 0.0 <= score <= 1.0


def test_compute_utility_score_minimum_zero() -> None:
    score = compute_utility_score(
        q_value=0.0,
        days_since_last_access=10000,
        recurrence_count=1,
        base_impact=0.0,
        q_observations=10,
    )
    assert score >= 0.0


# ---------------------------------------------------------------------------
# bayesian_calibrate
# ---------------------------------------------------------------------------


def test_bayesian_calibrate_basic() -> None:
    result = bayesian_calibrate(0.8, org_mean=0.5, user_weight=1.0, org_weight=0.5)
    expected = (0.8 * 1.0 + 0.5 * 0.5) / (1.0 + 0.5)
    assert abs(result - expected) < 1e-9


def test_bayesian_calibrate_org_weight_cap() -> None:
    # org_weight > 2.0 should be capped at 2.0
    uncapped = bayesian_calibrate(0.9, org_mean=0.5, user_weight=1.0, org_weight=2.0)
    capped = bayesian_calibrate(0.9, org_mean=0.5, user_weight=1.0, org_weight=10.0)
    assert abs(uncapped - capped) < 1e-9


def test_bayesian_calibrate_zero_weights_returns_user_impact() -> None:
    result = bayesian_calibrate(0.7, org_mean=0.5, user_weight=0.0, org_weight=0.0)
    assert result == 0.7


def test_bayesian_calibrate_clamp() -> None:
    result = bayesian_calibrate(1.0, org_mean=1.0, user_weight=100.0, org_weight=0.0)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_calibration_accuracy
# (restored release-verify 2026-07-17 P1: b6237f80d9 deleted these as "unused",
#  but b918df2f19 restored the function — it is a load-bearing cross-package
#  import consumed by trw-mcp _learning_helpers — without re-adding the tests.)
# ---------------------------------------------------------------------------


def test_calibration_accuracy_no_recalls() -> None:
    assert compute_calibration_accuracy({"total_recalls": 0, "positive_outcomes": 0}) == 1.0


def test_calibration_accuracy_high() -> None:
    acc = compute_calibration_accuracy({"total_recalls": 100, "positive_outcomes": 80})
    assert acc == 2.0  # >= 75% positive


def test_calibration_accuracy_medium_high() -> None:
    acc = compute_calibration_accuracy({"total_recalls": 100, "positive_outcomes": 60})
    assert acc == 1.5  # >= 50%


def test_calibration_accuracy_medium() -> None:
    acc = compute_calibration_accuracy({"total_recalls": 100, "positive_outcomes": 30})
    assert acc == 1.0  # >= 25%


def test_calibration_accuracy_low() -> None:
    acc = compute_calibration_accuracy({"total_recalls": 100, "positive_outcomes": 10})
    assert acc == 0.5  # < 25%


# ---------------------------------------------------------------------------
# enforce_tier_distribution
# ---------------------------------------------------------------------------


def test_enforce_tier_no_op_below_minimum() -> None:
    # < 5 entries — no enforcement
    entries = [("M-001", 0.95), ("M-002", 0.91), ("M-003", 0.75)]
    result = enforce_tier_distribution(entries)
    assert result == []


def test_enforce_tier_no_op_when_within_caps() -> None:
    # 10 entries, 1 critical (10%), cap is typically 0.05 but none exceed with <=1
    entries = [
        ("M-001", 0.95),  # critical
        ("M-002", 0.75),  # high
        ("M-003", 0.65),  # medium
        ("M-004", 0.60),  # medium
        ("M-005", 0.55),  # medium
    ]
    # 1/5 critical = 0.2, but default cap is 0.05 — expect demotion
    result = enforce_tier_distribution(entries, critical_cap=0.3, high_cap=0.5)
    assert result == []


def test_enforce_tier_demotes_critical() -> None:
    entries = [
        ("M-001", 0.98),
        ("M-002", 0.95),
        ("M-003", 0.91),  # 3/5 = 60% critical, cap is 0.05 → demote lowest
        ("M-004", 0.65),
        ("M-005", 0.50),
    ]
    result = enforce_tier_distribution(entries, critical_cap=0.05, high_cap=0.5)
    assert len(result) >= 1
    demoted_id = result[0][0]
    new_score = result[0][1]
    assert new_score < 0.9  # moved out of critical tier
    assert demoted_id in {"M-001", "M-002", "M-003"}


def test_enforce_tier_demotes_high() -> None:
    entries = [
        ("M-001", 0.50),
        ("M-002", 0.75),
        ("M-003", 0.78),
        ("M-004", 0.72),  # 3/4 of remaining = high, cap exceeded
        ("M-005", 0.60),
    ]
    result = enforce_tier_distribution(entries, critical_cap=0.5, high_cap=0.1)
    assert any(new_score < 0.7 for _, new_score in result)


def test_enforce_tier_empty_input() -> None:
    assert enforce_tier_distribution([]) == []


# ---------------------------------------------------------------------------
# S12: caps are config-driven (no drift vs trw-mcp)
# ---------------------------------------------------------------------------


def _critical_heavy_entries() -> list[tuple[str, float]]:
    # 3/5 = 60% critical — over the default 5% cap, under a custom 70% cap.
    return [
        ("M-001", 0.98),
        ("M-002", 0.95),
        ("M-003", 0.91),
        ("M-004", 0.60),
        ("M-005", 0.50),
    ]


def test_enforce_tier_caps_resolve_from_config() -> None:
    # FAILS before fix: with caps hardcoded to 0.05, a config raising the cap
    # to 0.70 would be ignored and a demotion would still occur.
    entries = _critical_heavy_entries()
    cfg = MemoryConfig(impact_tier_critical_cap=0.7, impact_tier_high_cap=0.9)
    result = enforce_tier_distribution(entries, config=cfg)
    assert result == []  # 60% critical is within the config's 70% cap


def test_enforce_tier_explicit_cap_overrides_config() -> None:
    # Explicit cap must win over the config value.
    entries = _critical_heavy_entries()
    cfg = MemoryConfig(impact_tier_critical_cap=0.7)
    result = enforce_tier_distribution(entries, critical_cap=0.05, config=cfg)
    assert len(result) >= 1  # tight explicit cap forces a demotion


def test_enforce_tier_default_config_matches_legacy_caps() -> None:
    # No behaviour change when caps unspecified: default config == old literals.
    entries = _critical_heavy_entries()
    assert enforce_tier_distribution(entries) != []  # default 0.05 cap exceeded


# ---------------------------------------------------------------------------
# PRD-CORE-251 FR02: converge_tier_distribution and persist_tier_convergence
# were RETIRED (2026-09-03). Both had zero production consumers in either
# package; the five test functions that pinned them
# (test_converge_brings_critical_within_cap, test_converge_no_op_within_caps,
# test_converge_empty_input, test_persist_tier_convergence_atomic,
# test_persist_tier_convergence_no_op_within_caps) were removed with them.
# test_single_enforce_leaves_cluster_over_cap survives below because it
# exercises enforce_tier_distribution directly, which FR08 keeps.
# ---------------------------------------------------------------------------


def _over_cap_critical_cluster() -> list[tuple[str, float]]:
    # 8 critical out of 10 (80%); default critical cap is 5% (=> max 0 allowed
    # by the >cap rule until count/total <= 0.05, i.e. 0 of 10 may stay critical).
    return [
        ("M-001", 0.99),
        ("M-002", 0.98),
        ("M-003", 0.97),
        ("M-004", 0.96),
        ("M-005", 0.95),
        ("M-006", 0.94),
        ("M-007", 0.93),
        ("M-008", 0.92),
        ("M-009", 0.40),
        ("M-010", 0.30),
    ]


def test_single_enforce_leaves_cluster_over_cap() -> None:
    # A single enforce call demotes at most one entry per tier, so an 8-over-cap
    # cluster is STILL over the critical cap afterwards. That is the documented
    # contract of the function, not a defect: the caller re-invokes until the
    # returned list is empty.
    entries = _over_cap_critical_cluster()
    one_step = enforce_tier_distribution(entries)
    # only one critical demotion in a single call
    assert len(one_step) <= 2  # at most one critical + one cascaded high
    # apply the step and recount critical members
    moved = dict(one_step)
    remaining_critical = sum(1 for mid, sc in entries if moved.get(mid, sc) >= 0.9)
    assert remaining_critical / len(entries) > 0.05  # STILL over cap


def test_retired_symbols_are_absent() -> None:
    """FR02 — the two retired functions are gone from source AND from the exports.

    An import that still resolves would mean the deletion only reached the
    ``__all__`` list, leaving the body reachable by anyone who imports the
    module directly.
    """
    import trw_memory.lifecycle as lifecycle
    from trw_memory.lifecycle import scoring

    for name in ("converge_tier_distribution", "persist_tier_convergence"):
        assert not hasattr(scoring, name), f"{name} is still defined in trw_memory.lifecycle.scoring"
        assert not hasattr(lifecycle, name), f"{name} is still exported by trw_memory.lifecycle"
        assert name not in lifecycle.__all__, f"{name} is still listed in trw_memory.lifecycle.__all__"


# ---------------------------------------------------------------------------
# Decay correctness: bounded, monotonic, no NaN at extreme ages
# ---------------------------------------------------------------------------


def test_apply_time_decay_bounded_and_monotonic_over_ages() -> None:
    now = _now_utc()
    ages = [0, 1, 30, 100, 365, 1000, 5000, 100_000]
    results = [apply_time_decay(1.0, now - timedelta(days=a)) for a in ages]
    # bounded [0,1], finite (no NaN/inf)
    for r in results:
        assert 0.0 <= r <= 1.0
        assert r == r  # NaN != NaN
    # non-increasing with age (monotonic decay)
    for earlier, later in pairwise(results):
        assert later <= earlier + 1e-9


def test_apply_time_decay_zero_age_not_penalized() -> None:
    # A brand-new entry keeps (essentially) its full impact.
    assert apply_time_decay(0.8, _now_utc()) >= 0.8 - 1e-6


def test_apply_time_decay_future_timestamp_not_penalized() -> None:
    # Clock skew: a future created_at clamps days to 0 (no over-1.0, no NaN).
    future = _now_utc() + timedelta(days=10)
    r = apply_time_decay(0.9, future)
    assert 0.0 <= r <= 1.0
    assert abs(r - 0.9) < 1e-6


def test_compute_utility_score_extreme_age_bounded() -> None:
    # Very large age must not produce NaN/negative; retention -> 0.
    score = compute_utility_score(
        q_value=0.9,
        days_since_last_access=10_000_000,
        recurrence_count=1,
        base_impact=0.9,
        q_observations=10,
    )
    assert 0.0 <= score <= 1.0
    assert score == score  # not NaN


# ---------------------------------------------------------------------------
# rank_by_utility
# ---------------------------------------------------------------------------


def test_rank_by_utility_empty() -> None:
    assert rank_by_utility([], ["python"], 0.5) == []


def test_rank_by_utility_compatibility_import_drops_expired_entries() -> None:
    expired = _entry(entry_id="M-expired")
    expired["expires"] = "2020-01-01T00:00:00+00:00"

    assert rank_by_utility([expired], ["python"], 0.5) == []


def test_rank_by_utility_wildcard_all_same_relevance() -> None:
    entries = [_entry(importance=0.9, entry_id="M-001"), _entry(importance=0.2, entry_id="M-002")]
    # With empty query tokens (wildcard), relevance=1.0 for all
    # Higher importance → higher utility → higher rank
    ranked = rank_by_utility(entries, [], 0.9)
    assert ranked[0]["id"] == "M-001"


def test_rank_by_utility_relevance_dominates_at_lambda_zero() -> None:
    # lambda=0 → pure relevance
    relevant = _entry(entry_id="M-001", importance=0.1)
    relevant["content"] = "python async coroutines"
    irrelevant = _entry(entry_id="M-002", importance=0.9)
    irrelevant["content"] = "unrelated topic"

    ranked = rank_by_utility([relevant, irrelevant], ["python"], lambda_weight=0.0)
    assert ranked[0]["id"] == "M-001"


def test_rank_by_utility_utility_dominates_at_lambda_one() -> None:
    # lambda=1 → pure utility, high importance wins
    low_imp = _entry(entry_id="M-001", importance=0.1, q_observations=10, q_value=0.1)
    high_imp = _entry(entry_id="M-002", importance=0.9, q_observations=10, q_value=0.9)

    ranked = rank_by_utility([low_imp, high_imp], ["python"], lambda_weight=1.0)
    assert ranked[0]["id"] == "M-002"


# ---------------------------------------------------------------------------
# entry_utility
# ---------------------------------------------------------------------------


def test_entry_utility_returns_float_in_range() -> None:
    e = _entry(importance=0.6, q_value=0.7, q_observations=5)
    score = entry_utility(e)
    assert 0.0 <= score <= 1.0


def test_entry_utility_fresh_entry_higher_than_old() -> None:
    fresh = _entry(created_at=_now_utc())
    old = _entry(created_at=_now_utc() - timedelta(days=300))
    assert entry_utility(fresh) >= entry_utility(old)


def test_entry_utility_custom_config() -> None:
    e = _entry(importance=0.8, q_value=0.8, q_observations=10)
    cfg = MemoryConfig()
    score = entry_utility(e, config=cfg)
    assert 0.0 <= score <= 1.0


def test_entry_utility_missing_timestamps_uses_defaults() -> None:
    e: dict[str, object] = {
        "id": "M-001",
        "content": "test",
        "importance": 0.5,
        "q_value": 0.5,
        "q_observations": 0,
        "recurrence": 1,
        "access_count": 0,
        "source": "agent",
        "created_at": None,
        "last_accessed_at": None,
        "status": "active",
    }
    score = entry_utility(e)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# utility_based_prune_candidates
# ---------------------------------------------------------------------------


def test_prune_candidates_empty() -> None:
    assert utility_based_prune_candidates([]) == []


def test_prune_candidates_status_cleanup() -> None:
    e = _entry(status="resolved")
    candidates = utility_based_prune_candidates([e])
    assert len(candidates) == 1
    assert candidates[0]["suggested_status"] == "resolved"


def test_prune_candidates_obsolete_status() -> None:
    e = _entry(status="obsolete")
    candidates = utility_based_prune_candidates([e])
    assert len(candidates) == 1
    assert candidates[0]["reason"].startswith("Already marked obsolete")


def test_prune_candidates_very_low_utility() -> None:
    # Entry with near-zero q_value and importance, old timestamp
    very_old = _now_utc() - timedelta(days=500)
    e = _entry(importance=0.01, q_value=0.01, q_observations=10, created_at=very_old)
    cfg = MemoryConfig()
    candidates = utility_based_prune_candidates([e], config=cfg)
    # Should be flagged as delete or prune candidate
    assert len(candidates) >= 1


def test_prune_candidates_active_high_utility_not_flagged() -> None:
    # Fresh entry with high utility should not be a prune candidate
    e = _entry(importance=0.9, q_value=0.9, q_observations=10, created_at=_now_utc())
    candidates = utility_based_prune_candidates([e])
    assert len(candidates) == 0


def test_prune_candidates_deduplicates_by_id() -> None:
    e1 = _entry(status="resolved", entry_id="M-DUP")
    e2 = _entry(status="resolved", entry_id="M-DUP")
    candidates = utility_based_prune_candidates([e1, e2])
    assert len(candidates) == 1


class TestFeedbackDecayFloor:
    """PRD-CORE-244 FR11 residual — a missing signal must not bury an entry.

    ``helpful_count`` is 0 on 100% of the 9,366-row corpus, so the exponent
    ``recall_count / max(1, helpful_count)`` degenerates to ``recall_count`` and
    the term becomes a pure recall-FREQUENCY penalty with no lower bound — on a
    counter PRD-QUAL-032/D1 established is not evidence of use. The floor bounds
    what an ABSENT rating can cost without weakening the rating itself.
    """

    def test_heavily_recalled_unrated_entry_does_not_sink_below_the_floor(self) -> None:
        from trw_memory.lifecycle.scoring import feedback_decay_score

        importance = 0.8
        for recalls in (10, 50, 100, 1000):
            score = feedback_decay_score(importance, recalls, 0)
            assert score >= importance * 0.5 - 1e-9, f"{recalls} recalls sank it to {score}"

    def test_the_floor_is_what_bounds_it_not_the_clamp(self) -> None:
        """Without the floor the same entry collapses to under 1% of importance."""
        from trw_memory.lifecycle.scoring import feedback_decay_score

        assert feedback_decay_score(0.8, 100, 0, min_factor=0.0) < 0.01
        assert feedback_decay_score(0.8, 100, 0) == pytest.approx(0.4)

    def test_helpful_feedback_still_lifts_an_entry_above_its_sibling(self) -> None:
        """The floor must not flatten the signal it bounds."""
        from trw_memory.lifecycle.scoring import feedback_decay_score

        rated = feedback_decay_score(0.8, 40, 20)
        unrated = feedback_decay_score(0.8, 40, 0)
        assert rated > unrated

    def test_a_cold_start_entry_survives_pure_recall_frequency_on_the_live_path(self) -> None:
        """End to end through entry_utility: heavy recall, no ratings, no observations."""
        from datetime import date

        from trw_memory.lifecycle.scoring import entry_utility

        today = date(2026, 9, 3)
        base: dict[str, object] = {
            "id": "L-hot",
            "importance": 0.9,
            "q_value": 0.9,
            "q_observations": 0,
            "recurrence": 1,
            "access_count": 0,
            "helpful_count": 0,
            "created_at": "2026-09-01",
            "last_accessed_at": "2026-09-03",
            "type": "pattern",
            "confidence": "verified",
        }
        surfaced = entry_utility({**base, "recall_count": 200}, today=today)
        never = entry_utility({**base, "recall_count": 0}, today=today)

        # It is allowed to rank lower, but not to be effectively erased.
        assert surfaced >= never * 0.5 - 1e-9
        assert surfaced > 0.4

    def test_zero_floor_restores_the_previous_behaviour_exactly(self) -> None:
        from trw_memory.lifecycle.scoring import feedback_decay_score

        for recalls, helpful in ((5, 0), (5, 5), (100, 3)):
            expected = 0.8 * (0.95 ** (recalls / max(1, helpful)))
            assert feedback_decay_score(0.8, recalls, helpful, min_factor=0.0) == pytest.approx(expected)


class TestNativePruneHonoursProtectionTier:
    """PRD-CORE-244 FR10 on the FIFTH destructive path.

    ``lifecycle._recall.utility_based_prune_candidates`` is publicly exported
    from ``lifecycle`` and ``lifecycle.scoring.utility_based_prune_candidates``
    delegates into it, so it nominated a ``permanent`` entry even after the
    trw-mcp twin was fixed. Found by independent review.
    """

    @staticmethod
    def _worthless(entry_id: str, tier: str, importance: float = 0.01) -> dict[str, object]:
        return {
            "id": entry_id,
            "content": "a forgotten entry",
            "importance": importance,
            "q_value": importance,
            "q_observations": 10,
            "recurrence": 1,
            "access_count": 0,
            "recall_count": 0,
            "helpful_count": 0,
            "status": "active",
            "created_at": "2023-01-01",
            "last_accessed_at": "2023-01-01",
            "protection_tier": tier,
        }

    def _nominated(self, tier: str, **kwargs: object) -> bool:
        from trw_memory.lifecycle import utility_based_prune_candidates

        entry = self._worthless(f"m-{tier}", tier, **kwargs)  # type: ignore[arg-type]
        return any(c["id"] == f"m-{tier}" for c in utility_based_prune_candidates([entry]))

    def test_permanent_entry_is_never_nominated(self) -> None:
        assert self._nominated("permanent") is False

    def test_protected_entry_is_never_nominated(self) -> None:
        assert self._nominated("protected") is False

    def test_the_same_entry_marked_normal_is_nominated(self) -> None:
        """Proves the exemption came from the TIER, not from the fixture."""
        assert self._nominated("normal") is True

    def test_status_tier_cleanup_also_honours_the_exemption(self) -> None:
        """'Already marked obsolete' is still automatic removal."""
        from trw_memory.lifecycle import utility_based_prune_candidates

        permanent = {**self._worthless("m-obs-permanent", "permanent"), "status": "obsolete"}
        normal = {**self._worthless("m-obs-normal", "normal"), "status": "obsolete"}

        ids = {c["id"] for c in utility_based_prune_candidates([permanent, normal])}
        assert ids == {"m-obs-normal"}

    def test_the_scoring_facade_delegate_inherits_the_exemption(self) -> None:
        """lifecycle.scoring.utility_based_prune_candidates delegates into the same code."""
        from trw_memory.lifecycle.scoring import utility_based_prune_candidates as facade

        permanent = self._worthless("m-facade-permanent", "permanent")
        normal = self._worthless("m-facade-normal", "normal")

        ids = {c["id"] for c in facade([permanent, normal])}
        assert ids == {"m-facade-normal"}
