"""Tests for lifecycle/scoring.py.

Covers:
- clamp01, ensure_utc helpers
- update_q_value: EMA formula, clamping, recurrence bonus
- compute_utility_score: cold-start blending, Ebbinghaus decay, access boost,
  source boost, clamp behaviour
- apply_time_decay: linear decay, 0.3 floor, naive/aware datetime handling
- bayesian_calibrate: weighted average, org_weight cap
- compute_calibration_accuracy: ratio tiers
- enforce_tier_distribution: demotion logic, small-set no-op
- rank_by_utility: relevance+utility blending, wildcard query
- entry_utility: field extraction from MemoryEntry dict
- utility_based_prune_candidates: three tiers
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
# rank_by_utility
# ---------------------------------------------------------------------------


def test_rank_by_utility_empty() -> None:
    assert rank_by_utility([], ["python"], 0.5) == []


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
