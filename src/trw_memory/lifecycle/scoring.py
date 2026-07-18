"""Utility-based scoring for the trw-memory lifecycle layer.

Core scoring functions:
- update_q_value: MemRL exponential moving average Q-learning
- compute_utility_score: Ebbinghaus decay + Q-value composite
- apply_time_decay: Linear time decay with 0.3 floor
- bayesian_calibrate: MACLA Bayesian impact calibration
- rank_by_utility: Relevance + utility blending for recall ranking
- utility_based_prune_candidates: Identify stale/low-utility entries

Research basis:
- MemRL Q-values (arXiv:2601.03192)
- Ebbinghaus forgetting curve
- MACLA Bayesian selection (arXiv:2512.18950)

Scoring math is identical to trw-mcp scoring.py — adapted to use MemoryConfig
and MemoryEntry field names (importance vs impact, source vs source_type,
created_at vs created).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.lifecycle._utils import days_since_access as _days_since_access
from trw_memory.models.config import MemoryConfig

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float_field(entry: dict[str, object], key: str, default: float) -> float:
    """Extract a float from an entry dict, coercing through str for safety."""
    return float(str(entry.get(key, default)))


def _int_field(entry: dict[str, object], key: str, default: int) -> int:
    """Extract an int from an entry dict, coercing through str for safety."""
    return int(str(entry.get(key, default)))


def _clamp01(value: float) -> float:
    """Clamp a value to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def _ensure_utc(ts: datetime) -> datetime:
    """Return a timezone-aware datetime, assuming UTC if naive."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


# ---------------------------------------------------------------------------
# Q-learning update
# ---------------------------------------------------------------------------


def update_q_value(
    q_old: float,
    reward: float,
    alpha: float = 0.15,
    recurrence_bonus: float = 0.0,
) -> float:
    """Update Q-value using MemRL exponential moving average.

    Formula: Q_new = Q_old + alpha * (reward - Q_old) + recurrence_bonus

    Args:
        q_old: Current Q-value (0.0-1.0).
        reward: Observed reward in [-1.0, 1.0].
        alpha: Learning rate. Default 0.15.
        recurrence_bonus: Small additive bonus for repeated recall.

    Returns:
        Updated Q-value clamped to [0.0, 1.0].
    """
    q_new = q_old + alpha * (reward - q_old) + recurrence_bonus
    return _clamp01(q_new)


# ---------------------------------------------------------------------------
# Ebbinghaus time decay
# ---------------------------------------------------------------------------


def apply_time_decay(impact: float, created_at: datetime) -> float:
    """Apply linear Ebbinghaus-inspired time decay to an impact score.

    Formula:
        days = (now - created_at).days
        decay_factor = max(0.3, 1.0 - (days / 365) * 0.3)
        effective_impact = impact * decay_factor

    Args:
        impact: Raw impact/importance score (0.0-1.0).
        created_at: Creation timestamp (timezone-aware or naive UTC).

    Returns:
        Decayed impact score in [0.0, 1.0].
    """
    now = datetime.now(timezone.utc)
    created_utc = _ensure_utc(created_at)
    days = max(0, (now - created_utc).days)
    decay_factor = max(0.3, 1.0 - (days / 365) * 0.3)
    return _clamp01(impact * decay_factor)


# ---------------------------------------------------------------------------
# Composite utility score
# ---------------------------------------------------------------------------


def compute_utility_score(
    q_value: float,
    days_since_last_access: int,
    recurrence_count: int,
    base_impact: float,
    q_observations: int,
    *,
    half_life_days: float = 14.0,
    use_exponent: float = 0.6,
    cold_start_threshold: int = 3,
    access_count: int = 0,
    source_type: str = "agent",
    access_count_boost_cap: float = 0.15,
    source_human_boost: float = 0.1,
) -> float:
    """Compute composite utility score combining Q-value with Ebbinghaus decay.

    Formula:
        retention = recurrence_strength * exp(-effective_decay * days)
        effective_q = blend(impact, q_value, q_observations)
        utility = effective_q * retention + access_boost + source_boost

    Args:
        q_value: Current Q-value from outcome tracking (0.0-1.0).
        days_since_last_access: Days since last recall.
        recurrence_count: Number of times recalled (minimum 1).
        base_impact: Original static importance score (0.0-1.0).
        q_observations: Number of outcome observations.
        half_life_days: Days until retention halves. Default 14.
        use_exponent: Sub-linear recurrence exponent. Default 0.6.
        cold_start_threshold: Q-observations before trusting q_value. Default 3.
        access_count: Number of times recalled (for sub-linear boost).
        source_type: 'human' or 'agent'.
        access_count_boost_cap: Maximum boost from access frequency.
        source_human_boost: Utility boost for human-sourced entries.

    Returns:
        Composite utility score in [0.0, 1.0].
    """
    # Cold-start blending: transition from impact to q_value
    if q_observations < cold_start_threshold:
        w = q_observations / max(cold_start_threshold, 1)
        effective_q = (1.0 - w) * base_impact + w * q_value
    else:
        effective_q = q_value

    # Ebbinghaus decay rate from half-life: lambda = ln(2) / half_life
    decay_rate = math.log(2) / max(half_life_days, 0.1)

    # Sub-linear recurrence strength: n^beta (minimum 1)
    recurrence_strength = max(1.0, recurrence_count) ** use_exponent

    # Strength-modulated decay: higher recurrence = slower decay
    effective_decay = decay_rate / recurrence_strength
    retention = math.exp(-effective_decay * max(days_since_last_access, 0))

    # Base composite score
    utility = effective_q * retention

    # Access count boost (sub-linear, capped)
    if access_count > 0:
        utility += min(access_count_boost_cap, 0.05 * math.log1p(access_count))

    # Source type boost for human-sourced entries
    if source_type == "human":
        utility += source_human_boost

    return _clamp01(utility)


# ---------------------------------------------------------------------------
# Feedback-aware dynamic scoring (PRD-CORE-132 FR04)
# ---------------------------------------------------------------------------


def feedback_decay_score(
    importance: float,
    recall_count: int,
    helpful_count: int,
) -> float:
    """Compute feedback-aware decay score.

    Learnings recalled often but never marked helpful decay faster.
    Helpful feedback counteracts decay.

    Formula: importance * (0.95 ** (recall_count / max(1, helpful_count)))

    Args:
        importance: Base importance/impact score (0.0-1.0).
        recall_count: Number of times this entry was recalled.
        helpful_count: Number of times this entry was marked helpful.

    Returns:
        Decayed score in [0.0, 1.0].
    """
    exponent = recall_count / max(1, helpful_count)
    return _clamp01(importance * (0.95**exponent))


def entry_utility(
    entry: dict[str, object],
    config: MemoryConfig | None = None,
    fallback_days: int | None = None,
) -> float:
    """Compute utility score for a MemoryEntry dict using config defaults.

    Extracts scoring fields from the entry dict and delegates to
    compute_utility_score. Applies time decay to both importance and q_value
    at query time (not written to disk).

    MemoryEntry field mapping vs LearningEntry:
    - importance (was impact)
    - source (was source_type)
    - created_at (was created) — ISO datetime string with time component

    Args:
        entry: Serialised MemoryEntry dict.
        config: MemoryConfig instance. Defaults to MemoryConfig() if None.
        fallback_days: Days to assume when timestamps are missing.

    Returns:
        Composite utility score in [0.0, 1.0].
    """
    cfg = config or MemoryConfig()
    effective_fallback = fallback_days if fallback_days is not None else 30
    today = datetime.now(tz=timezone.utc).date()

    # MemoryEntry uses 'importance' (was 'impact' in LearningEntry)
    q_value = _float_field(entry, "q_value", _float_field(entry, "importance", 0.5))
    base_impact = _float_field(entry, "importance", 0.5)
    q_observations = _int_field(entry, "q_observations", 0)
    recurrence = _int_field(entry, "recurrence", 1)
    access_count = _int_field(entry, "access_count", 0)
    # MemoryEntry uses 'source' (was 'source_type' in LearningEntry)
    source_type = str(entry.get("source", "agent"))
    days_unused = _days_since_access(entry, today, fallback_days=effective_fallback)

    # Double-decay fix (PRD-QUAL-032-FR03): apply_time_decay was removed here
    # because compute_utility_score() already applies Ebbinghaus exponential
    # decay internally via retention = exp(-decay_rate * days).

    # PRD-CORE-132 FR04: Apply feedback-aware decay to base_impact
    recall_ct = _int_field(entry, "recall_count", 0)
    helpful_ct = _int_field(entry, "helpful_count", 0)
    if recall_ct > 0:
        base_impact = feedback_decay_score(base_impact, recall_ct, helpful_ct)

    if cfg.lifecycle_use_fsrs:
        # FSRS measures retrieval practice count; recall_count (incremented by
        # the recall API) is more accurate than recurrence (re-store count).
        # Fall back to recurrence when recall_count is zero (cold-start entry).
        fsrs_practice = max(recall_ct, recurrence)
        return compute_fsrs_utility_score(
            importance=base_impact,
            elapsed_days=float(days_unused),
            recurrence=fsrs_practice,
        )

    return compute_utility_score(
        q_value=q_value,
        days_since_last_access=days_unused,
        recurrence_count=recurrence,
        base_impact=base_impact,
        q_observations=q_observations,
        half_life_days=cfg.decay_half_life_days,
        use_exponent=cfg.decay_use_exponent,
        cold_start_threshold=3,
        access_count=access_count,
        source_type=source_type,
        access_count_boost_cap=0.15,
        source_human_boost=0.1,
    )


# ---------------------------------------------------------------------------
# Bayesian impact calibration
# ---------------------------------------------------------------------------


def bayesian_calibrate(
    user_impact: float,
    org_mean: float = 0.5,
    user_weight: float = 1.0,
    org_weight: float = 0.5,
) -> float:
    """Compute Bayesian posterior impact score.

    Formula: (user_impact * user_weight + org_mean * org_weight) / (user_weight + org_weight)

    Args:
        user_impact: Score assigned by user (0.0-1.0).
        org_mean: Average importance across all entries (default 0.5).
        user_weight: User calibration accuracy weight.
        org_weight: Org evidence weight (capped at 2.0).

    Returns:
        Calibrated impact score (0.0-1.0).
    """
    if user_weight + org_weight == 0:
        return user_impact

    # Cap org_weight at 2.0
    org_weight = min(org_weight, 2.0)

    posterior = (user_impact * user_weight + org_mean * org_weight) / (user_weight + org_weight)
    return max(0.0, min(1.0, posterior))


def compute_calibration_accuracy(recall_stats: dict[str, object]) -> float:
    """Compute the recall-history weight consumed by TRW's Bayesian calibrator."""
    total = int(str(recall_stats.get("total_recalls", 0)))
    positive = int(str(recall_stats.get("positive_outcomes", 0)))
    if total == 0:
        return 1.0
    ratio = positive / total
    if ratio >= 0.75:
        return 2.0
    if ratio >= 0.50:
        return 1.5
    if ratio >= 0.25:
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# Forced distribution enforcement
# ---------------------------------------------------------------------------


def enforce_tier_distribution(
    entries: list[tuple[str, float]],
    *,
    critical_cap: float | None = None,
    high_cap: float | None = None,
    entry_dates: dict[str, str] | None = None,
    config: MemoryConfig | None = None,
) -> list[tuple[str, float]]:
    """Enforce forced distribution caps on importance tier percentages.

    When a tier exceeds its cap (critical >5%, high >20% by default), demotes
    the lowest-scored entry in that tier to the next tier down. Only one
    demotion per tier per call — callers may iterate to convergence via
    :func:`converge_tier_distribution`.

    Caps are config-driven (mirrors trw-mcp ``enforce_tier_distribution``):
    an explicit ``critical_cap``/``high_cap`` wins when provided, otherwise the
    value resolves from ``config`` (defaulting to ``MemoryConfig()``). This keeps
    the two implementations from drifting — the same ``.trw/config.yaml`` knobs
    now govern both.

    Args:
        entries: List of (memory_id, importance_score) tuples.
        critical_cap: Maximum fraction allowed in critical tier (0.9-1.0).
            ``None`` resolves from ``config.impact_tier_critical_cap``.
        high_cap: Maximum fraction allowed in high tier (0.7-0.89).
            ``None`` resolves from ``config.impact_tier_high_cap``.
        entry_dates: Optional mapping of memory_id -> ISO datetime string for
            time-decay-aware tier classification. Demotion target scores remain
            absolute — decay only affects which entries classify into each tier.
        config: MemoryConfig used to source caps when not given explicitly.

    Returns:
        List of (memory_id, new_importance) tuples for changed entries.
        Empty list if no demotions were needed.
    """
    cfg = config or MemoryConfig()
    effective_critical_cap = critical_cap if critical_cap is not None else cfg.impact_tier_critical_cap
    effective_high_cap = high_cap if high_cap is not None else cfg.impact_tier_high_cap

    if not entries:
        return []

    total = len(entries)

    # Don't enforce on very small sets — caps are meaningless below 5
    if total < 5:
        return []

    def _decayed_score(mid: str, score: float) -> float:
        if entry_dates is None:
            return score
        date_str = entry_dates.get(mid, "")
        if not date_str:
            return score
        try:
            created_dt = datetime.fromisoformat(date_str)
            return apply_time_decay(score, created_dt)
        except ValueError:
            return score

    # Separate into tiers using decayed scores for classification
    critical: list[tuple[str, float]] = []
    high: list[tuple[str, float]] = []

    for mid, score in entries:
        tier_score = _decayed_score(mid, score)
        if tier_score >= 0.9:
            critical.append((mid, score))
        elif tier_score >= 0.7:
            high.append((mid, score))

    demotions: list[tuple[str, float]] = []

    # Enforce critical cap: demote lowest-scored critical → high
    if critical and len(critical) / total > effective_critical_cap:
        critical_sorted = sorted(critical, key=lambda x: x[1])
        victim_id, victim_score = critical_sorted[0]
        new_score = round(min(0.89, max(0.7, victim_score - 0.1)), 4)
        demotions.append((victim_id, new_score))
        logger.info(
            "tier_demotion",
            memory_id=victim_id,
            from_tier="critical",
            to_tier="high",
            old_score=victim_score,
            new_score=new_score,
        )

    # Re-compute high count after potential demotion from critical
    demoted_ids = {d[0] for d in demotions}
    effective_high = [e for e in high if e[0] not in demoted_ids]
    effective_high_count = len(effective_high) + len(demotions)

    # Enforce high cap: demote lowest-scored high → medium
    if effective_high_count > 0 and effective_high_count / total > effective_high_cap:
        high_sorted = sorted(
            [(mid, s) for mid, s in high if mid not in demoted_ids],
            key=lambda x: x[1],
        )
        if high_sorted:
            victim_id, victim_score = high_sorted[0]
            new_score = round(min(0.69, max(0.4, victim_score - 0.1)), 4)
            demotions.append((victim_id, new_score))
            logger.info(
                "tier_demotion",
                memory_id=victim_id,
                from_tier="high",
                to_tier="medium",
                old_score=victim_score,
                new_score=new_score,
            )

    return demotions


def converge_tier_distribution(
    entries: list[tuple[str, float]],
    *,
    critical_cap: float | None = None,
    high_cap: float | None = None,
    entry_dates: dict[str, str] | None = None,
    config: MemoryConfig | None = None,
    max_iterations: int = 1000,
) -> list[tuple[str, float]]:
    """Batch-converge an over-cap tier cluster to within its caps.

    ``enforce_tier_distribution`` demotes at most one entry per tier per call,
    so a cluster that is many entries over a cap never heals at write time
    (R-RANK-003). This helper iterates that single-step enforcement on an
    in-memory working set until no further demotion is produced, then returns
    the *net* score change per entry (one tuple per entry that ended below its
    starting tier, carrying its final score).

    This is a pure function — it does not touch storage. Use
    :func:`persist_tier_convergence` to apply the result atomically.

    Args:
        entries: List of (memory_id, importance_score) tuples.
        critical_cap: See :func:`enforce_tier_distribution`.
        high_cap: See :func:`enforce_tier_distribution`.
        entry_dates: Optional decay-aware classification dates.
        config: MemoryConfig used to source caps when not given explicitly.
        max_iterations: Hard ceiling on convergence steps (safety bound).

    Returns:
        List of (memory_id, final_importance) tuples for every entry whose
        score changed from its starting value. Empty when already within caps.
    """
    if not entries:
        return []

    # Working set keyed by id so repeated demotions compound on the same entry.
    working: dict[str, float] = {}
    order: list[str] = []
    original: dict[str, float] = {}
    for mid, score in entries:
        if mid not in working:
            order.append(mid)
        working[mid] = score
        original.setdefault(mid, score)

    for _ in range(max(1, max_iterations)):
        snapshot = [(mid, working[mid]) for mid in order]
        step = enforce_tier_distribution(
            snapshot,
            critical_cap=critical_cap,
            high_cap=high_cap,
            entry_dates=entry_dates,
            config=config,
        )
        if not step:
            break
        working.update(dict(step))

    changed: list[tuple[str, float]] = [(mid, working[mid]) for mid in order if working[mid] != original[mid]]

    if changed:
        logger.info(
            "tier_convergence",
            n_changed=len(changed),
            total=len(order),
        )
    return changed


def persist_tier_convergence(
    backend: StorageBackend,
    entries: list[tuple[str, float]],
    *,
    critical_cap: float | None = None,
    high_cap: float | None = None,
    entry_dates: dict[str, str] | None = None,
    config: MemoryConfig | None = None,
) -> list[tuple[str, float]]:
    """Converge a tier cluster and persist the demotions atomically.

    Wraps :func:`converge_tier_distribution` and writes every resulting score
    change inside a single ``backend.transaction()`` so the cluster either
    converges fully or not at all — no partially-demoted intermediate state is
    ever committed. Uses the committed thread-safe ``transaction()`` (PRD-FIX-088
    FR02), so concurrent writers cannot observe a half-applied convergence.

    Args:
        backend: Storage backend exposing ``transaction()`` and ``update()``.
        entries: List of (memory_id, importance_score) tuples.
        critical_cap: See :func:`enforce_tier_distribution`.
        high_cap: See :func:`enforce_tier_distribution`.
        entry_dates: Optional decay-aware classification dates.
        config: MemoryConfig used to source caps when not given explicitly.

    Returns:
        The list of (memory_id, final_importance) changes that were persisted.
        Empty when the cluster was already within caps (no transaction opened).
    """
    changes = converge_tier_distribution(
        entries,
        critical_cap=critical_cap,
        high_cap=high_cap,
        entry_dates=entry_dates,
        config=config,
    )
    if not changes:
        return []

    with backend.transaction() as txn:
        for mid, new_score in changes:
            txn.update(mid, importance=new_score)

    logger.info("tier_convergence_persisted", n_persisted=len(changes))
    return changes


# ---------------------------------------------------------------------------
# Recall ranking
# ---------------------------------------------------------------------------


def rank_by_utility(
    matches: list[dict[str, object]],
    query_tokens: list[str],
    lambda_weight: float,
    config: MemoryConfig | None = None,
) -> list[dict[str, object]]:
    """Compatibility wrapper for :func:`lifecycle._recall.rank_by_utility`."""
    # Import lazily because _recall imports entry_utility from this module.
    from trw_memory.lifecycle._recall import rank_by_utility as _rank_by_utility

    return _rank_by_utility(matches, query_tokens, lambda_weight, config=config)


# ---------------------------------------------------------------------------
# Pruning candidate identification
# ---------------------------------------------------------------------------


def utility_based_prune_candidates(
    entries: list[dict[str, object]],
    config: MemoryConfig | None = None,
    *,
    delete_threshold: float = 0.05,
    prune_threshold: float = 0.15,
) -> list[dict[str, object]]:
    """Compatibility wrapper for the canonical recall-time implementation."""
    from trw_memory.lifecycle._recall import utility_based_prune_candidates as _prune_candidates

    return _prune_candidates(
        entries,
        config=config,
        delete_threshold=delete_threshold,
        prune_threshold=prune_threshold,
    )


# ---------------------------------------------------------------------------
# FSRS-inspired adaptive retention scoring (frontier-002)
# ---------------------------------------------------------------------------
# Simplified FSRS-4.5: tracks stability S (days at 90% retention) and
# difficulty D via the power-law retention curve
# R(t, S) = (1 + FACTOR * t/S)^DECAY, with FACTOR/DECAY chosen so R(S,S)=0.9.
# Full FSRS-4.5 uses a neural scheduler; we implement the closed-form
# retrieval-based update rules from the original paper.
# Reference: Ye et al. "A Stochastic Shortest Path Algorithm for
#   Optimizing Spaced Repetition Scheduling" (2022).


_FSRS_DECAY: float = -0.5  # power-law decay exponent (FSRS-4.5 default)
# FSRS-4.5 forgetting-curve FACTOR. Derived so that R(t=S, S) == 0.9 exactly:
#   (1 + FACTOR) ** DECAY == 0.9  =>  FACTOR == 0.9 ** (1/DECAY) - 1 == 19/81.
# (The 0.9 *target retrievability* is encoded in this constant, not stored as
# the FACTOR itself — a common point of confusion when transcribing the paper.)
_FSRS_FACTOR: float = 0.9 ** (1.0 / _FSRS_DECAY) - 1.0  # == 19/81 ~= 0.234568
_FSRS_DEFAULT_STABILITY: float = 1.0  # initial stability in days


def fsrs_retrievability(elapsed_days: float, stability: float) -> float:
    """FSRS power-law retention: R(t, S) = (1 + FACTOR * t / S)^DECAY.

    Gives R=1.0 at t=0, R~=0.9 at t=S, R->0 as t->inf.

    Args:
        elapsed_days: Days since the memory was last reviewed or created.
        stability: Current stability S in days (the interval at which R~=0.9).

    Returns:
        Retrievability in [0, 1].
    """
    if stability <= 0.0:
        stability = _FSRS_DEFAULT_STABILITY
    t = max(0.0, elapsed_days)
    return float((1.0 + _FSRS_FACTOR * t / stability) ** _FSRS_DECAY)


def compute_fsrs_utility_score(
    importance: float,
    elapsed_days: float,
    recurrence: int = 1,
    *,
    stability: float | None = None,
    difficulty: float = 5.0,
) -> float:
    """FSRS-powered utility score replacing Ebbinghaus decay.

    Blends FSRS retrievability R(t, S) with the entry's importance
    and recurrence to produce a single [0, 1] utility score.

        utility = R(t, S) * sqrt(importance) * (1 + log1p(recurrence) / 10)

    The recurrence bonus is capped at 1.5x to prevent viral entries
    from permanently dominating the utility ranking.

    Args:
        importance: Entry importance in [0, 1].
        elapsed_days: Days since last recall / creation.
        recurrence: Number of times the entry has been recalled (>=1).
        stability: FSRS stability S in days.  When ``None``, estimated
            from recurrence: S ~= 1.0 * (recurrence ** 0.3) * 7 (heuristic
            that gives S=7 at recurrence=1, S~=14 at recurrence=3).
        difficulty: Entry difficulty in [1, 10] (used when stability is None).

    Returns:
        Utility score in [0, 1].
    """
    if stability is None:
        # Heuristic: more recalls -> longer stability
        stability = max(
            _FSRS_DEFAULT_STABILITY,
            (max(1, recurrence) ** 0.3) * 7.0 / (difficulty / 5.0),
        )
    r = fsrs_retrievability(elapsed_days, stability)
    imp_factor = math.sqrt(max(0.0, min(1.0, importance)))
    rec_bonus = min(1.5, 1.0 + math.log1p(max(0, recurrence - 1)) / 10.0)
    return _clamp01(r * imp_factor * rec_bonus)
