"""Utility-based scoring for the trw-memory lifecycle layer.

Core scoring functions:
- compute_utility_score: Ebbinghaus decay over base impact
- apply_time_decay: Linear time decay with 0.3 floor

Research basis:
- Ebbinghaus forgetting curve

Scoring math is identical to trw-mcp scoring.py — adapted to use MemoryConfig
and MemoryEntry field names (importance vs impact, source vs source_type,
created_at vs created).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import structlog

from trw_memory.lifecycle._utility_params import UtilityParams
from trw_memory.lifecycle._utils import days_since_access as _days_since_access
from trw_memory.models.config import MemoryConfig

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


#: Floor on the recall-frequency decay FACTOR, which is ``0.95 ** recall_count``
#: and unbounded below; 0.5 caps the worst case at halving importance. Chosen as
#: the largest penalty that would still be recoverable by a single round of real
#: usefulness feedback rather than by a measured optimum -- no calibration data
#: exists, because PRD-CORE-293 confirmed the ``helpful_count``/``unhelpful_count``
#: signal this floor used to be sized against was never once supplied (0 of
#: 1,617 rows) and removed its only writer. Overridable per call and via
#: ``MemoryConfig.feedback_decay_min_factor``.
_DEFAULT_FEEDBACK_DECAY_MIN_FACTOR = 0.5

#: Days assumed when an entry carries no parseable timestamp at all. Preserves
#: the literal the pre-FR11 trw_memory implementation used; the trw-mcp caller
#: overrides it with ``TRWConfig.scoring_default_days_unused``.
_DEFAULT_FALLBACK_DAYS = 30


def _parse_expires(raw: str) -> date | None:
    """Parse an ``expires`` field to a date, or ``None`` when it never expires.

    Accepts a bare ISO date and a full ISO datetime. An empty or unparseable
    value never expires — matching ``trw_memory.retrieval.validity_prior`` so the
    ranking and the eligibility paths agree about the same string.
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None


def utility_params_from_config(config: MemoryConfig) -> UtilityParams:
    """Bind the trw-memory tuning surface to the shared knob bundle."""
    return UtilityParams(
        half_life_days=config.decay_half_life_days,
        use_exponent=config.decay_use_exponent,
        feedback_decay_min_factor=config.feedback_decay_min_factor,
    )


def _clamp01(value: float) -> float:
    """Clamp a value to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def _ensure_utc(ts: datetime) -> datetime:
    """Return a timezone-aware datetime, assuming UTC if naive."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


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
    days_since_last_access: int,
    recurrence_count: int,
    base_impact: float,
    *,
    half_life_days: float = 14.0,
    use_exponent: float = 0.6,
    access_count: int = 0,
    source_type: str = "agent",
    access_count_boost_cap: float = 0.15,
    source_human_boost: float = 0.1,
) -> float:
    """Compute composite utility: importance under Ebbinghaus decay, plus boosts.

    Formula:
        retention = recurrence_strength * exp(-effective_decay * days)
        utility = base_impact * retention + access_boost + source_boost

    PRD-CORE-293: the Q-value blend that used to replace ``base_impact`` after
    enough outcome observations is gone with the reward loop (no observation was
    ever recorded, so the blend always returned ``base_impact``).

    Args:
        days_since_last_access: Days since last recall.
        recurrence_count: Number of times recalled (minimum 1).
        base_impact: Original static importance score (0.0-1.0).
        half_life_days: Days until retention halves. Default 14.
        use_exponent: Sub-linear recurrence exponent. Default 0.6.
        access_count: Number of times recalled (for sub-linear boost).
        source_type: 'human' or 'agent'.
        access_count_boost_cap: Maximum boost from access frequency.
        source_human_boost: Utility boost for human-sourced entries.

    Returns:
        Composite utility score in [0.0, 1.0].
    """
    effective_q = base_impact

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
# Recall-frequency decay (PRD-CORE-132 FR04, rewritten by PRD-CORE-293 FR02)
# ---------------------------------------------------------------------------


def recall_frequency_decay_score(
    importance: float,
    recall_count: int,
    *,
    min_factor: float = _DEFAULT_FEEDBACK_DECAY_MIN_FACTOR,
) -> float:
    """Decay importance by how often an entry was recalled, with a floor.

    Formula: ``importance * max(min_factor, 0.95 ** recall_count)``

    PRD-CORE-293 FR02: this was named ``feedback_decay_score`` and took a
    ``helpful_count`` argument, with the exponent
    ``recall_count / max(1, helpful_count)``. ``helpful_count`` was written only
    by ``trw_learn_update(feedback=...)``, which no caller ever issued (0 of
    1,617 rows measured 2026-09-22) -- so the exponent always degenerated to
    plain ``recall_count`` and the "feedback-aware" name was never true of any
    observed call. Dropping the parameter changes nothing about the live
    formula's OUTPUT (every prior call site passed ``helpful_count=0``,
    ``max(1, 0) == 1``, same ``0.95 ** recall_count``); it removes an argument
    that could never carry a value.

    **Why the floor exists (PRD-CORE-244 FR11 residual).** Without it, being
    recalled often is a pure penalty, unbounded below: an entry surfaced 100
    times decays to 0.6% of its importance on retrieval frequency alone, and
    PRD-QUAL-032/D1 already established that being surfaced in a result set is
    not evidence of use. ``min_factor`` bounds that damage -- the default halves
    importance at worst, a large but recoverable penalty for a heavily-recalled
    entry.

    Args:
        importance: Base importance/impact score (0.0-1.0).
        recall_count: Number of times this entry was recalled.
        min_factor: Lower bound on the decay FACTOR (not the score). 0.0
            restores the pre-floor behaviour exactly.

    Returns:
        Decayed score in [0.0, 1.0].
    """
    return _clamp01(importance * max(min_factor, 0.95**recall_count))


def entry_utility(
    entry: dict[str, object],
    config: MemoryConfig | None = None,
    fallback_days: int | None = None,
    *,
    params: UtilityParams | None = None,
    today: date | None = None,
) -> float:
    """Compute the utility score for one serialized entry — the ONLY implementation.

    PRD-CORE-244 FR11 collapsed two independent implementations into this one and
    repointed the live ``trw_recall`` ranker at it. It is therefore the UNION of
    what both did, and each retained behaviour is asserted individually so a
    future merge cannot silently drop one:

    * the **expiry floor** (PRD-CORE-110, from the trw-mcp side) — an entry past
      its author-set ``expires`` date scores ``params.expired_utility_floor``.
      Day-exclusive: an entry expiring today is still current.
    * **unverified-incident preservation** (from the trw-mcp side) — an
      unverified incident gets an effectively infinite half-life so a postmortem
      is not decayed away before the fix is confirmed.
    * **per-type half-life, access-count and source-type terms** (from the
      trw-mcp side).
    * **feedback-aware decay** (PRD-CORE-132, from the trw-memory side) — the
      term ``trw_learn``'s own docstring credits ``feedback`` with, which the
      live ranker ignored for its entire life.

    Field names are read alias-tolerantly because the two callers serialize
    different models: ``importance``/``impact``, ``source``/``source_type``.
    Neither vocabulary is privileged, so a LearningEntry dict and a MemoryEntry
    dump score identically.

    Args:
        entry: Serialized entry dict (YAML, SQLite row, or ``model_dump()``).
        config: ``MemoryConfig`` used to derive *params* when it is not supplied.
        fallback_days: Days to assume when no timestamp is parseable.
        params: Pre-bound knob bundle. Callers ranking many entries build it ONCE
            (``MemoryConfig`` is a BaseSettings that reads config.yaml, so
            constructing it per entry is a filesystem read per row).
        today: Injectable reference date; defaults to today in UTC.

    Returns:
        Composite utility score in [0.0, 1.0].
    """
    effective_params = params if params is not None else utility_params_from_config(config or MemoryConfig())
    effective_fallback = fallback_days if fallback_days is not None else _DEFAULT_FALLBACK_DAYS
    reference_day = today or datetime.now(tz=timezone.utc).date()

    # Expiry short-circuits every other term: a record whose validity window has
    # closed is not "slightly less useful", it is out of date.
    expires_date = _parse_expires(str(entry.get("expires", "")))
    if expires_date is not None and reference_day > expires_date:
        return effective_params.expired_utility_floor

    base_impact = _float_field(entry, "importance", _float_field(entry, "impact", 0.5))
    recurrence = _int_field(entry, "recurrence", 1)
    access_count = _int_field(entry, "access_count", 0)
    source_type = str(entry.get("source", entry.get("source_type", "agent")))
    days_unused = _days_since_access(entry, reference_day, fallback_days=effective_fallback)

    # Double-decay fix (PRD-QUAL-032-FR03): apply_time_decay was removed here
    # because compute_utility_score() already applies Ebbinghaus exponential
    # decay internally via retention = exp(-decay_rate * days).

    # PRD-CORE-132 FR04 / PRD-CORE-293 FR02: recall-frequency decay of base_impact.
    recall_ct = _int_field(entry, "recall_count", 0)
    if recall_ct > 0:
        base_impact = recall_frequency_decay_score(
            base_impact,
            recall_ct,
            min_factor=effective_params.feedback_decay_min_factor,
        )

    half_life = effective_params.half_life_for(
        str(entry.get("type", "")),
        str(entry.get("confidence", "unverified")),
    )
    return compute_utility_score(
        days_since_last_access=days_unused,
        recurrence_count=recurrence,
        base_impact=base_impact,
        half_life_days=half_life,
        use_exponent=effective_params.use_exponent,
        access_count=access_count,
        source_type=source_type,
        access_count_boost_cap=effective_params.access_count_boost_cap,
        source_human_boost=effective_params.source_human_boost,
    )


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
    demotion per tier per call — a caller that needs a cluster brought fully
    within its caps re-invokes until the returned list is empty.

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
