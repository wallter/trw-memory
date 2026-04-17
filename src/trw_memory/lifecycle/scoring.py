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
from datetime import date, datetime, timezone

import structlog

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
    """Compute user calibration accuracy from recall tracking data.

    Returns:
        Accuracy score 0.0-2.0 (used as user_weight in bayesian_calibrate).
    """
    total = int(str(recall_stats.get("total_recalls", 0)))
    positive = int(str(recall_stats.get("positive_outcomes", 0)))

    if total == 0:
        return 1.0  # default weight

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
    critical_cap: float = 0.05,
    high_cap: float = 0.20,
    entry_dates: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Enforce forced distribution caps on importance tier percentages.

    When a tier exceeds its cap (critical >5%, high >20%), demotes the
    lowest-scored entry in that tier to the next tier down. Only one
    demotion per tier per call.

    Args:
        entries: List of (memory_id, importance_score) tuples.
        critical_cap: Maximum fraction allowed in critical tier (0.9-1.0).
        high_cap: Maximum fraction allowed in high tier (0.7-0.89).
        entry_dates: Optional mapping of memory_id -> ISO datetime string for
            time-decay-aware tier classification.

    Returns:
        List of (memory_id, new_importance) tuples for changed entries.
        Empty list if no demotions were needed.
    """
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
    if critical and len(critical) / total > critical_cap:
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
    if effective_high_count > 0 and effective_high_count / total > high_cap:
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


# ---------------------------------------------------------------------------
# Recall ranking
# ---------------------------------------------------------------------------


def rank_by_utility(
    matches: list[dict[str, object]],
    query_tokens: list[str],
    lambda_weight: float,
    config: MemoryConfig | None = None,
) -> list[dict[str, object]]:
    """Re-rank matched entries by combined relevance + utility score.

    Combined score = (1 - lambda) * relevance + lambda * utility

    Args:
        matches: List of MemoryEntry dicts.
        query_tokens: Lowercased query tokens for relevance scoring.
        lambda_weight: Blend factor. 0.0 = pure relevance, 1.0 = pure utility.
        config: MemoryConfig for utility calculation. Defaults to MemoryConfig().

    Returns:
        Sorted list (highest combined score first).
    """
    if not matches:
        return matches

    scored: list[tuple[float, dict[str, object]]] = []

    for entry in matches:
        # Text relevance score — MemoryEntry uses 'content' (was 'summary')
        content = str(entry.get("content", "")).lower()
        detail = str(entry.get("detail", "")).lower()
        raw_tags = entry.get("tags", [])
        tag_text = " ".join(str(t).lower() for t in raw_tags) if isinstance(raw_tags, list) else ""

        if query_tokens:
            content_hits = sum(1 for t in query_tokens if t in content)
            tag_hits = sum(1 for t in query_tokens if t in tag_text)
            detail_hits = sum(1 for t in query_tokens if t in detail)
            weighted_hits = content_hits * 3 + tag_hits * 2 + detail_hits
            max_possible = len(query_tokens) * 3
            relevance = min(1.0, weighted_hits / max(max_possible, 1))
        else:
            relevance = 1.0  # wildcard query

        utility = entry_utility(entry, config=config)
        combined = (1.0 - lambda_weight) * relevance + lambda_weight * utility

        scored.append((combined, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored]


# ---------------------------------------------------------------------------
# Pruning candidate identification
# ---------------------------------------------------------------------------

# Default thresholds (matching trw-mcp scoring.py defaults)
_DELETE_THRESHOLD = 0.05
_PRUNE_THRESHOLD = 0.15


def utility_based_prune_candidates(
    entries: list[dict[str, object]],
    config: MemoryConfig | None = None,
    *,
    delete_threshold: float = _DELETE_THRESHOLD,
    prune_threshold: float = _PRUNE_THRESHOLD,
) -> list[dict[str, object]]:
    """Identify prune candidates using composite utility scoring.

    Three tiers:
    1. Status-based cleanup: entries already resolved/obsolete
    2. Delete candidates: utility < delete_threshold
    3. Obsolete candidates: utility < prune_threshold and age > 14 days

    Args:
        entries: List of serialised MemoryEntry dicts.
        config: MemoryConfig for utility calculation.
        delete_threshold: Utility below this → delete candidate.
        prune_threshold: Utility below this (and age > 14 days) → obsolete candidate.

    Returns:
        List of candidate dicts with id, content, utility, and suggested_status.
    """
    candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    today = datetime.now(tz=timezone.utc).date()

    for data in entries:
        entry_id = str(data.get("id", ""))
        if entry_id in seen_ids:
            continue

        created_raw = data.get("created_at")
        created_str = str(created_raw) if created_raw is not None else ""
        try:
            created = (
                date.fromisoformat(created_str[:10]) if created_str and created_str not in ("None", "null") else today
            )
        except ValueError:
            created = today

        age_days = max(0, (today - created).days)
        entry_status = str(data.get("status", "active"))

        # Tier 1: Status-based cleanup
        if entry_status in ("resolved", "obsolete"):
            candidates.append(
                {
                    "id": entry_id,
                    "content": data.get("content", ""),
                    "age_days": age_days,
                    "utility": 0.0,
                    "suggested_status": entry_status,
                    "reason": f"Already marked {entry_status} — cleanup candidate",
                }
            )
            seen_ids.add(entry_id)
            continue

        utility = entry_utility(data, config=config, fallback_days=age_days)

        # Tier 2: Delete-level utility
        if utility < delete_threshold:
            candidates.append(
                {
                    "id": entry_id,
                    "content": data.get("content", ""),
                    "age_days": age_days,
                    "utility": round(utility, 3),
                    "suggested_status": "obsolete",
                    "reason": (
                        f"Utility {utility:.3f} below delete threshold ({delete_threshold:.3f}). age={age_days}d"
                    ),
                }
            )
            seen_ids.add(entry_id)
            continue

        # Tier 3: Prune-level utility (fading, older than 14 days)
        if utility < prune_threshold and age_days > 14:
            candidates.append(
                {
                    "id": entry_id,
                    "content": data.get("content", ""),
                    "age_days": age_days,
                    "utility": round(utility, 3),
                    "suggested_status": "obsolete",
                    "reason": (
                        f"Utility {utility:.3f} below prune threshold ({prune_threshold:.3f}) and age {age_days}d > 14d"
                    ),
                }
            )
            seen_ids.add(entry_id)

    return candidates
