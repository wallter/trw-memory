"""Recall-time scoring — ranking and pruning for memory retrieval.

Functions in this module operate on serialised MemoryEntry dicts at recall time:
- rank_by_utility: Re-rank matched entries by combined relevance + utility score
- utility_based_prune_candidates: Identify stale/low-utility entries for cleanup

These were extracted from scoring.py to keep module size below 500 lines.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import structlog

from trw_memory.lifecycle.protection import prune_threshold_multiplier
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.lexical import lexical_relevance
from trw_memory.retrieval.source_policy import resolve_expiry
from trw_memory.retrieval.validity_prior import expiry_has_passed
from trw_memory.storage.interface import StorageBackend

#: The key a recall path writes the retrieval score under. Named here, where the
#: ranker reads it, so the producer and the consumer cannot drift.
FUSED_SCORE_KEY = "score"

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Expiry filtering (F6)
# ---------------------------------------------------------------------------


def _expires_in_past(entry: dict[str, object], *, now: datetime | None = None) -> bool:
    """Return True when *entry*'s expiry has passed (PRD-CORE-278 FR05).

    Delegates to :func:`~trw_memory.retrieval.validity_prior.expiry_has_passed`,
    the ONE predicate: a value carrying a time expires at that instant (UTC), a
    bare date expires at the end of that UTC day, and a missing, empty or
    unparseable value never expires. Field resolution is shared too — the value
    is read through :func:`~trw_memory.retrieval.source_policy.resolve_expiry`,
    which looks at the top-level ``expires`` and then ``metadata["expires"]`` —
    so an entry cannot be expired on one surface and live on another.
    """
    return expiry_has_passed(resolve_expiry(entry), reference_time=now)


def drop_expired_entries(matches: list[dict[str, object]]) -> list[dict[str, object]]:
    """Filter out entries whose expiry has passed (F6, PRD-CORE-278 FR05).

    Entries with an empty / non-date / future expiry pass through unchanged.
    This is the recall-path guard that stops stale, already-expired learnings
    from being surfaced forever. It runs on the MERGED result set — after the
    tier and org merges — because running it earlier (inside ``rank_by_utility``)
    let a merge re-admit an entry it had just removed.
    """
    if not matches:
        return matches
    kept: list[dict[str, object]] = []
    dropped = 0
    for entry in matches:
        if _expires_in_past(entry):
            dropped += 1
            continue
        kept.append(entry)
    if dropped:
        logger.debug("recall_dropped_expired_entries", dropped=dropped, kept=len(kept))
    return kept


# ---------------------------------------------------------------------------
# Recall ranking
# ---------------------------------------------------------------------------


def _relevance(entry: dict[str, object], query_tokens: list[str], max_score: float) -> float:
    """Relevance in ``[0, 1]`` for one candidate (PRD-CORE-278 FR02).

    A candidate carrying a FINITE retrieval score is scored on it, normalised by
    the largest finite score in the same call. An unreadable or non-finite score
    is rejected individually — it falls back to lexical relevance — rather than
    collapsing the whole call. Normalisation is per-call, order-preserving
    within the call, and never persisted; the raw score is what the response
    carries.
    """
    raw = entry.get(FUSED_SCORE_KEY)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        # A score at or below zero carries NO usable relevance and is reported
        # as 0.0 rather than normalised: every producer in this package emits a
        # non-negative score (RRF positions, recency blend, utility), so a
        # negative value means the number is not on the expected scale, and
        # inventing an order for it would be inventing evidence. Such rows tie
        # at 0.0 and the utility tiebreak decides between them.
        if max_score <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(raw) / max_score))
    return lexical_relevance(entry, query_tokens)


def rank_by_utility(
    matches: list[dict[str, object]],
    query_tokens: list[str],
    config: MemoryConfig | None = None,
) -> list[dict[str, object]]:
    """Re-rank matched entries by relevance, with utility as a TIEBREAK.

    PRD-CORE-278 FR02. The relevance term is the retrieval score the pipeline
    already computed, when the caller carried it (see ``FUSED_SCORE_KEY``);
    otherwise it is whole-word lexical overlap with stopwords removed. Utility
    (recency, access count, importance) breaks ties between equal relevance and
    can no longer reorder two candidates whose relevance differs — which is what
    a ``0.6 * substring + 0.4 * utility`` blend did, discarding the ranking the
    retrieval pipeline had just produced.

    The ``lambda_weight`` parameter is gone with the blend: a tiebreak has
    nothing to weigh, and no new weighting knob replaces it (NFR01).

    Expiry is NOT applied here. Ranking and admission are different jobs; see
    :func:`drop_expired_entries`, which the recall path runs on the merged set.

    Args:
        matches: List of MemoryEntry dicts, optionally carrying ``score``.
        query_tokens: Query tokens for the lexical fallback. Empty = wildcard,
            which scores every entry 1.0 and therefore orders purely by utility.
        config: MemoryConfig for utility calculation. Defaults to MemoryConfig().

    Returns:
        Sorted list (highest relevance first, then highest utility).
    """
    if not matches:
        return matches

    finite_scores = [
        float(value)
        for entry in matches
        if isinstance(value := entry.get(FUSED_SCORE_KEY), (int, float)) and math.isfinite(float(value))
    ]
    max_score = max(finite_scores) if finite_scores else 0.0

    scored: list[tuple[float, float, dict[str, object]]] = []
    for entry in matches:
        relevance = _relevance(entry, query_tokens, max_score)
        scored.append((relevance, entry_utility(entry, config=config), entry))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry for _, _, entry in scored]


def record_recall_access(
    backend: StorageBackend,
    entry_ids: list[str],
    *,
    namespace: str,
    accessed_at: datetime | None = None,
) -> None:
    """Record recall-time access for *namespace*'s entries that were actually returned.

    Utility scoring already depends on ``access_count`` and
    ``last_accessed_at``. Updating only the final returned IDs keeps recall
    bookkeeping aligned with user-visible results instead of inflating scores
    for over-fetched candidates that were never surfaced.

    F-008: this batches into a SINGLE ``UPDATE ... WHERE id IN (...)`` (one
    commit / one WAL append) instead of the old per-entry get+update loop that
    issued 2 statements + 1 WAL append per recalled entry (50 statements / 25
    WAL appends for a 25-result recall). Per-entry increment semantics are
    preserved — each distinct id is incremented exactly once.
    """
    if not entry_ids:
        return

    touch_time = accessed_at or datetime.now(timezone.utc)
    backend.increment_recall_access(entry_ids, namespace=namespace, accessed_at=touch_time)


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

    PRD-CORE-244 FR10: every tier honours ``protection_tier``. This is the fifth
    of the five destructive paths the FR names and the one an independent review
    caught still uncovered — it is publicly exported from ``lifecycle`` and
    ``lifecycle.scoring.utility_based_prune_candidates`` delegates into it, so a
    ``permanent`` entry was nominated here even after the trw-mcp twin was fixed.
    ``protected`` and ``permanent`` are never nominated, including by the status
    tier, because "already marked obsolete" is still automatic removal. Every
    other tier multiplies both thresholds by its configured discount.

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

    discounts = (config or MemoryConfig()).protection_tier_prune_discount

    for data in entries:
        entry_id = str(data.get("id", ""))
        if entry_id in seen_ids:
            continue

        # Read the tier BEFORE any nomination branch, so the exemption cannot be
        # reached around by an earlier tier.
        multiplier = prune_threshold_multiplier(data, discounts)
        if multiplier is None:
            seen_ids.add(entry_id)
            continue
        entry_delete_threshold = delete_threshold * multiplier
        entry_prune_threshold = prune_threshold * multiplier

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
        if utility < entry_delete_threshold:
            candidates.append(
                {
                    "id": entry_id,
                    "content": data.get("content", ""),
                    "age_days": age_days,
                    "utility": round(utility, 3),
                    "suggested_status": "obsolete",
                    "reason": (
                        f"Utility {utility:.3f} below delete threshold ({entry_delete_threshold:.3f}). age={age_days}d"
                    ),
                }
            )
            seen_ids.add(entry_id)
            continue

        # Tier 3: Prune-level utility (fading, older than 14 days)
        if utility < entry_prune_threshold and age_days > 14:
            candidates.append(
                {
                    "id": entry_id,
                    "content": data.get("content", ""),
                    "age_days": age_days,
                    "utility": round(utility, 3),
                    "suggested_status": "obsolete",
                    "reason": (
                        f"Utility {utility:.3f} below prune threshold ({entry_prune_threshold:.3f}) and age {age_days}d > 14d"
                    ),
                }
            )
            seen_ids.add(entry_id)

    return candidates
