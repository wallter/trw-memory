"""Recall-time scoring — ranking and pruning for memory retrieval.

Functions in this module operate on serialised MemoryEntry dicts at recall time:
- rank_by_utility: Re-rank matched entries by combined relevance + utility score
- utility_based_prune_candidates: Identify stale/low-utility entries for cleanup

These were extracted from scoring.py to keep module size below 500 lines.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import structlog

from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


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


def record_recall_access(
    backend: StorageBackend,
    entry_ids: list[str],
    *,
    accessed_at: datetime | None = None,
) -> None:
    """Record recall-time access for the entries that were actually returned.

    Utility scoring already depends on ``access_count`` and
    ``last_accessed_at``. Updating only the final returned IDs keeps recall
    bookkeeping aligned with user-visible results instead of inflating scores
    for over-fetched candidates that were never surfaced.
    """
    if not entry_ids:
        return

    touch_time = accessed_at or datetime.now(timezone.utc)
    seen_ids: set[str] = set()
    for entry_id in entry_ids:
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)

        entry = backend.get(entry_id)
        if entry is None:
            logger.debug("recall_access_missing_entry", memory_id=entry_id)
            continue

        backend.update(
            entry_id,
            access_count=entry.access_count + 1,
            recall_count=entry.recall_count + 1,
            last_accessed_at=touch_time,
        )


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
