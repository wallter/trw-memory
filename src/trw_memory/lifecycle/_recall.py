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
# Expiry filtering (F6)
# ---------------------------------------------------------------------------


def _expires_in_past(raw: object) -> bool:
    """Return True only when ``raw`` parses as a PAST ISO datetime.

    ``MemoryEntry.expires`` is FREE-FORM: it may hold a date, a date-time, a
    natural-language condition (e.g. "when migration ships"), or be empty.
    We must only treat an entry as expired when its ``expires`` value parses
    unambiguously as an ISO datetime that is already in the past. Empty values,
    non-date strings, and future datetimes are never treated as expired.
    """
    if not isinstance(raw, str):
        return False
    text = raw.strip()
    if not text:
        return False
    candidate = text
    # Accept a trailing 'Z' (UTC) which datetime.fromisoformat historically
    # rejected on older Pythons; normalise to an explicit offset.
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        # Date-only ISO string (no time component).
        try:
            parsed_date = date.fromisoformat(candidate)
        except ValueError:
            return False
        parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    # Compare in UTC. Treat naive datetimes as UTC for a stable comparison.
    now = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed < now


def drop_expired_entries(matches: list[dict[str, object]]) -> list[dict[str, object]]:
    """Filter out entries whose ``expires`` is a PAST ISO datetime (F6).

    Entries with an empty / non-date / future ``expires`` value pass through
    unchanged. This is the recall-path guard that stops stale, already-expired
    learnings (e.g. ``expires='2025-01-01'``) from being surfaced forever.
    """
    if not matches:
        return matches
    kept: list[dict[str, object]] = []
    dropped = 0
    for entry in matches:
        if _expires_in_past(entry.get("expires")):
            dropped += 1
            continue
        kept.append(entry)
    if dropped:
        logger.debug("recall_dropped_expired_entries", dropped=dropped, kept=len(kept))
    return kept


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

    F6: expired entries (``expires`` parses as a past ISO datetime) are dropped
    up front so they never reach the ranked recall result.

    Args:
        matches: List of MemoryEntry dicts.
        query_tokens: Lowercased query tokens for relevance scoring.
        lambda_weight: Blend factor. 0.0 = pure relevance, 1.0 = pure utility.
        config: MemoryConfig for utility calculation. Defaults to MemoryConfig().

    Returns:
        Sorted list (highest combined score first).
    """
    matches = drop_expired_entries(matches)
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

    F-008: this batches into a SINGLE ``UPDATE ... WHERE id IN (...)`` (one
    commit / one WAL append) instead of the old per-entry get+update loop that
    issued 2 statements + 1 WAL append per recalled entry (50 statements / 25
    WAL appends for a 25-result recall). Per-entry increment semantics are
    preserved — each distinct id is incremented exactly once.
    """
    if not entry_ids:
        return

    touch_time = accessed_at or datetime.now(timezone.utc)
    backend.increment_recall_access(entry_ids, accessed_at=touch_time)


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
