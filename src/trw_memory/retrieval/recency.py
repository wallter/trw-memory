"""Recency-based ranking for trw-memory.

Ranks :class:`~trw_memory.models.memory.MemoryEntry` objects by how recently
they became valid, using exponential decay on the ``valid_from`` timestamp.
This ranking can be fed as a third source into RRF alongside BM25 and dense
retrieval, giving temporal queries a signal that neither text-matching nor
semantic similarity can provide.

Design rationale
----------------
Temporal queries ("most recent X", "what happened last week") are the
discrimination band for trw-memory retrieval — they score ~0.853 vs 0.91+ for
exact/semantic queries because RRF is purely position-based with no timestamp
awareness.  Adding recency as an explicit retrieval source lets RRF naturally
combine text relevance + semantic similarity + freshness without requiring a
separate post-processing step.

The decay function is:
    recency_score(entry) = exp(-ln(2) * days_since_valid_from / halflife_days)

A ``halflife_days`` of 14 means an entry from 14 days ago gets score 0.5; one
from today gets score ~1.0; one from 90 days ago gets ~0.125.  The halflife is
configurable to match different deployment profiles (short-lived session memory
vs. long-lived institutional knowledge).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import structlog

from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

# Default half-life: entries lose half their recency score after 14 days.
# Keep aligned with MemoryConfig.recall_recency_halflife_days so direct
# recency_rank() callers and MemoryClient.recall() use the same policy.
DEFAULT_HALFLIFE_DAYS: float = 14.0

# Minimum recency score floor — prevents ancient entries from getting score 0
# and being permanently excluded from recency-weighted fusions.
_MIN_RECENCY_SCORE: float = 1e-6


def _entry_timestamp(entry: MemoryEntry) -> datetime:
    """Resolve the most meaningful timestamp for recency ranking.

    Prefer ``valid_from`` (event time) over ``created_at`` (ingest time) so
    bi-temporal records are ranked by when the fact became true in the world,
    not when it was ingested.
    """
    return entry.valid_from


def recency_score(entry: MemoryEntry, now: datetime, halflife_days: float) -> float:
    """Compute the recency decay score for a single entry.

    Args:
        entry: The memory entry to score.
        now: Reference instant (UTC).
        halflife_days: Days until the score halves.

    Returns:
        Score in ``(_MIN_RECENCY_SCORE, 1.0]`` — 1.0 for a brand-new entry,
        decaying exponentially toward the floor as age increases.
    """
    ts = _entry_timestamp(entry)
    days_ago = max(0.0, (now - ts).total_seconds() / 86400.0)
    decay = math.exp(-math.log(2.0) * days_ago / halflife_days) if halflife_days > 0 else 0.0
    return max(_MIN_RECENCY_SCORE, decay)


def recency_rank(
    entries: list[MemoryEntry],
    *,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    now: datetime | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank entries by recency using exponential decay on ``valid_from``.

    Returns a list of ``(entry_id, recency_score)`` pairs sorted by score
    descending (most recent first), suitable for feeding directly into
    :func:`~trw_memory.retrieval.fusion.rrf_fuse` alongside BM25 and dense
    rankings.

    Args:
        entries: Candidate entries to rank.
        halflife_days: Decay half-life in days.  At this age the recency
            score is 0.5 relative to a brand-new entry.  Smaller values
            favour very recent entries more aggressively; larger values
            create a flatter recency curve.
        now: Reference instant for age computation.  Defaults to
            ``datetime.now(timezone.utc)`` — pass an explicit value in tests
            for reproducibility.
        top_k: When set, return only the top-K most recent entries.

    Returns:
        List of ``(entry_id, score)`` pairs sorted by score descending.
        Empty entries (no ``id``) are silently skipped.
    """
    if not entries:
        return []

    ref = now if now is not None else datetime.now(timezone.utc)
    pairs: list[tuple[str, float]] = []
    for entry in entries:
        if not entry.id:
            continue
        score = recency_score(entry, ref, halflife_days)
        pairs.append((entry.id, score))

    pairs.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "recency_rank_complete",
        entry_count=len(entries),
        halflife_days=halflife_days,
        returned=len(pairs) if top_k is None else min(len(pairs), top_k),
    )

    return pairs[:top_k] if top_k is not None else pairs
