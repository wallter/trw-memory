"""Vector clock conflict resolution for concurrent memory edits.

Implements FR04 (vector clocks) and FR05 (conflict resolution) from PRD-CORE-047.
Clocks are ``dict[str, int]`` mapping node_id to a monotonically increasing counter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import structlog

from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger()

MAX_MERGED_DETAIL_LENGTH = 2000


def compare_clocks(
    a: dict[str, int],
    b: dict[str, int],
) -> Literal["a_wins", "b_wins", "concurrent"]:
    """Compare two vector clocks.

    Returns:
        ``"a_wins"`` if *a* causally dominates *b* (all counters >= and at
        least one >).
        ``"b_wins"`` if *b* causally dominates *a*.
        ``"concurrent"`` if neither dominates (including equal clocks).
    """
    all_keys = set(a.keys()) | set(b.keys())

    a_gte_b = all(a.get(k, 0) >= b.get(k, 0) for k in all_keys)
    a_gt_b = any(a.get(k, 0) > b.get(k, 0) for k in all_keys)
    b_gte_a = all(b.get(k, 0) >= a.get(k, 0) for k in all_keys)
    b_gt_a = any(b.get(k, 0) > a.get(k, 0) for k in all_keys)

    if a_gte_b and a_gt_b:
        return "a_wins"
    if b_gte_a and b_gt_a:
        return "b_wins"
    return "concurrent"


def increment_clock(
    clock: dict[str, int],
    node_id: str,
) -> dict[str, int]:
    """Increment the local node's counter in a vector clock.

    Returns a new dict — the original is not mutated.
    """
    new_clock = dict(clock)
    new_clock[node_id] = new_clock.get(node_id, 0) + 1
    return new_clock


def init_clock(node_id: str) -> dict[str, int]:
    """Initialize a new vector clock with counter 1 for *node_id*."""
    return {node_id: 1}


def merge_clocks(
    a: dict[str, int],
    b: dict[str, int],
) -> dict[str, int]:
    """Merge two vector clocks by taking the max counter for each node."""
    all_keys = set(a.keys()) | set(b.keys())
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in all_keys}


def resolve_conflict(
    local: MemoryEntry,
    remote: MemoryEntry,
) -> MemoryEntry:
    """Resolve a conflict between local and remote versions of a memory entry.

    Rules (FR05):

    * **Causal order** — if one clock dominates, that version wins.
    * **Concurrent** — merge: local content preferred, details concatenated,
      importance = max, tags = sorted union, clock = merged max.
    """
    ordering = compare_clocks(local.vector_clock, remote.vector_clock)

    if ordering == "a_wins":
        return local
    if ordering == "b_wins":
        return remote

    # Concurrent: merge
    merged_detail = local.detail
    if remote.detail and remote.detail != local.detail:
        combined = f"{local.detail}\n\n---\n\n{remote.detail}"
        merged_detail = combined[:MAX_MERGED_DETAIL_LENGTH]

    merged_tags = sorted(set(local.tags) | set(remote.tags))
    merged_clock = merge_clocks(local.vector_clock, remote.vector_clock)
    merged_importance = max(local.importance, remote.importance)

    # Build merged_from tracking
    merged_from = list(set(local.merged_from + remote.merged_from))

    now = datetime.now(timezone.utc).isoformat()
    outcome = f"conflict_merged:local={local.id}:remote={remote.id}:timestamp={now}"

    # Use local as the base, update with merged values
    merged = local.model_copy(
        update={
            "detail": merged_detail,
            "tags": merged_tags,
            "vector_clock": merged_clock,
            "importance": round(merged_importance, 4),
            "merged_from": merged_from,
            "outcome_history": [*local.outcome_history, outcome],
            "updated_at": datetime.now(timezone.utc),
        },
    )

    return merged
