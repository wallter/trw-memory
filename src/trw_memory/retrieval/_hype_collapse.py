"""HyPE recall-path parent collapse (PRD-CORE-195 FR04).

Dense search over ``vec_memories`` can return synthetic ``{parent_id}#hype{n}``
ids when HyPE sibling vectors are present in the candidate pool. Before fusion,
those hits must collapse back to their parent entry id, deduped so a parent
reached via several siblings (and/or its own primary vector) is counted exactly
once at its BEST rank. Parents absent from the live ``entry_map`` (forgotten /
superseded / filtered out) are dropped so a stale sibling never resurrects a
parent that is no longer eligible, and no synthetic id ever leaks to the caller.

The collapse runs pre-fusion and only ever emits real parent ids, so all
downstream stages (RRF/CombMAX, recency blend, ``apply_validity_prior``,
rerank) operate on real entries — composing cleanly with PRD-CORE-194.
"""

from __future__ import annotations

from trw_memory._hype_ids import is_hype_id, parent_of_hype_id


def hype_sibling_ids_in(
    stored_embeddings: dict[str, list[float]] | None,
    valid_parent_ids: set[str] | None = None,
) -> list[str]:
    """Return the synthetic ``#hype`` ids present in *stored_embeddings*."""
    if not stored_embeddings:
        return []
    canonical_ids = valid_parent_ids or set()
    return [eid for eid in stored_embeddings if eid not in canonical_ids and is_hype_id(eid)]


def collapse_hype_ranking(
    ranking: list[tuple[str, float]],
    valid_parent_ids: set[str],
) -> tuple[list[tuple[str, float]], int]:
    """Collapse ``#hype`` ids in *ranking* to their parent ids, deduped by rank.

    Iterates *ranking* in order (rank ascending). Each id is mapped to its
    parent (a non-HyPE id maps to itself). The FIRST occurrence of a parent
    wins, so a parent keeps its best (lowest-index) rank position — equivalent
    to max reciprocal-rank since reciprocal rank is monotone decreasing in
    position. Ids whose parent is not in *valid_parent_ids* are dropped.

    Args:
        ranking: ``(entry_id, score)`` pairs ordered by relevance descending.
            May contain synthetic ``#hype`` ids.
        valid_parent_ids: Ids of real, eligible parent entries. A collapsed id
            absent from this set is dropped (orphan / superseded sibling).

    Returns:
        ``(collapsed_ranking, collapsed_hit_count)`` where *collapsed_ranking*
        is a deduped ``(parent_id, score)`` list preserving best-rank order, and
        *collapsed_hit_count* is how many input rows were synthetic ``#hype``
        hits (for telemetry).
    """
    seen: set[str] = set()
    collapsed: list[tuple[str, float]] = []
    collapsed_hits = 0
    for entry_id, score in ranking:
        if entry_id in valid_parent_ids:
            parent_id = entry_id
        elif is_hype_id(entry_id):
            collapsed_hits += 1
            parent_id = parent_of_hype_id(entry_id)
        else:
            parent_id = entry_id
        if parent_id in seen:
            continue
        if parent_id not in valid_parent_ids:
            continue
        seen.add(parent_id)
        collapsed.append((parent_id, score))
    return collapsed, collapsed_hits
