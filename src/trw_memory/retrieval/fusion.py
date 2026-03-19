"""Reciprocal Rank Fusion (RRF) for trw-memory.

Combines multiple ranked result lists into a single fused ranking using the
RRF formula from Cormack, Clarke & Buettcher (2009):

    score(d) = Σ_i  1 / (k + rank_i(d))

where ``rank_i`` is the **1-based** position of document *d* in ranking *i*
and ``k`` is a smoothing constant (default 60, recommended in the paper).

This module is intentionally minimal — a single pure function with no
external dependencies beyond the standard library.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    Implements the RRF formula:
        score(d) = Σ 1 / (k + rank_i(d))
    where ``rank_i`` is the **1-based** rank of document *d* in ranking list *i*.

    Documents that do not appear in a given ranking contribute nothing to the
    sum for that ranking.  Documents that appear in multiple rankings
    accumulate a score from each.

    Args:
        rankings: List of ranked result lists.  Each inner list is a sequence
            of ``(entry_id, score)`` pairs ordered by relevance descending.
            The individual scores are ignored — only rank position matters.
        k: RRF smoothing constant.  The default value of 60 is from the
            original paper and works well in practice.

    Returns:
        Fused list of ``(entry_id, rrf_score)`` pairs sorted by RRF score
        descending.  Returns an empty list when *rankings* is empty or all
        inner lists are empty.
    """
    if not rankings:
        return []

    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (entry_id, _) in enumerate(ranking):
            fused_scores[entry_id] = fused_scores.get(entry_id, 0.0) + 1.0 / (k + rank + 1)

    result = list(fused_scores.items())
    result.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "rrf_fuse_complete",
        ranking_count=len(rankings),
        unique_docs=len(result),
    )
    return result
