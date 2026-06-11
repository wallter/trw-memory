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

__all__ = ["combmax_fuse", "rrf_fuse"]

logger = structlog.get_logger(__name__)


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    *,
    importances: dict[str, float] | None = None,
    alpha: float = 1.0,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    Implements the RRF formula:
        score(d) = Σ 1 / (k + rank_i(d))
    where ``rank_i`` is the **1-based** rank of document *d* in ranking list *i*.

    Documents that do not appear in a given ranking contribute nothing to the
    sum for that ranking.  Documents that appear in multiple rankings
    accumulate a score from each.

    R-FUSION-001: the bare RRF score is **position-only** — it discards the
    entry's importance/impact, so two results at the same fused rank tie even
    when one is impact-0.95 tribal knowledge and the other is impact-0.2 noise.
    When *importances* is supplied and ``alpha < 1.0``, the per-document RRF
    score is normalised to ``[0, 1]`` (divided by the max RRF score in the run)
    and blended with the entry's importance:

        final(d) = alpha * rrf_norm(d) + (1 - alpha) * importance(d)

    ``alpha=1.0`` (the default) preserves the legacy pure-position behaviour
    bit-for-bit, so existing callers are unaffected.

    Args:
        rankings: List of ranked result lists.  Each inner list is a sequence
            of ``(entry_id, score)`` pairs ordered by relevance descending.
            The individual scores are ignored — only rank position matters.
        k: RRF smoothing constant.  The default value of 60 is from the
            original paper and works well in practice.
        importances: Optional mapping of ``entry_id`` → importance/impact in
            ``[0, 1]``.  Ignored when ``alpha >= 1.0``.  Missing ids default
            to 0.0 importance.
        alpha: Blend weight on the (normalised) RRF position score vs.
            importance.  ``1.0`` = pure position (legacy), ``0.0`` = pure
            importance.  Clamped to ``[0, 1]``.

    Returns:
        Fused list of ``(entry_id, score)`` pairs sorted by score descending.
        Returns an empty list when *rankings* is empty or all inner lists are
        empty.
    """
    if not rankings:
        return []

    if k < 1:
        logger.warning("rrf_k_invalid", k=k, default=60)
        k = 60

    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (entry_id, _) in enumerate(ranking):
            fused_scores[entry_id] = fused_scores.get(entry_id, 0.0) + 1.0 / (k + rank + 1)

    blend_alpha = max(0.0, min(1.0, alpha))
    if importances is not None and blend_alpha < 1.0:
        # Normalise RRF to [0, 1] so it shares a scale with importance before
        # blending; otherwise the tiny absolute RRF magnitudes (~1/61) would
        # let importance dominate unconditionally regardless of alpha.
        max_rrf = max(fused_scores.values()) if fused_scores else 0.0
        if max_rrf > 0.0:
            for entry_id, rrf_score in fused_scores.items():
                rrf_norm = rrf_score / max_rrf
                imp = importances.get(entry_id, 0.0)
                fused_scores[entry_id] = blend_alpha * rrf_norm + (1.0 - blend_alpha) * imp

    result = list(fused_scores.items())
    result.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "rrf_fuse_complete",
        ranking_count=len(rankings),
        unique_docs=len(result),
        importance_blended=importances is not None and blend_alpha < 1.0,
    )
    return result


def combmax_fuse(
    rankings: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """CombMAX rank fusion for hard-tail recall improvement.

    Assigns each document the MAXIMUM reciprocal-rank score it achieves across
    any single input ranking, rather than the sum used by RRF.  When two
    retrievers each have a strong individual champion that the other misses,
    CombMAX preserves both at their individual peak; RRF-sum dilutes them.

    Regime note (MEMORY.md rca_rank_fusion_combiner): CombMAX lifts
    hard-tail recall significantly (n=12: recall@12 0.583→0.750,
    McNemar p=0.0074) versus RRF-sum.  In easy regimes where content
    dominates both lists the difference is negligible, so this function is
    offered as a configurable alternative — the pipeline default remains
    ``rrf_fuse``.

    Args:
        rankings: List of ranked result lists.  Each inner list is a sequence
            of ``(entry_id, score)`` pairs ordered by relevance descending.
            Scores are ignored — only 1-based rank position matters.

    Returns:
        Fused list of ``(entry_id, score)`` pairs sorted by score descending,
        where ``score(d) = max_i 1 / (60 + rank_i(d))`` with the standard
        k=60 smoothing constant.
    """
    if not rankings:
        return []

    k = 60
    best_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (entry_id, _) in enumerate(ranking):
            rr = 1.0 / (k + rank + 1)
            if rr > best_scores.get(entry_id, 0.0):
                best_scores[entry_id] = rr

    result = list(best_scores.items())
    result.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "combmax_fuse_complete",
        ranking_count=len(rankings),
        unique_docs=len(result),
    )
    return result
