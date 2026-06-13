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

__all__ = ["blend_recency", "combmax_fuse", "rrf_fuse"]

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
    k: int = 60,
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
        k: RRF smoothing constant (default 60, matching the paper canonical
            default). Use the same value as rrf_fuse for a fair comparison;
            pipeline passes rrf_k so both modes use consistent smoothing.

    Returns:
        Fused list of ``(entry_id, score)`` pairs sorted by score descending,
        where ``score(d) = max_i 1 / (k + rank_i(d))``.
    """
    if not rankings:
        return []

    if k < 1:
        logger.warning("combmax_k_invalid", k=k, default=60)
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


def blend_recency(
    fused: list[tuple[str, float]],
    *,
    recency_results: list[tuple[str, float]],
    recency_weight: float,
) -> list[tuple[str, float]]:
    """Blend relevance-fused scores with recency using linear interpolation.

    Mirrors the production hybrid_search recency blend path:
        final = (1 - w) * normalised_relevance + w * recency_score

    The fused relevance scores are normalised to ``[0, 1]`` (divided by the max
    fused score in the run) so they share a scale with the recency scores before
    blending; the recency scores are expected to already lie in ``[0, 1]``.

    When ``recency_weight <= 0``, returns *fused* unchanged.
    When *recency_results* (or *fused*) is empty, returns *fused* unchanged.
    Entries that appear in *recency_results* but not in *fused* are appended
    after all fused entries at their recency-blended score.

    Ties on the blended score are broken deterministically by (1) original
    relevance rank, then (2) recency rank, then (3) ``entry_id``.

    Args:
        fused: Relevance-fused ``(entry_id, score)`` pairs ordered by relevance
            descending (e.g. the output of :func:`rrf_fuse`).
        recency_results: ``(entry_id, recency_score)`` pairs ordered by recency
            descending, with recency scores in ``[0, 1]``.
        recency_weight: Blend weight ``w`` on recency vs. normalised relevance.

    Returns:
        Blended list of ``(entry_id, score)`` pairs sorted by score descending.
    """
    if recency_weight <= 0.0 or not fused or not recency_results:
        return fused

    rel_max = max(score for _, score in fused)
    rel_norm = {entry_id: score / rel_max for entry_id, score in fused} if rel_max > 0.0 else {}
    rec_map = dict(recency_results)
    rel_rank = {entry_id: rank for rank, (entry_id, _) in enumerate(fused)}
    rec_rank = {entry_id: rank for rank, (entry_id, _) in enumerate(recency_results)}

    all_ids = [entry_id for entry_id, _ in fused]
    all_ids.extend(entry_id for entry_id, _ in recency_results if entry_id not in rel_rank)

    blended = [
        (
            entry_id,
            (1.0 - recency_weight) * rel_norm.get(entry_id, 0.0) + recency_weight * rec_map.get(entry_id, 0.0),
        )
        for entry_id in all_ids
    ]
    rank_sentinel = len(all_ids) + 1
    blended.sort(
        key=lambda x: (
            -x[1],
            rel_rank.get(x[0], rank_sentinel),
            rec_rank.get(x[0], rank_sentinel),
            x[0],
        )
    )

    logger.debug(
        "blend_recency_complete",
        fused_count=len(fused),
        recency_count=len(recency_results),
        recency_weight=recency_weight,
        unique_docs=len(blended),
    )
    return blended
