"""Hybrid search pipeline for trw-memory.

Orchestrates BM25 sparse retrieval, dense vector search, optional recency
ranking, Reciprocal Rank Fusion, and optional cross-encoder re-ranking into a
single ``hybrid_search`` entry point.

Graceful degradation matrix:
- ``rank_bm25`` unavailable → BM25 step skipped
- ``embedder`` is ``None`` or unavailable → dense step skipped
- Both unavailable → returns empty list
- Only one source available → uses that source directly (no fusion needed)
- ``recency_weight > 0`` → blend normalised relevance with recency score
- ``rerank=True`` → cross-encoder re-ranking applied post-fusion (requires
  sentence-transformers; gracefully skipped when unavailable)
"""

from __future__ import annotations

from datetime import datetime

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import dense_search
from trw_memory.retrieval.fusion import combmax_fuse, rrf_fuse
from trw_memory.retrieval.recency import recency_rank
from trw_memory.retrieval.validity_prior import apply_validity_prior

logger = structlog.get_logger(__name__)


def hybrid_search(
    query: str,
    entries: list[MemoryEntry],
    *,
    embedder: EmbeddingProvider | None = None,
    query_embedding: list[float] | None = None,
    stored_embeddings: dict[str, list[float]] | None = None,
    bm25_candidates: int = 50,
    vector_candidates: int = 50,
    # rrf_k=15 (was 60): promoted 2026-06-12 by the memory meta-harness loop —
    # sharper rank decay improved nDCG@10/recall@10 on two independent evals
    # (real-learnings gold set, paired held-out t=2.48 n=294; LongMemEval_S
    # full-500 recall@5 0.904→0.914) with the temporal slice preserved. Tuned
    # with all-MiniLM-L6-v2 on note/session corpora; the rrf_fuse primitive
    # keeps the paper-canonical k=60 default.
    rrf_k: int = 15,
    importance_alpha: float = 1.0,
    top_k: int = 25,
    fusion_mode: str = "rrf",
    as_of: datetime | None = None,
    include_superseded: bool = False,
    validity_age_decay: bool = False,
    recency_weight: float = 0.0,
    recency_halflife_days: float = 30.0,
    recency_now: datetime | None = None,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_candidates: int = 50,
    rerank_query: str | None = None,
) -> list[MemoryEntry]:
    """Hybrid BM25 + vector search with configurable rank fusion.

    Runs BM25 and dense retrieval in sequence then fuses their rankings using
    the selected fusion strategy.  Either retrieval path is skipped when its
    dependency is unavailable, allowing the pipeline to degrade to single-
    source search without raising.

    Graceful degradation:
    - ``rank_bm25`` not installed → BM25 skipped
    - *embedder* is ``None`` or ``embedder.available()`` is ``False`` →
      dense search skipped
    - Both skipped → returns ``[]``
    - Only one source produces results → fusion is a no-op (passthrough)

    Args:
        query: Free-text search query.
        entries: Candidate :class:`~trw_memory.models.memory.MemoryEntry`
            objects to rank.  Typically the full active entry set from the
            storage backend.
        embedder: Optional embedding provider used for dense search.  When
            ``None`` or unavailable the dense path is skipped (unless
            *query_embedding* is supplied).
        query_embedding: Pre-computed query vector forwarded to
            :func:`~trw_memory.retrieval.dense.dense_search`.  When supplied the
            dense path reuses it instead of calling ``embedder.embed(query)``,
            avoiding a redundant embedding pass when the caller already computed
            the query vector (e.g. for tier scoring).  ``None`` (the default)
            preserves the legacy behaviour of embedding the query inside the
            dense step bit-for-bit.
        stored_embeddings: Mapping of ``entry_id`` → embedding vector.
            Required for dense search; dense path is skipped when ``None`` or
            empty.
        bm25_candidates: Maximum BM25 candidates passed to
            :func:`~trw_memory.retrieval.bm25.bm25_search`.
        vector_candidates: Maximum dense candidates passed to
            :func:`~trw_memory.retrieval.dense.dense_search`.
        rrf_k: RRF smoothing constant forwarded to
            :func:`~trw_memory.retrieval.fusion.rrf_fuse` (ignored when
            *fusion_mode* is ``"combmax"``).
        importance_alpha: R-FUSION-001 blend weight on the normalised RRF
            position score vs. the candidate's ``importance``. ``1.0`` (the
            default) preserves pure position-only fusion; lower values let a
            high-impact entry edge out an equally-ranked low-impact one.
            Ignored when *fusion_mode* is ``"combmax"``.
        top_k: Final number of entries to return after fusion.
        fusion_mode: Fusion algorithm to use.  ``"rrf"`` (the default) uses
            Reciprocal Rank Fusion (sum of reciprocal ranks).  ``"combmax"``
            uses CombMAX (max reciprocal rank per document), which lifts
            hard-tail recall@12 by ~28% (0.583→0.750, McNemar p=0.0074) at
            the cost of weaker cross-list boosting.  Unknown values fall back
            to ``"rrf"`` with a warning.
        recency_weight: When > 0, blend normalised relevance with the
            exponential half-life recency score using this fraction as the
            freshness weight; ``0.0`` (default) disables recency ranking
            completely, preserving pure text-relevance behaviour.  Values up to
            ``1.0`` are meaningful; ``0.3`` is a reasonable starting point for
            temporal query workloads.
        recency_halflife_days: Decay half-life used by the recency ranker.
            An entry ``halflife_days`` old receives score 0.5 relative to a
            brand-new entry.  Default ``30.0`` days.  Ignored when
            ``recency_weight == 0``.
        recency_now: Reference instant for age computation.  ``None`` (the
            default) resolves to ``datetime.now(timezone.utc)`` inside
            :func:`~trw_memory.retrieval.recency.recency_rank`.  Pass an
            explicit value when the "now" of the query differs from wall-clock
            time — e.g. when replaying historical queries or when entries were
            recorded in the past and the caller knows the evaluation reference
            point.  Ignored when ``recency_weight == 0``.
        rerank: When ``True``, apply cross-encoder re-ranking after fusion
            to re-score the top ``rerank_candidates`` entries jointly on
            (query, passage).  Requires ``sentence-transformers``; silently
            falls back to fusion order when the dependency or model is
            unavailable.
        rerank_model: HuggingFace model id for cross-encoder re-ranking.
            Default ``"cross-encoder/ms-marco-MiniLM-L-6-v2"`` (66M params,
            MS MARCO passage re-ranker).  Ignored when ``rerank=False``.
        rerank_candidates: Number of top-fusion candidates to pass to the
            cross-encoder.  Re-ranking all candidates is expensive; limiting
            to the top-50 captures the quality gain at reasonable latency.
            Ignored when ``rerank=False``.

    Returns:
        Up to *top_k* :class:`~trw_memory.models.memory.MemoryEntry` objects
        ordered by fused (and optionally re-ranked) relevance score descending.
    """
    if not entries:
        return []

    # Index entries by id for fast lookup after fusion
    entry_map: dict[str, MemoryEntry] = {e.id: e for e in entries}
    entry_ids: list[str] = list(entry_map.keys())

    rankings: list[list[tuple[str, float]]] = []

    # ---------------------------------------------------------------- BM25
    bm25_results = bm25_search(query, entries, top_k=bm25_candidates)
    if bm25_results:
        rankings.append(bm25_results)

    # -------------------------------------------------------------- Dense
    if embedder is not None or query_embedding is not None or stored_embeddings:
        dense_results = dense_search(
            query=query,
            entry_ids=entry_ids,
            embedder=embedder,
            query_embedding=query_embedding,
            stored_embeddings=stored_embeddings,
            top_k=vector_candidates,
        )
        if dense_results:
            rankings.append(dense_results)

    if not rankings:
        logger.debug(
            "hybrid_search_no_results",
            query=query[:80],
            entry_count=len(entries),
        )
        return []

    # --------------------------------------------------------------- Fusion
    # R-FUSION-001: blend the entry's importance into the position-only RRF
    # score so two equally-ranked candidates are broken by impact. alpha=1.0
    # (default) keeps the legacy pure-position behaviour bit-for-bit.
    # combmax_fuse is a configurable alternative that lifts hard-tail recall
    # (MEMORY.md rca_rank_fusion_combiner); default unchanged.
    if fusion_mode == "combmax":
        relevance_fused = combmax_fuse(rankings, k=rrf_k)
    else:
        if fusion_mode != "rrf":
            logger.warning("hybrid_search_unknown_fusion_mode", fusion_mode=fusion_mode, fallback="rrf")
        importances = {e.id: e.importance for e in entries} if importance_alpha < 1.0 else None
        relevance_fused = rrf_fuse(rankings, k=rrf_k, importances=importances, alpha=importance_alpha)

    recency_results: list[tuple[str, float]] = []

    # ------------------------------------------------------------- Recency blend
    # When recency_weight > 0, compute a separate recency score (exponential half-life
    # decay on valid_from) and LINEARLY BLEND it with the relevance-fused score:
    #
    #   final(d) = (1 - w) * relevance_norm(d) + w * recency_score(d)
    #
    # Both sides are normalised to [0,1] independently before blending so the
    # recency_weight value is a true proportion (0.3 = "30% freshness, 70% relevance").
    # Adding recency as an equal 1/3 RRF source (previous approach) produced a
    # binary switch: all non-zero weights had identical output and nDCG dropped
    # uniformly because time-ordering overrode relevance for non-temporal queries.
    if recency_weight > 0.0:
        recency_results = recency_rank(
            entries,
            halflife_days=recency_halflife_days,
            now=recency_now,
        )
        if recency_results:
            # Normalise relevance scores to [0, 1]
            rel_max = max(s for _, s in relevance_fused) if relevance_fused else 1.0
            rel_norm = {eid: s / rel_max for eid, s in relevance_fused} if rel_max > 0 else {}

            # Recency scores are already in (0, 1] from the half-life formula.
            rec_map = dict(recency_results)
            rel_rank = {eid: rank for rank, (eid, _) in enumerate(relevance_fused)}
            rec_rank = {eid: rank for rank, (eid, _) in enumerate(recency_results)}

            # Blend: cover all entries from either source while preserving a
            # deterministic pre-sort order. Building this from a set made equal
            # final scores depend on PYTHONHASHSEED, which perturbed downstream
            # validity-prior tie buckets.
            all_ids = [eid for eid, _ in relevance_fused]
            all_ids.extend(eid for eid, _ in recency_results if eid not in rel_rank)
            blended: list[tuple[str, float]] = []
            for eid in all_ids:
                rel_s = rel_norm.get(eid, 0.0)
                rec_s = rec_map.get(eid, 0.0)
                blended.append((eid, (1.0 - recency_weight) * rel_s + recency_weight * rec_s))
            rank_sentinel = len(all_ids) + 1
            blended.sort(
                key=lambda x: (
                    -x[1],
                    rel_rank.get(x[0], rank_sentinel),
                    rec_rank.get(x[0], rank_sentinel),
                    x[0],
                )
            )
            fused = blended
        else:
            fused = relevance_fused
    else:
        fused = relevance_fused

    fused_scores = dict(fused)

    # Map fused ids back to MemoryEntry objects, preserving fusion order.
    fused_entries: list[MemoryEntry] = []
    for entry_id, _ in fused:
        entry = entry_map.get(entry_id)
        if entry is not None:
            fused_entries.append(entry)

    # PRD-CORE-194 FR03: apply the validity prior as a POST-FUSION pass (in-memory
    # field compare, no extra query; NFR04). It excludes superseded records by
    # default, re-scopes by ``as_of``, positionally appends superseded ones when
    # ``include_superseded`` (so they never outrank an open record), and applies a
    # bounded age advantage. Fusion order is otherwise preserved. ``top_k`` is
    # applied AFTER the prior so excluded records do not consume result slots.
    fused_entries = apply_validity_prior(
        fused_entries,
        as_of=as_of,
        include_superseded=include_superseded,
        age_decay=validity_age_decay,
        fusion_scores=fused_scores,
    )

    # --------------------------------------------------------- Re-ranking
    # Optional cross-encoder re-ranking: score top-N candidates jointly as
    # (query, passage) pairs.  This captures finer-grained relevance than
    # bi-encoder + RRF at the cost of O(rerank_candidates) model calls.
    if rerank and fused_entries:
        from trw_memory.retrieval.reranker import cross_encode_rerank

        # Use rerank_query (original un-preprocessed query) when supplied so
        # the cross-encoder receives the full user intent even if the search
        # query was stripped of temporal boilerplate.
        effective_rerank_query = rerank_query if rerank_query is not None else query
        rerank_input = fused_entries[:rerank_candidates]
        tail = fused_entries[rerank_candidates:]
        rerank_input = cross_encode_rerank(effective_rerank_query, rerank_input, model_name=rerank_model)
        fused_entries = [*rerank_input, *tail]

    results: list[MemoryEntry] = fused_entries[:top_k]

    logger.debug(
        "hybrid_search_complete",
        query=query[:80],
        entry_count=len(entries),
        bm25_hits=len(bm25_results) if bm25_results else 0,
        recency_hits=len(recency_results),
        fused_total=len(fused),
        returned=len(results),
        fusion_mode=fusion_mode,
        recency_weight=recency_weight,
        rerank=rerank,
    )
    return results
