"""Hybrid recall pipeline — BM25 + dense + RRF, with latency telemetry.

Belongs to ``client.py`` recall pipeline. Re-exported via
``_client_recall.py`` so the ``MemoryClient._try_hybrid_recall`` delegator
keeps its ``from trw_memory._client_recall import try_hybrid_recall``
import unchanged. Split out from the parent recall module so each file
stays under the 350 effective-LOC gate (PRD-DIST-246; loc-tracker
self-improve split).

This is a deep module: the public ``try_hybrid_recall`` interface is
narrow (one async call returning ``list[MemoryResultDict] | None``, where
``None`` signals "fall back to LIKE + TF scoring") while the
implementation hides candidate-pool sizing, namespace-aware BM25/vector
candidate auto-scaling, RRF top-K depth, and per-recall latency/shape
telemetry.

Public surface (delegated from ``MemoryClient._try_hybrid_recall``):

- ``try_hybrid_recall`` — async BM25 + dense + RRF pipeline; returns
  ``None`` to signal the caller should fall back to LIKE + TF scoring.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict
    from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)


def _entry_to_result(entry: MemoryEntry, score: float = 0.0) -> MemoryResultDict:
    from trw_memory._client_distilled_tiering import entry_to_result as _impl

    return _impl(entry, score=score)


async def try_hybrid_recall(
    client: MemoryClient,
    query: str,
    limit: int,
    tags: list[str] | None,
    query_embedding: list[float] | None = None,
    *,
    as_of: datetime | None = None,
    include_superseded: bool = False,
) -> list[MemoryResultDict] | None:
    """Hybrid pipeline (BM25 + dense + RRF). Returns None to signal fallback.

    PRD-DIST-2047 Phase 2 (recall-latency telemetry): emits a structlog event
    ``hybrid_recall_complete`` carrying per-call timings + namespace shape +
    effective candidate caps + returned-result count, so operators can right-
    size ``hybrid_search_candidate_pool_size`` against measured cost. The
    event fires on every terminating exit (success, no-candidates, hybrid-
    search-failed) so operators can attribute latency to outcome.

    *query_embedding* is the query vector the caller already computed (for tier
    scoring); when supplied it is forwarded to ``hybrid_search`` so the dense
    step reuses it instead of re-embedding the query. ``None`` preserves the
    legacy behaviour of embedding the query inside the dense step.
    """
    try:
        from trw_memory.retrieval.pipeline import hybrid_search
    except ImportError:
        return None

    total_start = perf_counter()

    async with client._lock:
        backend = client._get_backend()
        # PRD-DIST-2047 c796: load up to hybrid_search_candidate_pool_size
        # entries (default 1000) so BM25 + dense can rank the full namespace.
        # Pre-c796 the pool was capped at limit*5 (=50 for default limit=10),
        # which silently lost targets ranked past position 50 on namespaces > 50.
        candidate_pool_size = max(limit * 5, client._config.hybrid_search_candidate_pool_size)
        list_entries_start = perf_counter()
        all_entries = backend.list_entries(
            namespace=client._namespace,
            limit=candidate_pool_size,
        )
        list_entries_ms = (perf_counter() - list_entries_start) * 1000.0
        stored_embeddings = backend.get_stored_embeddings([entry.id for entry in all_entries])

    namespace_size = len(all_entries)
    if not all_entries:
        _emit_hybrid_recall_telemetry(
            outcome="no_candidates",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=0,
            effective_vector_candidates=0,
            effective_top_k=limit * client._config.recall_top_k_multiplier,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=0.0,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None

    embedder = client._get_embedder()
    # PRD-DIST-2047 c796: auto-scale bm25/vector candidate caps to namespace
    # size so the 50-default acts as a FLOOR, not a CEILING. Eliminates the
    # structural cap on recall@10 for namespaces > 50 records.
    effective_bm25_candidates = max(client._config.bm25_candidates, namespace_size)
    effective_vector_candidates = max(client._config.vector_candidates, namespace_size)

    # PRD-DIST-2050 c804: deepen the candidate pool when the admission filter
    # is opt-in enabled, so baseline records ranked past top-30 can survive the
    # filter and enter the merged top-K. Default multiplier=3 preserves pre-c804
    # behaviour (top-30); operators raise via MEMORY_RECALL_TOP_K_MULTIPLIER.
    effective_top_k = limit * client._config.recall_top_k_multiplier
    # When a tag filter is requested it is applied AFTER hybrid_search ranks and
    # truncates to top_k (below). Tag-matching entries ranked past top_k would be
    # silently dropped, reducing recall below the caller-requested limit. Rank the
    # FULL candidate pool when tags are present so the post-rank tag filter sees
    # every entry the namespace scan loaded — the tag filter then narrows back
    # down. namespace_size (== len(all_entries)) is already bounded by
    # candidate_pool_size, so this cannot widen cost beyond the entries we already
    # hold in memory.
    if tags:
        effective_top_k = max(effective_top_k, namespace_size)

    # Auto-detect temporal queries and inject recency_weight when the config
    # hasn't explicitly enabled it.  Preserves explicit config — if the operator
    # set recall_recency_weight > 0 we use that value; only the zero-default
    # case gets the auto-detected weight.
    effective_recency_weight = client._config.recall_recency_weight
    if effective_recency_weight == 0.0 and client._config.recall_auto_temporal:
        from trw_memory.retrieval.temporal_query import classify_temporal

        tc = classify_temporal(query)
        if tc.is_temporal:
            effective_recency_weight = tc.recency_weight
            logger.debug(
                "temporal_query_detected",
                query=query[:80],
                confidence=tc.confidence,
                recency_weight=effective_recency_weight,
                patterns=tc.matched_patterns,
            )

    hybrid_search_start = perf_counter()
    try:
        ranked = hybrid_search(
            query=query,
            entries=all_entries,
            embedder=embedder,
            query_embedding=query_embedding,
            stored_embeddings=stored_embeddings or None,
            bm25_candidates=effective_bm25_candidates,
            vector_candidates=effective_vector_candidates,
            rrf_k=client._config.rrf_k,
            importance_alpha=client._config.rrf_importance_alpha,
            top_k=effective_top_k,
            as_of=as_of,
            include_superseded=include_superseded,
            recency_weight=effective_recency_weight,
            recency_halflife_days=client._config.recall_recency_halflife_days,
            fusion_mode=client._config.recall_fusion_mode,
            validity_age_decay=client._config.recall_validity_age_decay,
            rerank=client._config.recall_rerank,
            rerank_model=client._config.recall_rerank_model,
            rerank_candidates=client._config.recall_rerank_candidates,
        )
    except Exception:
        hybrid_search_ms = (perf_counter() - hybrid_search_start) * 1000.0
        # warning, not debug: hybrid search failing silently drops recall to the
        # weaker fallback path with no operator-visible signal — the exact
        # silent-degradation class that let the compounding pipeline rot.
        logger.warning(
            "hybrid_search_failed",
            op="recall",
            outcome="failure",
            exc_info=True,
        )
        _emit_hybrid_recall_telemetry(
            outcome="hybrid_search_failed",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=effective_bm25_candidates,
            effective_vector_candidates=effective_vector_candidates,
            effective_top_k=effective_top_k,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=hybrid_search_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None
    hybrid_search_ms = (perf_counter() - hybrid_search_start) * 1000.0

    if not ranked:
        _emit_hybrid_recall_telemetry(
            outcome="empty_ranking",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=effective_bm25_candidates,
            effective_vector_candidates=effective_vector_candidates,
            effective_top_k=effective_top_k,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=hybrid_search_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None

    if tags:
        tag_set = set(tags)
        ranked = [e for e in ranked if tag_set.issubset(set(e.tags))]

    results: list[MemoryResultDict] = []
    for rank, entry in enumerate(ranked):
        score = round(1.0 / (1 + rank), 4)
        results.append(_entry_to_result(entry, score=score))

    _emit_hybrid_recall_telemetry(
        outcome="ok",
        namespace=client._namespace,
        namespace_size=namespace_size,
        candidate_pool_size=candidate_pool_size,
        effective_bm25_candidates=effective_bm25_candidates,
        effective_vector_candidates=effective_vector_candidates,
        effective_top_k=effective_top_k,
        returned_count=len(results),
        list_entries_ms=list_entries_ms,
        hybrid_search_ms=hybrid_search_ms,
        total_ms=(perf_counter() - total_start) * 1000.0,
    )
    return results


def _emit_hybrid_recall_telemetry(
    *,
    outcome: str,
    namespace: str,
    namespace_size: int,
    candidate_pool_size: int,
    effective_bm25_candidates: int,
    effective_vector_candidates: int,
    effective_top_k: int,
    returned_count: int,
    list_entries_ms: float,
    hybrid_search_ms: float,
    total_ms: float,
) -> None:
    """PRD-DIST-2047 Phase 2: emit a per-recall latency + shape event.

    Operators sample this event stream to right-size
    ``hybrid_search_candidate_pool_size`` for very large namespaces (where
    BM25 cost grows linearly with namespace_size). Latencies are reported in
    milliseconds rounded to 3 decimals.
    """
    logger.info(
        "hybrid_recall_complete",
        op="recall",
        outcome=outcome,
        namespace=namespace,
        namespace_size=namespace_size,
        candidate_pool_size=candidate_pool_size,
        effective_bm25_candidates=effective_bm25_candidates,
        effective_vector_candidates=effective_vector_candidates,
        effective_top_k=effective_top_k,
        returned_count=returned_count,
        list_entries_ms=round(list_entries_ms, 3),
        hybrid_search_ms=round(hybrid_search_ms, 3),
        total_ms=round(total_ms, 3),
    )
