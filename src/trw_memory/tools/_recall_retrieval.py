"""Retrieval stage of the ``memory_recall`` tool path.

Belongs to the ``tools/recall.py`` facade; called from ``memory_recall_impl``.
Extracted by PRD-CORE-278 when carrying the retrieval score through the tool
path pushed ``recall.py`` past the 350 effective-LOC gate.

The contract worth stating: this is where a candidate stops being a stored row
and becomes a RANKED row. It runs the temporal rewrite, sizes the candidate
caps, calls the scored retrieval boundary, and writes each candidate's score
onto its serialised dict under ``FUSED_SCORE_KEY`` — the single point where the
pipeline's ranking becomes the caller-visible ``score``. Everything downstream
of it preserves that number rather than recomputing one.

``hybrid_search_scored`` is looked up through the parent module rather than
imported directly, so a test that patches ``trw_memory.tools.recall.
hybrid_search_scored`` still reaches the call it is aiming at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from trw_memory.embeddings._query_prompts import embed_query
from trw_memory.lifecycle._recall import FUSED_SCORE_KEY
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.models.config import MemoryConfig

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.security.namespace_scope import NamespaceScope

logger = structlog.get_logger(__name__)

__all__ = ["build_scored_candidates"]


def build_scored_candidates(
    query: str,
    all_entries: list[MemoryEntry],
    *,
    cfg: MemoryConfig,
    scope: NamespaceScope,
    embedder: EmbeddingProvider | None,
    stored_embeddings: dict[str, list[float]],
    limit: int,
    tags: list[str] | None,
) -> tuple[list[dict[str, object]], list[float] | None]:
    """Rank the candidate pool and return (scored entry dicts, query embedding).

    The query embedding is returned because tier scoring downstream must use the
    SAME vector the dense step used; recomputing it there would score the tier
    candidates against different text whenever the temporal rewrite stripped a
    prefix.
    """
    from trw_memory.tools import recall as _recall

    # Retrieve via hybrid search (gracefully degrades to BM25-only or empty)
    if query and all_entries:
        namespace_size = len(all_entries)
        effective_bm25_candidates = max(cfg.bm25_candidates, namespace_size)
        effective_vector_candidates = max(cfg.vector_candidates, namespace_size)
        effective_top_k = limit * cfg.recall_top_k_multiplier
        if tags:
            effective_top_k = max(effective_top_k, namespace_size)

        from trw_memory.retrieval.temporal_query import prepare_temporal_query

        rewrite = prepare_temporal_query(
            query,
            current_recency_weight=cfg.recall_recency_weight,
            auto_temporal=cfg.recall_auto_temporal,
            strip_prefix=cfg.recall_strip_temporal_prefix,
        )
        retrieval_query = rewrite.retrieval_query
        effective_recency_weight = rewrite.recency_weight
        temporal = rewrite.classification
        if temporal is not None and temporal.is_temporal:
            logger.debug(
                "temporal_query_detected",
                query=query[:80],
                retrieval_query=retrieval_query[:80],
                confidence=temporal.confidence,
                recency_weight=effective_recency_weight,
                patterns=temporal.matched_patterns,
                prefix_stripped=rewrite.prefix_stripped,
                surface="memory_recall_tool",
            )
        query_embedding = embed_query(embedder, retrieval_query) if embedder is not None else None
        # dense_search() needs the stored vector map, not just the entry IDs, so
        # tool recall must hydrate the embeddings before calling hybrid_search().
        # Forward the stripped-query embedding so dense search uses the same
        # search text as BM25 and rerank.  Tier scoring below receives the same
        # embedding, keeping the tool path internally consistent.
        scored = _recall.hybrid_search_scored(
            query=retrieval_query,
            entries=all_entries,
            scope=scope,
            embedder=embedder,
            query_embedding=query_embedding,
            stored_embeddings=stored_embeddings or None,
            bm25_candidates=effective_bm25_candidates,
            vector_candidates=effective_vector_candidates,
            rrf_k=cfg.rrf_k,
            importance_alpha=cfg.rrf_importance_alpha,
            top_k=effective_top_k,
            recency_weight=effective_recency_weight,
            recency_halflife_days=cfg.recall_recency_halflife_days,
            fusion_mode=cfg.recall_fusion_mode,
            validity_age_decay=cfg.recall_validity_age_decay,
            rerank=cfg.recall_rerank,
            rerank_model=cfg.recall_rerank_model,
            rerank_candidates=cfg.recall_rerank_candidates,
            # rerank_query omitted intentionally: when temporal boilerplate is
            # stripped, the cross-encoder inherits retrieval_query so it scores
            # against the same topical text as BM25 and dense search.
        )
    else:
        query_embedding = None
        # Empty query: return all entries sorted by utility
        scored = []

    # Convert to dicts for scoring, carrying the retrieval score with each row.
    # PRD-CORE-278 FR03: this is the single point where the pipeline's ranking
    # becomes the caller-visible ``score``; everything downstream preserves it.
    if query and all_entries:
        entry_dicts = []
        for candidate in scored:
            row = candidate.entry.model_dump(mode="json")
            row[FUSED_SCORE_KEY] = candidate.score
            entry_dicts.append(row)
    else:
        # Wildcard recall: there is no retrieval score to carry, so the reported
        # number is the utility the wildcard path has always ordered by. Leaving
        # it absent would hand every row a 0.0 through the source policy below
        # and then let a positive ``min_score`` delete the whole result.
        entry_dicts = []
        for entry in all_entries:
            row = entry.model_dump(mode="json")
            row[FUSED_SCORE_KEY] = round(entry_utility(row, config=cfg), 6)
            entry_dicts.append(row)

    return entry_dicts, query_embedding
