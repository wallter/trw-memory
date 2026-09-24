"""Retrieval stage of the ``memory_recall`` tool path.

Belongs to the ``tools/recall.py`` facade; called from ``memory_recall_impl``.
Extracted by PRD-CORE-278 when carrying the retrieval score through the tool
path pushed ``recall.py`` past the 350 effective-LOC gate.

The contract worth stating: this is where a candidate stops being a stored row
and becomes a RANKED row. It ranks with the policy trw-mcp's ``trw_recall`` uses
(``recall_policy.resolve_query`` and ``ranking_arguments``), calls the scored
retrieval boundary, and writes each candidate's score
onto its serialised dict under ``FUSED_SCORE_KEY`` — the single point where the
pipeline's ranking becomes the caller-visible ``score``. Everything downstream
of it preserves that number rather than recomputing one.

``hybrid_search_scored`` is looked up through the parent module rather than
imported directly, so a test that patches ``trw_memory.tools.recall.
hybrid_search_scored`` still reaches the call it is aiming at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.embeddings._query_prompts import embed_query
from trw_memory.lifecycle._recall import FUSED_SCORE_KEY
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.recall_policy import ranking_arguments, resolve_query

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.security.namespace_scope import NamespaceScope


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

    # One ranking with trw-mcp's trw_recall (PRD-CORE-298 FR05): the resolved
    # query and ``ranking_arguments`` -- rerank, fusion, the adaptive floor and
    # the bridge hop -- so the same pool is ordered the same way on both surfaces.
    if query and all_entries:
        retrieval = resolve_query(query, cfg)
        query_embedding = embed_query(embedder, retrieval.text) if embedder is not None else None
        # dense_search() needs the stored vector map, not just the entry IDs. The
        # resolved-query embedding goes to tier scoring below as well.
        scored = _recall.hybrid_search_scored(
            query=retrieval.text,
            entries=all_entries,
            scope=scope,
            embedder=embedder,
            query_embedding=query_embedding,
            stored_embeddings=stored_embeddings or None,
            **ranking_arguments(
                cfg, limit=limit, pool_size=len(all_entries), recency_weight=retrieval.recency_weight, tags=tags
            ),
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
