"""One candidate acquisition and one resolved ranking policy for every recall surface.

``MemoryClient.recall`` and trw-mcp's ``trw_recall``/``trw_session_start`` rank with
the same ``hybrid_search``; what used to differ was everything around it. The MCP
path acquired only the recency-ordered pool (no FTS leg, so rows past the pool
window were unreachable) and called the ranker without rerank, bridge hop, the
adaptive floor or the fusion settings -- measured 6.2% vs 87.5% hit@10 at 5,000 rows
on EngMem-Synth (PRD-CORE-292). Both surfaces now take these two functions, so
they cannot drift apart again.

Client-free on purpose: the callers own their locks, embedders, vector admission
and namespace authorization; this module owns only what the candidates are and
how they are ranked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.models.memory import MemoryStatus
from trw_memory.security.namespace_scope import NamespaceScopeError

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.retrieval.temporal_selection import TemporalSelection
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


@dataclass
class CandidatePool:
    entries: list[MemoryEntry]
    #: True when the recency scan returned fewer rows than the cap, i.e. saw them all.
    complete: bool
    #: The recency scan's cap, for the caller's telemetry.
    pool_size: int
    #: What the FTS leg did (PRD-FIX-148 FR04), for the caller's telemetry.
    fts_leg: dict[str, object] = field(default_factory=dict)


def acquire_candidates(
    backend: StorageBackend,
    query: str,
    *,
    namespace: str | None,
    limit: int,
    config: MemoryConfig,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    tags: list[str] | None = None,
    exclude_superseded: bool = False,
    temporal_selection: TemporalSelection | None = None,
    acquire: Callable[..., list[MemoryEntry]] | None = None,
) -> CandidatePool:
    """The recency-ordered pool plus the rows full-text search finds beyond it.

    The budget is one rule for every surface: the recency scan reads
    ``max(limit * 5, hybrid_search_candidate_pool_size)`` rows and the FTS leg adds
    up to ``min(pool, max(bm25_candidates * 2, 100))``.

    ``acquire`` is ``RecallInvocation.acquire`` when the caller partitions by policy;
    without it each leg is one direct query. The legs differ on purpose when
    ``status`` is None: the recency scan returns every status (the caller's
    temporal selection and admission decide), while the FTS leg -- which only adds
    rows the recency window missed -- is limited to ACTIVE, as it always was in
    ``MemoryClient.recall``. It never fails the recall: an error is recorded in
    ``fts_leg`` and the recency pool is returned alone.
    """

    def list_leg(predicate: Callable[[MemoryEntry], bool] | None, remaining: int) -> list[MemoryEntry]:
        return backend.list_entries(
            namespace=namespace,
            limit=remaining,
            status=status,
            min_importance=min_importance,
            tags=tags,
            exclude_superseded=exclude_superseded,
            temporal_selection=temporal_selection,
            entry_filter=predicate,
        )

    pool_size = max(limit * 5, config.hybrid_search_candidate_pool_size)
    fts_top_k = min(pool_size, max(config.bm25_candidates * 2, 100))
    entries = acquire(list_leg, limit=pool_size) if acquire is not None else list_leg(None, pool_size)
    complete = len(entries) < pool_size

    fts_leg: dict[str, object] = {"outcome": "not_consulted"}
    if query and not getattr(backend, "_fts_available", False):
        fts_leg = {"outcome": "unavailable"}
    elif query:

        def fts_leg_fetch(predicate: Callable[[MemoryEntry], bool] | None, remaining: int) -> list[MemoryEntry]:
            return backend.search_fts(
                query,
                top_k=remaining,
                namespace=namespace,
                status=status or MemoryStatus.ACTIVE,
                min_importance=min_importance,
                tags=tags,
                temporal_selection=temporal_selection,
                entry_filter=predicate,
            )

        try:
            fts_entries = (
                acquire(fts_leg_fetch, limit=fts_top_k) if acquire is not None else fts_leg_fetch(None, fts_top_k)
            )
            seen = {(e.namespace, e.id) for e in entries}
            new_entries = [e for e in fts_entries if (e.namespace, e.id) not in seen]
            entries = list(entries) + new_entries
            fts_leg = {
                "outcome": "augmented" if new_entries else ("no_new_entries" if fts_entries else "no_rows"),
                "fts_candidates": len(fts_entries),
                "new_entries": len(new_entries),
            }
        except NamespaceScopeError:
            raise
        except Exception:  # justified: augmentation only; recall proceeds on the recency pool
            fts_leg = {"outcome": "failed"}
            logger.debug("fts5_augmentation_failed", exc_info=True)
    logger.debug("fts5_leg", query=query[:80], total_pool=len(entries), **fts_leg)
    return CandidatePool(entries=list(entries), complete=complete, pool_size=pool_size, fts_leg=fts_leg)


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    recency_weight: float
    #: A leading "latest guidance on"-style prefix was removed; a caller-supplied
    #: query vector for the original text no longer matches ``text``.
    prefix_stripped: bool = False


def resolve_query(query: str, config: MemoryConfig) -> RetrievalQuery:
    """The text to retrieve with and its recency weight, auto-temporal applied.

    Explicit config wins: a non-zero ``recall_recency_weight`` is kept; only the
    zero default takes the weight the temporal classifier detects.
    """
    if not config.recall_auto_temporal:
        return RetrievalQuery(query, config.recall_recency_weight)
    from trw_memory.retrieval.temporal_query import prepare_temporal_query

    rewrite = prepare_temporal_query(
        query,
        current_recency_weight=config.recall_recency_weight,
        auto_temporal=True,
        strip_prefix=config.recall_strip_temporal_prefix,
    )
    tc = rewrite.classification
    if tc is not None and tc.is_temporal:
        logger.debug(
            "temporal_query_detected",
            query=query[:80],
            retrieval_query=rewrite.retrieval_query[:80],
            confidence=tc.confidence,
            recency_weight=rewrite.recency_weight,
            patterns=tc.matched_patterns,
            prefix_stripped=rewrite.prefix_stripped,
        )
    return RetrievalQuery(rewrite.retrieval_query, rewrite.recency_weight, rewrite.prefix_stripped)


def hybrid_policy(config: MemoryConfig, *, limit: int, recency_weight: float) -> dict[str, Any]:
    """The ``hybrid_search`` keyword arguments every recall surface ranks with.

    Rerank is unconditional (PRD-CORE-284): only a missing, uncached or -- under
    ``local_only`` -- undownloadable model skips it. The floor scales with ``limit``.
    """
    from trw_memory.retrieval import _adaptive_floor

    floor = _adaptive_floor.adaptive_rerank_floor(limit)
    return {
        "rrf_k": config.rrf_k,
        "importance_alpha": config.rrf_importance_alpha,
        "recency_weight": recency_weight,
        "recency_halflife_days": config.recall_recency_halflife_days,
        "fusion_mode": config.recall_fusion_mode,
        "validity_age_decay": config.recall_validity_age_decay,
        "rerank": True,
        "rerank_model": config.recall_rerank_model,
        "rerank_candidates": config.recall_rerank_candidates,
        "rerank_min_score": floor.min_score,
        "rerank_min_keep": floor.min_keep,
        "rerank_local_only": config.local_only,
        # The entity-bridge second hop only runs when the cross-encoder scored the
        # pool; MEMORY_RECALL_BRIDGE_HOP=false turns it off.
        "bridge_hop": config.recall_bridge_hop,
    }


#: How far past the caller's ``limit`` a tool surface ranks, so rows that source
#: admission, the recall filter or the token budget drop are refilled from ranked
#: rows rather than lost (trw-mcp F-001). Both tool surfaces rank to this depth.
RECALL_PREFETCH_MULTIPLIER = 5


def ranking_arguments(
    config: MemoryConfig, *, limit: int, pool_size: int, recency_weight: float, tags: list[str] | None = None
) -> dict[str, Any]:
    """Every ``hybrid_search`` argument that decides the order, for the tool surfaces.

    *limit* is the ranking depth, ``RECALL_PREFETCH_MULTIPLIER`` times the rows
    the caller will be given; each surface caps to its own limit at the end.

    trw-mcp's ``trw_recall`` and the daemon's ``memory_recall`` both rank with
    this (PRD-CORE-298 FR05), so the same pool yields the same order on both.
    BM25 and dense caps scale to the pool, so the configured values are floors.
    A tag filter applies after ranking, so it asks for the whole pool back.
    ``importance_alpha`` is 1.0 (CORE116 RA2), the value the MCP path has always
    ranked with, overriding ``hybrid_policy``'s configured blend.
    """
    return {
        **hybrid_policy(config, limit=limit, recency_weight=recency_weight),
        "importance_alpha": 1.0,
        "bm25_candidates": max(config.bm25_candidates, pool_size),
        "vector_candidates": max(config.vector_candidates, pool_size),
        "top_k": max(limit, pool_size) if tags else limit,
    }


__all__ = [
    "CandidatePool",
    "RetrievalQuery",
    "acquire_candidates",
    "hybrid_policy",
    "ranking_arguments",
    "resolve_query",
]
