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

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING

import structlog

from trw_memory._client_distilled_tiering import entry_to_result as _entry_to_result
from trw_memory.embeddings._query_prompts import embed_query
from trw_memory.embeddings._space_gate import active_embedding_space, admit_space_vectors
from trw_memory.retrieval.recall_policy import acquire_candidates, hybrid_policy, resolve_query
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation, entry_policy_fields
from trw_memory.security.namespace_scope import NamespaceScopeError, authorize_namespaces
from trw_memory.security.rbac import Permission

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.retrieval.pipeline import ScoredCandidate

logger = structlog.get_logger(__name__)


@dataclass
class HybridPool:
    """Out-parameter: the candidate pool ``try_hybrid_recall`` ranked.

    ``entry_ids`` is every entry the pool held (namespace scan plus FTS
    augmentation). ``complete`` is True when the namespace scan returned fewer
    rows than the pool cap -- the cap did not bind, so every row the scan admits
    was ranked. Left at its defaults when the pipeline is unavailable.
    """

    entry_ids: frozenset[str] = frozenset()
    complete: bool = False

    def covered_ids(self) -> frozenset[str]:
        """Rows a complete pool already ranked; empty when the cap may have cut some."""
        return self.entry_ids if self.complete else frozenset()


async def try_hybrid_recall(
    client: MemoryClient,
    query: str,
    limit: int,
    tags: list[str] | None,
    query_embedding: list[float] | None = None,
    *,
    as_of: datetime | None = None,
    include_superseded: bool = False,
    invocation: RecallInvocation | None = None,
    pool: HybridPool | None = None,
) -> list[MemoryResultDict] | list[LocalCandidate] | None:
    """Hybrid pipeline (BM25 + dense + RRF). Returns None to signal fallback.

    When *pool* is given it is filled with the loaded candidate pool (see
    :class:`HybridPool`) so the caller can skip re-discovering those rows.

    PRD-DIST-2047 Phase 2 (recall-latency telemetry): emits a structlog event
    ``hybrid_recall_complete`` carrying per-call timings + namespace shape +
    effective candidate caps + returned-result count, so operators can right-
    size ``hybrid_search_candidate_pool_size`` against measured cost. The
    event fires on every terminating exit (success, no-candidates, hybrid-
    search-failed) so operators can attribute latency to outcome.

    *query_embedding* is the query vector the caller already computed (for tier
    scoring); when supplied it is forwarded to ``hybrid_search`` so the dense
    step reuses it instead of re-embedding the query. If temporal prefix
    stripping rewrites the retrieval query, the embedding is recomputed for the
    stripped query so BM25, dense, and rerank all share the same search text.
    ``None`` preserves the legacy behaviour of embedding inside the dense step.
    """
    try:
        from trw_memory.retrieval.pipeline import hybrid_search
    except ImportError:  # trw-fail-silent-allow: optional retrieval extras absent, caller falls back to keyword recall
        return None

    total_start = perf_counter()

    async with client._lock:
        backend = client._get_backend()
        list_entries_start = perf_counter()
        # One acquisition for every recall surface (PRD-CORE-292, PRD-CORE-298 FR05):
        # the recency pool, superseded rows excluded at the SQL level unless
        # time-travelling, plus the rows full-text search finds past the pool
        # window (PRD-FIX-148).
        candidates = acquire_candidates(
            backend,
            query,
            namespace=client._namespace,
            limit=limit,
            config=client._config,
            exclude_superseded=not include_superseded and as_of is None,
            temporal_selection=invocation.temporal if invocation is not None else None,
            acquire=invocation.acquire if invocation is not None else None,
        )
        candidate_pool_size = candidates.pool_size
        all_entries = candidates.entries
        list_entries_ms = (perf_counter() - list_entries_start) * 1000.0
        scan_complete = candidates.complete

        # Vectors are read with their provenance; which of them may be dense-
        # scored is decided below, once the active embedder's space is known.
        vector_records = backend.get_vector_records([entry.id for entry in all_entries], namespace=client._namespace)

    if pool is not None:
        pool.entry_ids = frozenset(entry.id for entry in all_entries)
        pool.complete = scan_complete

    namespace_size = len(all_entries)
    effective_bm25_candidates = effective_vector_candidates = 0
    # PRD-DIST-2050 c804: deepen the candidate pool when the admission filter
    # is opt-in enabled, so baseline records ranked past top-30 can survive the
    # filter and enter the merged top-K. Default multiplier=3 preserves pre-c804
    # behaviour (top-30); operators raise via MEMORY_RECALL_TOP_K_MULTIPLIER.
    effective_top_k = limit * client._config.recall_top_k_multiplier

    def emit_telemetry(outcome: str, returned_count: int = 0, hybrid_search_ms: float = 0.0) -> None:
        """PRD-DIST-2047 Phase 2: one per-recall latency + shape event, with the candidate caps current at the call.

        Operators sample this event stream to right-size
        ``hybrid_search_candidate_pool_size`` for very large namespaces (where
        BM25 cost grows linearly with namespace_size). Latencies are reported in
        milliseconds rounded to 3 decimals.
        """
        logger.info(
            "hybrid_recall_complete",
            op="recall",
            outcome=outcome,
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=effective_bm25_candidates,
            effective_vector_candidates=effective_vector_candidates,
            effective_top_k=effective_top_k,
            returned_count=returned_count,
            list_entries_ms=round(list_entries_ms, 3),
            hybrid_search_ms=round(hybrid_search_ms, 3),
            total_ms=round((perf_counter() - total_start) * 1000.0, 3),
        )

    if not all_entries:
        emit_telemetry("no_candidates")
        return None

    embedder = client._get_embedder()
    # Only vectors from the active embedder's space are dense-scored; the rest
    # stay in the pool for BM25 (embeddings/_space_gate.py).
    stored_embeddings = (
        admit_space_vectors(
            vector_records, active_embedding_space(embedder), namespace=client._namespace, surface="hybrid_recall"
        )
        if embedder is not None and vector_records
        else {}
    )
    # PRD-DIST-2047 c796: auto-scale bm25/vector candidate caps to namespace
    # size so the 50-default acts as a FLOOR, not a CEILING. Eliminates the
    # structural cap on recall@10 for namespaces > 50 records.
    effective_bm25_candidates = max(client._config.bm25_candidates, namespace_size)
    effective_vector_candidates = max(client._config.vector_candidates, namespace_size)
    # When a tag filter is requested it is applied AFTER hybrid_search ranks and
    # truncates to top_k (below). Tag-matching entries ranked past top_k would be
    # silently dropped, reducing recall below the caller-requested limit. Rank the
    # FULL candidate pool when tags are present so the post-rank tag filter sees
    # every entry the namespace scan loaded — the tag filter then narrows back
    # down. namespace_size (== len(all_entries)) is already bounded by
    # candidate_pool_size, so this cannot widen cost beyond the entries we already
    # hold in memory.
    if tags or invocation is not None:
        effective_top_k = max(effective_top_k, namespace_size)

    # Auto-detect temporal queries and inject recency_weight + strip boilerplate
    # prefixes when the config hasn't explicitly enabled either.  Preserves
    # explicit config — if the operator set recall_recency_weight > 0 we use
    # that value; only the zero-default case gets the auto-detected weight.
    resolved = resolve_query(query, client._config)
    retrieval_query, effective_recency_weight = resolved.text, resolved.recency_weight

    effective_query_embedding = query_embedding
    if resolved.prefix_stripped and embedder is not None:
        # The caller's precomputed vector represents the original query
        # ("latest guidance on X"). Once the retrieval query is stripped to
        # "X", dense search must use the stripped vector too; otherwise prefix
        # stripping only affects BM25 while dense similarity keeps the semantic
        # drift the rewrite was meant to remove.
        try:
            effective_query_embedding = await asyncio.to_thread(embed_query, embedder, retrieval_query)
        except (RuntimeError, ValueError, TypeError):
            logger.warning(
                "hybrid_recall_temporal_embedding_failed",
                query_chars=len(retrieval_query),
                exc_info=True,
            )
            effective_query_embedding = None

    # PRD-CORE-245 FR04: the client ranks its own namespace and nothing else,
    # so the scope is minted for exactly that one -- through the authorizer, not
    # by hand, so the RBAC check runs on this surface too.
    scope = authorize_namespaces(client._config, [client._namespace], Permission.READ, "recall")
    hybrid_search_start = perf_counter()
    observed: list[tuple[ScoredCandidate, ...]] = []
    try:
        ranked = hybrid_search(
            query=retrieval_query,
            entries=all_entries,
            scope=scope,
            embedder=embedder,
            query_embedding=effective_query_embedding,
            stored_embeddings=stored_embeddings or None,
            bm25_candidates=effective_bm25_candidates,
            vector_candidates=effective_vector_candidates,
            top_k=effective_top_k,
            as_of=as_of,
            include_superseded=include_superseded,
            validity_reference_time=invocation.temporal.reference_time if invocation else None,
            # PRD-CORE-284/292: the one resolved policy every recall surface ranks with.
            **hybrid_policy(client._config, limit=limit, recency_weight=effective_recency_weight),
            # PRD-CORE-336 FR01: the distilled weight enters the order here, once.
            distilled_weight=invocation.source.weights.get("git_distilled", 1.0) if invocation else None,
            score_observer=observed.append,
            # When prefix was stripped, the cross-encoder also uses the stripped
            # query — the original "latest guidance on X" confuses the ms-marco
            # reranker (entries lack "guidance" vocabulary): -4.5pp T-HR.
            # rerank_query=None → the cross-encoder inherits retrieval_query.
        )
    except NamespaceScopeError:
        raise
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
        emit_telemetry("hybrid_search_failed", hybrid_search_ms=hybrid_search_ms)
        return None
    hybrid_search_ms = (perf_counter() - hybrid_search_start) * 1000.0

    if not ranked:
        emit_telemetry("empty_ranking", hybrid_search_ms=hybrid_search_ms)
        return None

    if tags:
        tag_set = set(tags)
        ranked = [e for e in ranked if tag_set.issubset(set(e.tags))]

    scored = {candidate.entry.id: candidate for candidate in observed[0]} if observed else {}
    policy = invocation.source if invocation is not None else None
    scores = positional_scores(
        [scored.get(entry.id) for entry in ranked],
        None
        if policy is None
        else lambda e, score: policy.rank_key(entry_policy_fields(e, score=score), pipeline_weighted=True),
    )
    emit_telemetry("ok", len(ranked), hybrid_search_ms)
    if invocation is not None:
        return [
            LocalCandidate(entry, score, relevance_hint=score, distilled_weighted=True)
            for entry, score in zip(ranked, scores, strict=True)
        ]
    return [_entry_to_result(entry, score=score) for entry, score in zip(ranked, scores, strict=True)]


def positional_scores(
    ranked: list[ScoredCandidate | None], sort_key: Callable[[MemoryEntry, float], tuple[int, float]] | None = None
) -> list[float]:
    """The library's positional score per row of the pipeline's final order (PRD-CORE-336 NFR01).

    Every non-distilled row scores ``round(1/(1+i), 4)`` at its index ``i`` in
    the PRE-weight order -- its lead/int-700 score -- so ``SourcePolicy.rank_key``
    multiplies the same gaps it always did and a distilled row moving past two
    rows of different families cannot swap them. Re-scoring positions after the
    distilled re-sort did exactly that, because 1/(1+i) is nonlinear.

    A distilled row the pipeline weighted is placed in *sort_key*'s space -- the
    ``(bucket, -score x family weight)`` key ``finish_candidates`` sorts by, where
    such a row's own weight is 1.0 -- not in raw positions: its own pre-weight
    position, raised to the best key of any non-distilled row the pipeline ranked
    BEHIND it, then capped at the worst key of every row ranked AHEAD of it in its
    bucket. A tie keeps the pipeline order (the sort is stable), so a distilled
    row is never promoted above a row the pipeline ranked ahead of it. If a lower
    row's family weight already lifts it above a higher one (the pre-existing
    cross-family reorder), the cap wins: demotion is never undone.

    With no weighted row (no distilled rows, weight 1.0, or no scores observed)
    this is ``round(1/(1+i), 4)`` over *ranked*, unchanged.
    """
    order = sorted(
        range(len(ranked)),
        key=lambda i: row.preweight_rank if (row := ranked[i]) and row.preweight_rank is not None else i,
    )
    own = [0.0] * len(ranked)
    for position, index in enumerate(order):
        own[index] = round(1.0 / (1 + position), 4)
    rows = [row for row in ranked if row is not None]
    if len(rows) != len(ranked) or not any(row.distilled_weighted for row in rows):
        return own
    key = sort_key or (lambda _entry, score: (0, -score))
    buckets = [key(row.entry, own[index])[0] for index, row in enumerate(rows)]
    floor = [0.0] * len(rows)
    best_below: dict[int, float] = {}
    for index in reversed(range(len(rows))):
        floor[index] = best_below.get(buckets[index], 0.0)
        if not rows[index].distilled_weighted:
            below = -key(rows[index].entry, own[index])[1]
            best_below[buckets[index]] = max(below, best_below.get(buckets[index], 0.0))
    ceiling: dict[int, float] = {}
    scores: list[float] = []
    for index, row in enumerate(rows):
        score = own[index]
        if row.distilled_weighted:
            score = min(ceiling.get(buckets[index], math.inf), max(floor[index], score))
        effective = -key(row.entry, score)[1]
        ceiling[buckets[index]] = min(effective, ceiling.get(buckets[index], math.inf))
        scores.append(score)
    return scores
