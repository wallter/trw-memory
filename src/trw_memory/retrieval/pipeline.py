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

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import dense_search
from trw_memory.retrieval.fusion import blend_recency, combmax_fuse, rrf_fuse
from trw_memory.retrieval.lexical import lexical_relevance, tokenize_query
from trw_memory.retrieval.recency import recency_rank
from trw_memory.retrieval.validity_prior import apply_validity_prior
from trw_memory.security.namespace_scope import NamespaceScope, NamespaceScopeError

logger = structlog.get_logger(__name__)
_RETIRED_UNSET = object()


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One ranked candidate and the number that put it where it is.

    PRD-CORE-278 FR01. ``basis`` is part of the contract because the fused score
    is NOT always the authority for the final order: ``apply_validity_prior``
    positionally appends superseded records and ``cross_encode_rerank`` re-scores
    the head of the list. Returning the original fusion numbers after either ran
    would produce a score sequence that contradicts the order it claims to
    explain.

    - ``basis="fused"`` — no post-fusion pass reordered the list; ``score`` is
      the blended fusion score.
    - ``basis="position"`` — a post-fusion pass ran; ``score`` is ``1 / (1 +
      rank)`` over the FINAL order, for every candidate in the call.

    The basis is uniform per call, never per candidate, so a consumer never
    compares two scales inside one result.
    """

    entry: MemoryEntry
    score: float
    basis: str


def hybrid_search(
    query: str,
    entries: list[MemoryEntry],
    *,
    score_observer: Callable[[tuple[ScoredCandidate, ...]], None] | None = None,
    **kwargs: object,
) -> list[MemoryEntry]:
    """Entry-only view of :func:`hybrid_search_scored` (unchanged contract).

    Kept as the package's stable surface: callers that only need the ranking
    order are unaffected by PRD-CORE-278's scored boundary. A caller that also
    needs the scores (trw-mcp's recall, PRD-CORE-292) passes ``score_observer``
    and receives the scored candidates exactly as ranked, before the entry view.
    """
    scored = hybrid_search_scored(query, entries, **kwargs)  # type: ignore[arg-type]
    if score_observer is not None:
        score_observer(tuple(scored))
    return [candidate.entry for candidate in scored]


def hybrid_search_scored(
    query: str,
    entries: list[MemoryEntry],
    *,
    scope: NamespaceScope,
    embedder: EmbeddingProvider | None = None,
    query_embedding: list[float] | None = None,
    stored_embeddings: dict[str, list[float]] | None = None,
    bm25_candidates: int = 50,
    vector_candidates: int = 50,
    # rrf_k=5 (was 60→15→5): promoted 2026-06-13 by the memory meta-harness
    # loop after sibling expansion + adaptive temporal windows were in place.
    # MemoryConfig is the runtime source of truth; keep this direct helper
    # default aligned so tests and ad-hoc callers do not silently grade a
    # different retrieval policy than MemoryClient.recall().
    rrf_k: int = 5,
    importance_alpha: float = 1.0,
    top_k: int = 25,
    fusion_mode: str = "rrf",
    as_of: datetime | None = None,
    valid_from_min: datetime | None = None,
    include_superseded: bool = False,
    validity_age_decay: bool = False,
    validity_reference_time: datetime | None = None,
    recency_weight: float = 0.0,
    recency_halflife_days: float = 14.0,
    recency_now: datetime | None = None,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_candidates: int = 50,
    rerank_query: str | None = None,
    rerank_min_score: float | None = None,
    rerank_min_keep: int = 5,
    rerank_local_only: bool = False,
    collapse_hype: object = _RETIRED_UNSET,
    dense_observer: Callable[[tuple[tuple[str, float], ...]], None] | None = None,
    bridge_hop: bool = False,
) -> list[ScoredCandidate]:
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
        scope: The namespaces this call is cleared to rank, minted by
            :func:`~trw_memory.security.namespace_scope.authorize_namespaces`.
            Required with no default (PRD-CORE-245 FR04): a default would put
            isolation back where it was, in each caller's discipline. Every
            candidate must belong to it, and a candidate that does not raises
            :class:`~trw_memory.security.namespace_scope.NamespaceScopeError`
            BEFORE any retrieval step runs, rather than being quietly dropped.
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
        dense_observer: Optional request-owned callback receiving an immutable
            snapshot of finite raw cosine scores, before candidate caps, fusion,
            importance, recency, or temporal filtering. HyPE hits are collapsed
            to authorized parent IDs first. Called once when the dense path runs
            (possibly with an empty tuple), never for an unauthorized candidate
            scope. Does not persist metadata or identify a request/source: the
            caller must bind that provenance and intersect observations with its
            eligible results. Callback exceptions propagate; an observer must not
            publish partial state before completing successfully. Opting in retains
            all scores already computed by dense_search, without another model
            call, and leaves the legacy capped fusion input unchanged.
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
            brand-new entry.  Default ``14.0`` days, matching
            ``MemoryConfig.recall_recency_halflife_days``.  Ignored when
            ``recency_weight == 0``.
        valid_from_min: When set, only include entries whose ``valid_from`` is
            at or after this datetime.  Useful for narrowing results to a
            specific date range — e.g. when temporal arithmetic resolves
            "10 days ago" to a target date, pass
            ``valid_from_min = target - slack`` to exclude sessions from before
            the approximate target period.  Applied after fusion as an AND
            filter alongside *as_of*.
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
        bridge_hop: When ``True`` and the cross-encoder scored the pool, run
            the LLM-free entity-bridge second hop
            (:func:`~trw_memory.retrieval.bridge.extend_with_bridge`): rare
            terms of the top re-ranked rows retrieve further tail candidates,
            which the cross-encoder scores against the same query. Default
            ``False``; ``MemoryClient.recall`` turns it on with re-ranking.

    Returns:
        Up to *top_k* :class:`ScoredCandidate` objects ordered by descending
        score. The score is the fused relevance score, or a position-derived
        score when a post-fusion pass reordered the list; ``basis`` says which,
        uniformly for the whole call.
    """
    if collapse_hype is not _RETIRED_UNSET:
        if collapse_hype is not False:
            raise TypeError("collapse_hype: HyPE is retired; remove this argument")
        warnings.warn("collapse_hype: HyPE is retired; remove this argument", UserWarning, stacklevel=2)
    if not entries:
        return []

    # PRD-CORE-245 FR04: assert containment BEFORE any retrieval step. It
    # asserts rather than filters -- a caller that assembled a list spanning
    # namespaces it was not cleared for has a bug, and truncating the list here
    # would hide it. An empty scope therefore admits nothing, which is the
    # fail-closed behaviour NFR03 asks for.
    outside = {entry.namespace for entry in entries} - scope.namespaces
    if outside:
        raise NamespaceScopeError(
            f"hybrid_search received {len(outside)} namespace(s) outside the authorized scope "
            f"(scope holds {len(scope.namespaces)}); the caller assembled candidates it was not cleared to rank"
        )

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
            top_k=len(entry_ids) if dense_observer is not None else vector_candidates,
        )
        if dense_observer is not None:
            # Keep the canonical candidate cap for fusion. Observations
            # carry evidence, not an alternative ranking or persisted metadata.
            observations = [(eid, score) for eid, score in dense_results if math.isfinite(score)]
            dense_observer(tuple(observations))
            dense_results = dense_results[:vector_candidates]
        if dense_results:
            rankings.append(dense_results)

    # ------------------------------------------------- Lexical fallback source
    # PRD-CORE-278 FR04: when BOTH sources produced nothing — no BM25 match and
    # no stored vectors — rank whole-word lexical overlap rather than returning
    # an empty list. It is added HERE, as a third ranking source before fusion,
    # so it inherits the namespace-scope assertion above, the validity prior
    # below and ``top_k``. Doing it in the caller would resurrect superseded and
    # ``as_of``-excluded records, because those exclusions live in the prior.
    lexical_fallback: list[tuple[str, float]] = []
    if not rankings and query.strip():
        query_tokens = tokenize_query(query)
        lexical_fallback = sorted(
            (
                (entry.id, relevance)
                for entry in entries
                if (relevance := lexical_relevance(entry.model_dump(mode="json"), query_tokens)) > 0.0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if lexical_fallback:
            rankings.append(lexical_fallback)

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
    # blend_recency() (fusion.py) is the single implementation; this call is the
    # pipeline integration point that feeds it the right inputs.
    if recency_weight > 0.0:
        recency_results = recency_rank(
            entries,
            halflife_days=recency_halflife_days,
            now=recency_now,
        )
    fused = blend_recency(relevance_fused, recency_results=recency_results, recency_weight=recency_weight)

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
    pre_prior_order = [entry.id for entry in fused_entries]
    fused_entries = apply_validity_prior(
        fused_entries,
        as_of=as_of,
        valid_from_min=valid_from_min,
        include_superseded=include_superseded,
        age_decay=validity_age_decay,
        reference_time=validity_reference_time,
        fusion_scores=fused_scores,
    )
    # The prior may drop records (order preserved) or REORDER them (superseded
    # rows appended, age decay applied). Only a reorder invalidates the fused
    # numbers as an explanation of the final order.
    survivors = {entry.id for entry in fused_entries}
    prior_reordered = [entry.id for entry in fused_entries] != [
        entry_id for entry_id in pre_prior_order if entry_id in survivors
    ]

    # --------------------------------------------------------- Re-ranking
    # Optional cross-encoder re-ranking: score top-N candidates jointly as
    # (query, passage) pairs.  This captures finer-grained relevance than
    # bi-encoder + RRF at the cost of O(rerank_candidates) model calls.
    reranked = False
    if rerank and fused_entries:
        from trw_memory.retrieval.reranker import cross_encode_scores

        # Use rerank_query (original un-preprocessed query) when supplied so
        # the cross-encoder receives the full user intent even if the search
        # query was stripped of temporal boilerplate.
        effective_rerank_query = rerank_query if rerank_query is not None else query
        rerank_input = fused_entries[:rerank_candidates]
        tail = fused_entries[rerank_candidates:]
        pre_rerank_order = [entry.id for entry in rerank_input]
        scored = cross_encode_scores(
            effective_rerank_query, rerank_input, model_name=rerank_model, local_only=rerank_local_only
        )
        # Entity-bridge second hop: salient terms of the best first-hop rows
        # pull further candidates out of the un-reranked tail, scored against
        # the same query so the cross-encoder still decides where they land.
        bridged = False
        if scored and bridge_hop:
            from trw_memory.retrieval.bridge import extend_with_bridge

            scored, tail, bridged = extend_with_bridge(
                query,
                scored,
                tail,
                entries,
                score=lambda fresh: cross_encode_scores(
                    effective_rerank_query, fresh, model_name=rerank_model, local_only=rerank_local_only
                ),
            )
        if scored is None:
            pass  # cross-encoder unavailable: keep fusion order and every candidate
        elif rerank_min_score is None:
            fused_entries = [*(e for e, _ in scored), *tail]
        else:
            # Confidence-bounded recall: return only what the cross-encoder finds
            # plausibly relevant, never fewer than rerank_min_keep, and nothing
            # from the un-scored tail (it ranked below everything scored). On
            # LOCOMO a -8 logit cutoff shrank the list from 50 to 34 entries
            # while keeping 99% of the evidence the uncut top-50 held.
            keep = max(0, rerank_min_keep)
            fused_entries = [e for i, (e, s) in enumerate(scored) if i < keep or s >= rerank_min_score]
        # PRD-CORE-278: a reorder OR a cut means the fused numbers no longer
        # explain the returned list, so the scores below switch to position.
        reranked = scored is not None and (
            bridged
            or [entry.id for entry in fused_entries[: len(pre_rerank_order)]] != pre_rerank_order
            or len(fused_entries) != len(pre_rerank_order) + len(tail)
        )

    ranked_entries: list[MemoryEntry] = fused_entries[:top_k]
    # PRD-CORE-278 FR01: report the score that explains the order actually
    # returned, and say which basis it is on.
    if prior_reordered or reranked:
        results = [
            # Not rounded: at a deep top_k, rounding collapses adjacent
            # positions onto one value and hands the ordering back to utility.
            ScoredCandidate(entry=entry, score=1.0 / (1 + rank), basis="position")
            for rank, entry in enumerate(ranked_entries)
        ]
    else:
        results = [
            ScoredCandidate(entry=entry, score=float(fused_scores.get(entry.id, 0.0)), basis="fused")
            for entry in ranked_entries
        ]

    logger.debug(
        "hybrid_search_complete",
        query=query[:80],
        entry_count=len(entries),
        bm25_hits=len(bm25_results) if bm25_results else 0,
        lexical_fallback_hits=len(lexical_fallback),
        recency_hits=len(recency_results),
        fused_total=len(fused),
        returned=len(results),
        fusion_mode=fusion_mode,
        recency_weight=recency_weight,
        rerank=rerank,
        score_basis=results[0].basis if results else "",
    )
    return results
