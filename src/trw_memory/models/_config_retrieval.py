"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field

__all__ = ["_RetrievalConfigMixin"]


class _RetrievalConfigMixin:
    # Retrieval
    bm25_candidates: int = Field(default=50, gt=0, description="Number of BM25 candidates to consider")
    vector_candidates: int = Field(default=50, gt=0, description="Number of dense vector candidates to consider")
    rrf_k: int = Field(
        default=5,
        gt=0,
        description=(
            "RRF constant k for reciprocal rank fusion. Default 5 (was 60→15→5) "
            "promoted 2026-06-13 by the memory meta-harness loop: rrf_k=5 gave "
            "+0.8pp recall@5 on LongMemEval-500 over rrf_k=15 (0.9870 vs 0.9790) "
            "after sibling expansion + adaptive temporal window were in place."
        ),
    )
    rrf_importance_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("rrf_importance_alpha", "memory_rrf_importance_alpha"),
        description=(
            "R-FUSION-001: blend weight on the (normalised) RRF position score "
            "vs. the entry's importance in hybrid_search. final = alpha * "
            "rrf_norm + (1 - alpha) * importance. 1.0 = pure position (legacy "
            "behaviour, ignores importance); 0.0 = pure importance. Default 0.7 "
            "lets a high-impact entry edge out an equally-ranked low-impact one "
            "without overriding strong relevance signal."
        ),
    )
    hybrid_search_candidate_pool_size: int = Field(
        default=1000,
        ge=10,
        description=(
            "PRD-DIST-2047 c796: max entries loaded from a namespace for the "
            "hybrid_search candidate pool. Pre-c796 the pool was capped at "
            "limit*5 (=50 for default limit=10), which silently lost targets "
            "ranked past position 50 on namespaces > 50 records. Set higher "
            "for very large namespaces; recall-time cost is O(namespace_size) "
            "for BM25 + O(namespace_size x embedding_dim) for dense search "
            "when the auto-scaled bm25_candidates/vector_candidates lift the "
            "50-cap floor."
        ),
    )
    recall_confidence_filter: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("recall_confidence_filter", "memory_recall_confidence_filter"),
        description=(
            "PRD-DIST-2049 c802: opt-in recall-time confidence floor. When "
            "set, records with metadata['confidence'] < value are suppressed "
            "from MemoryClient.recall() results between merge_tier_results "
            "and apply_source_policy. Default None = filter OFF (current "
            "behavior bit-for-bit). Closes the c800/c801 contamination lever "
            "(2-5pp absolute SC2 lift on full-corpus shapes across "
            "Python/TS/PHP)."
        ),
    )
    recall_filter_historical_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("recall_filter_historical_only", "memory_recall_filter_historical_only"),
        description=(
            "PRD-DIST-2049 c802: opt-in suppression of F2-softened records. "
            "When True, records with metadata['currentness_status'] == "
            "'historical_only' are suppressed from MemoryClient.recall() "
            "results between merge_tier_results and apply_source_policy. "
            "Default False = filter OFF. Mirrors the trw-distill eval-side "
            "_retrieval_policy_filter behaviour into the memory-side recall "
            "path (closes the c800 c763 finding that the F2 label is "
            "necessary but not fully sufficient at recall time)."
        ),
    )
    recall_top_k_multiplier: int = Field(
        default=3,
        ge=1,
        le=50,
        validation_alias=AliasChoices("recall_top_k_multiplier", "memory_recall_top_k_multiplier"),
        description=(
            "PRD-DIST-2050 c804: depth multiplier for the hybrid_search "
            "candidate pool returned by _try_hybrid_recall. Effective top_k "
            "= limit * recall_top_k_multiplier (default 3 → top-30 for "
            "limit=10, matches pre-c804 hardcoded behaviour). Raise to 10 "
            "or higher when the recall-time admission filter (PRD-DIST-2049) "
            "is enabled on pure-zombie corpora — c803 found those filters "
            "can suppress but not promote baseline records ranked past the "
            "current top-30 candidate pool depth. Capped at 50 to keep "
            "downstream per-result cost bounded."
        ),
    )
    recall_preserve_hybrid_order: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_preserve_hybrid_order", "memory_recall_preserve_hybrid_order"),
        description=(
            "PRD-DIST-2051 c806 / PRD-DIST-2058 c817: when True, "
            "merge_tier_results returns "
            "local_results[:limit] (preserving the BM25+dense+RRF ordering "
            "from _try_hybrid_recall) whenever len(local_results) >= limit. "
            "Skips the compute_importance_score rescore that mixes hybrid "
            "RRF (1/(1+rank)) and tier-only entry_utility (absolute) scales. "
            "c805 trace showed all 4 missing hono baselines were at "
            "hybrid_rank=2 but got pushed past top-10 by the rescore; "
            "c811-c815 showed default-ON is robust across curated-query "
            "oracles, languages, and K-depths. Set "
            "MEMORY_RECALL_PRESERVE_HYBRID_ORDER=false to opt out."
        ),
    )

    # Recency ranking — blend valid_from-based exponential decay into relevance
    recall_recency_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("recall_recency_weight", "memory_recall_recency_weight"),
        description=(
            "When > 0, blend valid_from recency decay into the BM25+dense fused "
            "relevance score. Targets the temporal discrimination band "
            "(recall 0.853). Recommended starting point: 0.3. Default 0.0 = "
            "disabled (pure text-relevance behaviour)."
        ),
    )
    recall_recency_halflife_days: float = Field(
        default=14.0,
        gt=0.0,
        validation_alias=AliasChoices("recall_recency_halflife_days", "memory_recall_recency_halflife_days"),
        description=(
            "Half-life in days for the recency decay function. An entry this many "
            "days old receives score 0.5 relative to a brand-new entry. Default 14 "
            "days; reduce for short-lived session corpora, increase for long-lived "
            "institutional knowledge. Ignored when recall_recency_weight == 0."
        ),
    )
    # Fusion algorithm — expose combmax as an alternative to default RRF
    recall_fusion_mode: str = Field(
        default="rrf",
        validation_alias=AliasChoices("recall_fusion_mode", "memory_recall_fusion_mode"),
        description=(
            "Fusion algorithm for hybrid_search. 'rrf' (default) = Reciprocal Rank "
            "Fusion (sum of reciprocal ranks). 'combmax' = CombMAX (max reciprocal "
            "rank per document), which lifts hard-tail recall@12 by ~28% "
            "(McNemar p=0.0074) at the cost of weaker cross-list boosting. "
            "Set MEMORY_RECALL_FUSION_MODE=combmax to enable."
        ),
    )
    # Validity age decay — break ties by valid_from recency in the eligibility pass
    recall_validity_age_decay: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_validity_age_decay", "memory_recall_validity_age_decay"),
        description=(
            "When True, apply tie-only valid_from recency inside the validity prior "
            "pass so a newer record floats above an older one only when their fused "
            "scores are equal. Fusion order is otherwise preserved. Default True."
        ),
    )
    # Cross-encoder re-ranking (optional; requires sentence-transformers)
    recall_rerank: bool = Field(
        default=False,
        validation_alias=AliasChoices("recall_rerank", "memory_recall_rerank"),
        description=(
            "When True, apply cross-encoder re-ranking after RRF fusion using "
            "recall_rerank_model. Requires sentence-transformers and a cached model. "
            "Silently falls back to fusion order when unavailable. Latency: ~20-80ms "
            "on CPU for 50 candidates. Default False = disabled."
        ),
    )
    recall_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("recall_rerank_model", "memory_recall_rerank_model"),
        description=(
            "HuggingFace model id for cross-encoder re-ranking. Default is the "
            "66M-param ms-marco passage re-ranker. Ignored when recall_rerank=False."
        ),
    )
    recall_rerank_candidates: int = Field(
        default=50,
        gt=0,
        validation_alias=AliasChoices("recall_rerank_candidates", "memory_recall_rerank_candidates"),
        description=(
            "Number of top-fusion candidates to pass to the cross-encoder. "
            "Limiting to top-50 captures the quality gain at reasonable latency. "
            "Ignored when recall_rerank=False."
        ),
    )
    recall_auto_temporal: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_auto_temporal", "memory_recall_auto_temporal"),
        description=(
            "When True (default), queries containing temporal language (e.g. "
            "'recent', 'last week', 'latest') automatically receive a "
            "recency_weight derived from the classifier confidence. Only "
            "activates when recall_recency_weight=0.0 (explicit config wins). "
            "Disable to enforce position-only RRF for all queries."
        ),
    )
    recall_strip_temporal_prefix: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_strip_temporal_prefix", "memory_recall_strip_temporal_prefix"),
        description=(
            "When True (default) and the query is classified as temporal, "
            "strip common boilerplate prefixes ('latest guidance on X' → 'X') "
            "before running BM25, dense retrieval, and optional cross-encoder "
            "reranking. Set False to disable prefix stripping and pass the raw "
            "query to all retrieval stages."
        ),
    )
