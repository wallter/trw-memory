"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field

__all__ = ["_RetrievalConfigMixin"]


class _RetrievalConfigMixin(BaseModel):
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
            "Max entries loaded from a namespace for the hybrid_search "
            "candidate pool. An earlier build capped the pool at limit*5 "
            "(=50 for default limit=10), which silently lost targets ranked "
            "past position 50 on namespaces > 50 records. Set higher for very "
            "large namespaces; recall-time cost is O(namespace_size) for BM25 "
            "+ O(namespace_size x embedding_dim) for dense search when the "
            "auto-scaled bm25_candidates/vector_candidates lift the 50-cap "
            "floor."
        ),
    )
    recall_confidence_filter: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("recall_confidence_filter", "memory_recall_confidence_filter"),
        description=(
            "Opt-in recall-time confidence floor. When set, records with "
            "metadata['confidence'] < value are suppressed from "
            "MemoryClient.recall() results between merge_tier_results and "
            "apply_source_policy. Default None = filter OFF (current behavior "
            "bit-for-bit). Closes a low-confidence contamination lever "
            "(a few percentage points of retrieval-accuracy lift on "
            "full-corpus shapes across multiple languages in offline "
            "evaluation)."
        ),
    )
    recall_filter_historical_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("recall_filter_historical_only", "memory_recall_filter_historical_only"),
        description=(
            "Opt-in suppression of records softened to a historical-only "
            "currentness status. When True, records with "
            "metadata['currentness_status'] == 'historical_only' are "
            "suppressed from MemoryClient.recall() results between "
            "merge_tier_results and apply_source_policy. Default False = "
            "filter OFF. Mirrors the eval-side retrieval-policy filter into "
            "the memory-side recall path (offline evaluation found the "
            "currentness label necessary but not fully sufficient at recall "
            "time)."
        ),
    )
    recall_top_k_multiplier: int = Field(
        default=3,
        ge=1,
        le=50,
        validation_alias=AliasChoices("recall_top_k_multiplier", "memory_recall_top_k_multiplier"),
        description=(
            "Depth multiplier for the hybrid_search candidate pool returned "
            "by _try_hybrid_recall. Effective top_k = limit * "
            "recall_top_k_multiplier (default 3 → top-30 for limit=10, "
            "matches the prior hardcoded behaviour). Raise to 10 or higher "
            "when a recall-time admission filter (recall_confidence_filter / "
            "recall_filter_historical_only) is enabled on corpora dominated "
            "by stale records — offline evaluation found those filters can "
            "suppress but not promote baseline records ranked past the "
            "current top-30 candidate pool depth. Capped at 50 to keep "
            "downstream per-result cost bounded."
        ),
    )
    recall_preserve_hybrid_order: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_preserve_hybrid_order", "memory_recall_preserve_hybrid_order"),
        description=(
            "When True, merge_tier_results returns local_results[:limit] "
            "(preserving the BM25+dense+RRF ordering from _try_hybrid_recall) "
            "whenever len(local_results) >= limit. Skips the "
            "compute_importance_score rescore that mixes hybrid RRF "
            "(1/(1+rank)) and tier-only entry_utility (absolute) scales. An "
            "offline trace showed missing baseline records sat at "
            "hybrid_rank=2 but got pushed past top-10 by the rescore; "
            "follow-up sweeps showed default-ON is robust across "
            "curated-query oracles, languages, and K-depths. Set "
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

    # HyPE — index-time hypothetical-question expansion (PRD-CORE-195).
    hype_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("hype_enabled", "memory_hype_enabled"),
        description=(
            "PRD-CORE-195: when True, generate hypothetical questions at WRITE "
            "time (via an injected QuestionGenerator), embed them, and store "
            "them as secondary '{parent_id}#hype{n}' retrieval vectors that fuse "
            "back to the parent entry at recall. Default False = pre-HyPE "
            "behaviour bit-for-bit (no siblings written, no collapse pass)."
        ),
    )
    hype_questions_per_entry: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias=AliasChoices("hype_questions_per_entry", "memory_hype_questions_per_entry"),
        description=(
            "PRD-CORE-195: maximum number of hypothetical-question sibling "
            "vectors stored per entry when HyPE is enabled. The frontier-refresh "
            "note recommends 3-5; default 3. Caps the per-store embedding cost."
        ),
    )
    hype_min_question_chars: int = Field(
        default=8,
        ge=1,
        validation_alias=AliasChoices("hype_min_question_chars", "memory_hype_min_question_chars"),
        description=(
            "PRD-CORE-195: minimum character length for a generated question to "
            "be embedded + stored as a HyPE sibling. Shorter questions are "
            "skipped (likely degenerate generator output). Default 8."
        ),
    )
    cold_search_cache_max: int = Field(
        default=1000,
        gt=0,
        description=(
            "Maximum number of cold-tier YAML files held in the in-memory search "
            "cache (trw-memory-15). The cache is a bounded LRU: once it exceeds "
            "this many entries the least-recently-used file is evicted, capping "
            "RAM growth on long-lived processes with large cold archives. Each "
            "cached entry holds the deserialized YAML + search text for one file."
        ),
    )
    graph_tag_min_shared_tags: int = Field(
        default=2,
        ge=1,
        le=10,
        validation_alias=AliasChoices("graph_tag_min_shared_tags", "memory_graph_tag_min_shared_tags"),
        description=(
            "PRD-CORE-245 FR07: how many tags two entries must share before the "
            "derived tag relation links them. Two is the predicate the deleted "
            "materialised tag_cooccurrence edges claimed to encode; one makes "
            "every entry sharing a single common tag a neighbour, which on the "
            "reference corpus means 3,467 neighbours for 'documentation' alone."
        ),
    )
    graph_tag_max_tag_postings: int = Field(
        default=500,
        ge=1,
        le=100_000,
        validation_alias=AliasChoices("graph_tag_max_tag_postings", "memory_graph_tag_max_tag_postings"),
        description=(
            "PRD-CORE-245 FR07: a tag with more postings than this is treated as "
            "noise and excluded from the derivation. Measured on the reference "
            "corpus (2026-09-03): the top tags are 'documentation' 3,467, "
            "'architecture' 2,915, 'testing' 2,612 — labels that say nothing "
            "about which two entries belong together. 500 measured 7.80 ms per "
            "root at 13.2 neighbours; 1000 measured 7.91 ms at 14.9."
        ),
    )
    graph_tag_derive_top_k: int = Field(
        default=25,
        ge=1,
        le=200,
        validation_alias=AliasChoices("graph_tag_derive_top_k", "memory_graph_tag_derive_top_k"),
        description=(
            "PRD-CORE-245 FR07: maximum derived tag neighbours returned for one "
            "root. Unbounded derivation returns a mean 573.3 neighbours per root "
            "on the reference corpus, which is a result set no caller can use."
        ),
    )
