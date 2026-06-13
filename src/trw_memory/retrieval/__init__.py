"""Retrieval pipeline for trw-memory.

Public API:

- :func:`~trw_memory.retrieval.bm25.bm25_search` — BM25 sparse retrieval
- :func:`~trw_memory.retrieval.dense.dense_search` — dense vector search
- :func:`~trw_memory.retrieval.dense.cosine_similarity` — vector similarity helper
- :func:`~trw_memory.retrieval.fusion.rrf_fuse` — Reciprocal Rank Fusion (sum)
- :func:`~trw_memory.retrieval.fusion.combmax_fuse` — CombMAX fusion (max reciprocal rank)
- :func:`~trw_memory.retrieval.pipeline.hybrid_search` — combined pipeline
- :func:`~trw_memory.retrieval.recency.recency_rank` — recency-based ranking
- :func:`~trw_memory.retrieval.recency.recency_score` — per-entry recency score
- :func:`~trw_memory.retrieval.reranker.cross_encode_rerank` — cross-encoder re-ranking
- :func:`~trw_memory.retrieval.temporal_query.classify_temporal` — temporal query classifier
- :func:`~trw_memory.retrieval.token_budget.estimate_tokens` — word-count token estimate
- :func:`~trw_memory.retrieval.token_budget.estimate_entry_tokens` — entry-level token cost
- :func:`~trw_memory.retrieval.token_budget.apply_token_budget` — budget-fit a result list
- :data:`~trw_memory.retrieval.token_budget.TOKEN_MULTIPLIER` — tokens-per-word ratio
- :data:`~trw_memory.retrieval.token_budget.METADATA_OVERHEAD` — fixed per-entry overhead
"""

from __future__ import annotations

from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import cosine_similarity, dense_search
from trw_memory.retrieval.fusion import combmax_fuse, rrf_fuse
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.retrieval.recency import recency_rank, recency_score
from trw_memory.retrieval.reranker import cross_encode_rerank
from trw_memory.retrieval.temporal_query import classify_temporal
from trw_memory.retrieval.token_budget import (
    METADATA_OVERHEAD as METADATA_OVERHEAD,
)
from trw_memory.retrieval.token_budget import (
    TOKEN_MULTIPLIER as TOKEN_MULTIPLIER,
)
from trw_memory.retrieval.token_budget import (
    apply_token_budget as apply_token_budget,
)
from trw_memory.retrieval.token_budget import (
    estimate_entry_tokens as estimate_entry_tokens,
)
from trw_memory.retrieval.token_budget import (
    estimate_tokens as estimate_tokens,
)

__all__ = [
    "METADATA_OVERHEAD",
    "TOKEN_MULTIPLIER",
    "apply_token_budget",
    "bm25_search",
    "combmax_fuse",
    "cosine_similarity",
    "cross_encode_rerank",
    "dense_search",
    "estimate_entry_tokens",
    "estimate_tokens",
    "hybrid_search",
    "recency_rank",
    "recency_score",
    "rrf_fuse",
    "classify_temporal",
]
