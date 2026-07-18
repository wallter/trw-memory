"""Retrieval pipeline for trw-memory.

Public API:

- :func:`~trw_memory.retrieval.bm25.bm25_search` — BM25 sparse retrieval
- :func:`~trw_memory.retrieval.dense.dense_search` — dense vector search
- :func:`~trw_memory.retrieval.dense.cosine_similarity` — vector similarity helper
- :func:`~trw_memory.retrieval.fusion.rrf_fuse` — Reciprocal Rank Fusion (sum)
- :func:`~trw_memory.retrieval.fusion.combmax_fuse` — CombMAX fusion (max reciprocal rank)
- :func:`~trw_memory.retrieval.fusion.blend_recency` — linear recency/relevance blend
- :func:`~trw_memory.retrieval.pipeline.hybrid_search` — combined pipeline
- :func:`~trw_memory.retrieval.recency.recency_rank` — recency-based ranking
- :func:`~trw_memory.retrieval.recency.recency_score` — per-entry recency score
- :func:`~trw_memory.retrieval.reranker.cross_encode_rerank` — cross-encoder re-ranking
- :func:`~trw_memory.retrieval.temporal_query.classify_temporal` — temporal query classifier
- :func:`~trw_memory.retrieval.token_budget.estimate_tokens` — word-count token estimate
- :func:`~trw_memory.retrieval.token_budget.estimate_entry_tokens` — entry-level token cost
- :func:`~trw_memory.retrieval.token_budget.estimate_serialized_entry_tokens` — full-serialization token cost
- :func:`~trw_memory.retrieval.token_budget.apply_token_budget` — budget-fit a result list
- :data:`~trw_memory.retrieval.token_budget.TOKEN_MULTIPLIER` — tokens-per-word ratio
- :data:`~trw_memory.retrieval.token_budget.METADATA_OVERHEAD` — fixed per-entry overhead
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import cosine_similarity, dense_search
from trw_memory.retrieval.fusion import blend_recency, combmax_fuse, rrf_fuse
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.retrieval.recency import recency_rank, recency_score
from trw_memory.retrieval.temporal_query import (
    classify_temporal,
    prepare_temporal_query,
    resolve_temporal_arithmetic_offset,
    strip_temporal_arithmetic,
    strip_temporal_prefix,
)
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
    estimate_serialized_entry_tokens as estimate_serialized_entry_tokens,
)
from trw_memory.retrieval.token_budget import (
    estimate_tokens as estimate_tokens,
)

if TYPE_CHECKING:
    # Re-export for static tooling / IDEs.  At runtime this name is resolved
    # lazily by ``__getattr__`` below so importing this package does not drag in
    # ``sentence_transformers`` (and therefore ``torch``) — see reranker.py.
    from trw_memory.retrieval.reranker import cross_encode_rerank as cross_encode_rerank


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export of the cross-encoder reranker.

    ``cross_encode_rerank`` lives in :mod:`trw_memory.retrieval.reranker`, which
    only imports the heavy ``sentence_transformers``/``torch`` stack on demand.
    Keeping the re-export lazy means ``import trw_memory.retrieval`` (on the
    ``trw_mcp.server`` boot path) stays torch-free — production feedback
    sub_psVs_nUWnLJGvOs3.
    """
    if name == "cross_encode_rerank":
        from trw_memory.retrieval.reranker import cross_encode_rerank

        return cross_encode_rerank
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "METADATA_OVERHEAD",
    "TOKEN_MULTIPLIER",
    "apply_token_budget",
    "blend_recency",
    "bm25_search",
    "classify_temporal",
    "combmax_fuse",
    "cosine_similarity",
    "cross_encode_rerank",
    "dense_search",
    "estimate_entry_tokens",
    "estimate_serialized_entry_tokens",
    "estimate_tokens",
    "hybrid_search",
    "prepare_temporal_query",
    "recency_rank",
    "recency_score",
    "resolve_temporal_arithmetic_offset",
    "rrf_fuse",
    "strip_temporal_arithmetic",
    "strip_temporal_prefix",
]
