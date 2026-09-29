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

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static tooling sees every re-export. At runtime each name resolves on first
    # access through ``__getattr__`` below, so importing this package, or any of
    # its submodules (which runs this file first), costs nothing. The eager form
    # pulled numpy in through ``dense`` on every edit hook's recall path, and the
    # reranker would pull in ``sentence_transformers``/``torch`` (production
    # feedback sub_psVs_nUWnLJGvOs3).
    from trw_memory.retrieval.bm25 import bm25_search as bm25_search
    from trw_memory.retrieval.dense import cosine_similarity as cosine_similarity
    from trw_memory.retrieval.dense import dense_search as dense_search
    from trw_memory.retrieval.fusion import blend_recency as blend_recency
    from trw_memory.retrieval.fusion import combmax_fuse as combmax_fuse
    from trw_memory.retrieval.fusion import rrf_fuse as rrf_fuse
    from trw_memory.retrieval.pipeline import ScoredCandidate as ScoredCandidate
    from trw_memory.retrieval.pipeline import hybrid_search as hybrid_search
    from trw_memory.retrieval.pipeline import hybrid_search_scored as hybrid_search_scored
    from trw_memory.retrieval.recency import recency_rank as recency_rank
    from trw_memory.retrieval.recency import recency_score as recency_score
    from trw_memory.retrieval.reranker import cross_encode_rerank as cross_encode_rerank
    from trw_memory.retrieval.temporal_query import classify_temporal as classify_temporal
    from trw_memory.retrieval.temporal_query import prepare_temporal_query as prepare_temporal_query
    from trw_memory.retrieval.temporal_query import strip_temporal_arithmetic as strip_temporal_arithmetic
    from trw_memory.retrieval.temporal_query import strip_temporal_prefix as strip_temporal_prefix
    from trw_memory.retrieval.token_budget import METADATA_OVERHEAD as METADATA_OVERHEAD
    from trw_memory.retrieval.token_budget import TOKEN_MULTIPLIER as TOKEN_MULTIPLIER
    from trw_memory.retrieval.token_budget import apply_token_budget as apply_token_budget
    from trw_memory.retrieval.token_budget import estimate_entry_tokens as estimate_entry_tokens
    from trw_memory.retrieval.token_budget import estimate_serialized_entry_tokens as estimate_serialized_entry_tokens
    from trw_memory.retrieval.token_budget import estimate_tokens as estimate_tokens

#: Public name -> the submodule that defines it.
_LAZY: dict[str, str] = {
    "bm25_search": "bm25",
    "cosine_similarity": "dense",
    "dense_search": "dense",
    "blend_recency": "fusion",
    "combmax_fuse": "fusion",
    "rrf_fuse": "fusion",
    "ScoredCandidate": "pipeline",
    "hybrid_search": "pipeline",
    "hybrid_search_scored": "pipeline",
    "recency_rank": "recency",
    "recency_score": "recency",
    "cross_encode_rerank": "reranker",
    "classify_temporal": "temporal_query",
    "prepare_temporal_query": "temporal_query",
    "strip_temporal_arithmetic": "temporal_query",
    "strip_temporal_prefix": "temporal_query",
    "METADATA_OVERHEAD": "token_budget",
    "TOKEN_MULTIPLIER": "token_budget",
    "apply_token_budget": "token_budget",
    "estimate_entry_tokens": "token_budget",
    "estimate_serialized_entry_tokens": "token_budget",
    "estimate_tokens": "token_budget",
}


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export: resolve *name* from its submodule on first access, then cache it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Lazy exports are listed before first access, as eager re-exports were."""
    return sorted({*globals(), *_LAZY})


#: Derived from ``_LAZY`` rather than hand-duplicated: every lazily-exported name is public.
__all__ = sorted(_LAZY)
