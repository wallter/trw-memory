"""Retrieval pipeline for trw-memory.

Public API:

- :func:`~trw_memory.retrieval.bm25.bm25_search` — BM25 sparse retrieval
- :func:`~trw_memory.retrieval.dense.dense_search` — dense vector search
- :func:`~trw_memory.retrieval.dense.cosine_similarity` — vector similarity helper
- :func:`~trw_memory.retrieval.fusion.rrf_fuse` — Reciprocal Rank Fusion
- :func:`~trw_memory.retrieval.pipeline.hybrid_search` — combined pipeline
"""

from __future__ import annotations

from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import cosine_similarity, dense_search
from trw_memory.retrieval.fusion import rrf_fuse
from trw_memory.retrieval.pipeline import hybrid_search

__all__ = [
    "bm25_search",
    "cosine_similarity",
    "dense_search",
    "hybrid_search",
    "rrf_fuse",
]
