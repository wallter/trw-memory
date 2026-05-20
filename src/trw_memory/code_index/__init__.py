"""Explicit code-index APIs for trw-memory.

The package is intentionally separate from default memory recall surfaces. Callers
must opt in by constructing a code index store/indexer or by calling
``code_search`` with a code-index store.
"""

from __future__ import annotations

from trw_memory.code_index.chunker import CodeChunker
from trw_memory.code_index.indexer import CodeIndexer, InMemoryCodeIndex
from trw_memory.code_index.models import CodeChunk, CodeFile, CodeSearchResult, CodeSymbol, EmbeddingMetadata
from trw_memory.code_index.search import CodeSearchEngine, code_search
from trw_memory.code_index.symbols import lookup_symbols

__all__ = [
    "CodeChunk",
    "CodeChunker",
    "CodeFile",
    "CodeIndexer",
    "CodeSearchEngine",
    "CodeSearchResult",
    "CodeSymbol",
    "EmbeddingMetadata",
    "InMemoryCodeIndex",
    "code_search",
    "lookup_symbols",
]
