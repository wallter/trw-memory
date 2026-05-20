"""Explicit code search API with lexical fallback and optional embeddings."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

from trw_memory.code_index.indexer import InMemoryCodeIndex
from trw_memory.code_index.models import CodeChunk, CodeSearchResult, EmbeddingMetadata, make_bounded_snippet

__all__ = ["CodeEmbeddingProvider", "CodeSearchEngine", "code_search"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class CodeEmbeddingProvider(Protocol):
    """Optional code embedding provider, independent from general memory embeddings."""

    name: str
    model: str

    def embed_query(self, query: str) -> Sequence[float]:
        """Embed a query string."""

    def embed_documents(self, documents: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed code snippets."""


class CodeSearchEngine:
    """Search an explicit code-index store without touching default memory recall."""

    def __init__(self, *, store: InMemoryCodeIndex, embedding_provider: CodeEmbeddingProvider | None = None) -> None:
        self._store = store
        self._embedding_provider = embedding_provider

    def code_search(
        self,
        *,
        query: str,
        namespace: str,
        path_glob: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[CodeSearchResult]:
        """Return ranked code chunks with file, symbol, line range, score, and bounded snippet."""

        if limit <= 0:
            return []
        chunks = self._store.list_chunks(namespace=namespace, path_glob=path_glob, language=language)
        embedding_metadata = self._embedding_status(query=query, chunks=chunks)
        query_tokens = _tokens(query)
        scored = []
        for chunk in chunks:
            score = _lexical_score(query_tokens=query_tokens, chunk=chunk)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: (-item[1], item[0].path, item[0].start_line))
        return [
            CodeSearchResult(
                file=chunk.path,
                symbol=chunk.symbols[0] if chunk.symbols else "",
                language=chunk.language,
                line_range=(chunk.start_line, chunk.end_line),
                score=score,
                snippet=make_bounded_snippet(
                    chunk.snippet,
                    start_line=1,
                    end_line=len(chunk.snippet.splitlines()),
                    max_lines=20,
                    max_chars=400,
                ),
                embedding=embedding_metadata,
            )
            for chunk, score in scored[:limit]
        ]

    def _embedding_status(self, *, query: str, chunks: list[CodeChunk]) -> EmbeddingMetadata:
        if self._embedding_provider is None:
            return EmbeddingMetadata(available=False, reason="embedding provider not configured")
        provider = self._embedding_provider
        try:
            query_vector = provider.embed_query(query)
            document_vectors = provider.embed_documents([chunk.snippet for chunk in chunks[:1]]) if chunks else []
        except Exception:
            return EmbeddingMetadata(
                provider=provider.name,
                model=provider.model,
                available=False,
                reason="embedding provider unavailable; lexical fallback used",
            )
        dimensions = len(query_vector)
        if document_vectors:
            dimensions = max(dimensions, len(document_vectors[0]))
        return EmbeddingMetadata(provider=provider.name, model=provider.model, dimensions=dimensions, available=True)


def code_search(
    store: InMemoryCodeIndex,
    *,
    query: str,
    namespace: str,
    path_glob: str | None = None,
    language: str | None = None,
    limit: int = 10,
    embedding_provider: CodeEmbeddingProvider | None = None,
) -> list[CodeSearchResult]:
    """Convenience explicit code search API."""

    return CodeSearchEngine(store=store, embedding_provider=embedding_provider).code_search(
        query=query,
        namespace=namespace,
        path_glob=path_glob,
        language=language,
        limit=limit,
    )


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        lowered = token.lower()
        tokens.add(lowered)
        tokens.update(part for part in lowered.split("_") if part)
    return tokens


def _lexical_score(*, query_tokens: set[str], chunk: CodeChunk) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokens(f"{chunk.snippet} {' '.join(chunk.symbols)}")
    overlap = query_tokens & chunk_tokens
    symbol_bonus = sum(1 for symbol in chunk.symbols if _tokens(symbol) & query_tokens)
    return float(len(overlap)) + math.log1p(symbol_bonus)
