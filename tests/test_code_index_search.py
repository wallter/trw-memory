"""Tests for explicit code search and symbol lookup APIs."""

from __future__ import annotations

from collections.abc import Sequence

from trw_memory.code_index.indexer import InMemoryCodeIndex
from trw_memory.code_index.models import CodeChunk, CodeFile, CodeSymbol
from trw_memory.code_index.search import CodeEmbeddingProvider, CodeSearchEngine, code_search
from trw_memory.code_index.symbols import lookup_symbols


class UnavailableEmbeddingProvider:
    name = "offline"
    model = "none"

    def embed_query(self, query: str) -> Sequence[float]:
        raise RuntimeError("offline")

    def embed_documents(self, documents: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError("offline")


def _seed_store() -> InMemoryCodeIndex:
    store = InMemoryCodeIndex()
    content_hash = "1" * 64
    code_file = CodeFile(namespace="project", path="src/service.py", language="python", content_hash=content_hash)
    chunk = CodeChunk(
        namespace="project",
        file_id=code_file.id,
        path=code_file.path,
        language="python",
        start_line=1,
        end_line=4,
        content_hash=content_hash,
        snippet="def calculate_total(items: list[int]) -> int:\n    return sum(items)",
        symbols=["calculate_total"],
    )
    symbol = CodeSymbol(
        namespace="project",
        file_id=code_file.id,
        path=code_file.path,
        language="python",
        name="calculate_total",
        kind="function",
        start_line=1,
        end_line=2,
    )
    duplicate = CodeSymbol(
        namespace="project",
        file_id=code_file.id,
        path="src/other.py",
        language="python",
        name="calculate_total",
        kind="function",
        start_line=10,
        end_line=12,
    )
    store.upsert_file(code_file, chunks=[chunk], symbols=[symbol, duplicate])
    return store


def test_code_search_is_explicit_and_returns_bounded_ranked_results() -> None:
    store = _seed_store()

    results = code_search(store, query="calculate total", namespace="project", limit=5)

    assert len(results) == 1
    assert results[0].file == "src/service.py"
    assert results[0].symbol == "calculate_total"
    assert results[0].line_range == (1, 4)
    assert results[0].score > 0
    assert len(results[0].snippet) <= 400


def test_code_search_filters_namespace_path_language_and_degrades_without_embeddings() -> None:
    store = _seed_store()
    engine = CodeSearchEngine(store=store, embedding_provider=UnavailableEmbeddingProvider())

    results = engine.code_search(
        query="sum items",
        namespace="project",
        path_glob="src/*.py",
        language="python",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].embedding.provider == "offline"
    assert results[0].embedding.available is False
    assert results[0].embedding.reason == "embedding provider unavailable; lexical fallback used"
    assert engine.code_search(query="sum", namespace="other") == []
    assert engine.code_search(query="sum", namespace="project", path_glob="tests/*.py") == []
    assert engine.code_search(query="sum", namespace="project", language="typescript") == []


def test_symbol_lookup_returns_duplicate_matches_with_disambiguation() -> None:
    store = _seed_store()

    matches = lookup_symbols(store, namespace="project", name="calculate_total", kind="function")

    assert [(match.name, match.kind, match.path, match.line_range) for match in matches] == [
        ("calculate_total", "function", "src/other.py", (10, 12)),
        ("calculate_total", "function", "src/service.py", (1, 2)),
    ]
    assert [match.disambiguation for match in matches] == [
        "python:function:src/other.py:10-12",
        "python:function:src/service.py:1-2",
    ]


def test_embedding_provider_protocol_is_independent_from_general_memory_embeddings() -> None:
    provider: CodeEmbeddingProvider = UnavailableEmbeddingProvider()

    assert provider.name == "offline"
