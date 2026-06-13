"""Wave 15: coverage gap-fill for code_index/search.py (lines 50, 85, 93-96, 131)."""
from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory.code_index.models import CodeChunk
from trw_memory.code_index.indexer import InMemoryCodeIndex
from trw_memory.code_index.search import CodeSearchEngine, _lexical_score


_CONTENT_HASH = "a" * 64  # valid sha256 hex placeholder


def _chunk(**kwargs: object) -> CodeChunk:
    defaults: dict[str, object] = {
        "namespace": "project:default",
        "file_id": "file-001",
        "path": "src/foo.py",
        "language": "python",
        "start_line": 1,
        "end_line": 10,
        "content_hash": _CONTENT_HASH,
        "symbols": ["my_func"],
        "snippet": "def my_func(): pass",
    }
    defaults.update(kwargs)
    return CodeChunk(**defaults)  # type: ignore[arg-type]


class TestCodeSearchEngineLimitZero:
    def test_limit_zero_returns_empty(self) -> None:
        """code_search with limit <= 0 → return [] immediately (line 50)."""
        store = InMemoryCodeIndex()
        engine = CodeSearchEngine(store=store)
        result = engine.code_search(query="anything", namespace="project:default", limit=0)
        assert result == []


class TestEmbeddingStatusSuccessPath:
    def test_embedding_available_with_chunks(self) -> None:
        """_embedding_status with provider → embed_query + embed_documents (lines 84-96)."""
        store = InMemoryCodeIndex()
        provider = MagicMock()
        provider.name = "test-provider"
        provider.model = "test-model"
        provider.embed_query.return_value = [0.1, 0.2, 0.3]
        provider.embed_documents.return_value = [[0.4, 0.5, 0.6, 0.7]]  # longer dim

        engine = CodeSearchEngine(store=store, embedding_provider=provider)
        chunk = _chunk()
        meta = engine._embedding_status(query="foo", chunks=[chunk])

        assert meta.available is True
        assert meta.dimensions == 4  # max(3, 4) = 4 (lines 93-95)

    def test_embedding_available_without_chunks_uses_query_dim(self) -> None:
        """_embedding_status with no chunks → only query vector dim used (lines 93-96)."""
        store = InMemoryCodeIndex()
        provider = MagicMock()
        provider.name = "test-provider"
        provider.model = "test-model"
        provider.embed_query.return_value = [0.1, 0.2]
        provider.embed_documents.return_value = []

        engine = CodeSearchEngine(store=store, embedding_provider=provider)
        meta = engine._embedding_status(query="foo", chunks=[])

        assert meta.available is True
        assert meta.dimensions == 2  # line 93: dimensions = len(query_vector)


class TestLexicalScoreEmptyQuery:
    def test_empty_query_tokens_returns_zero(self) -> None:
        """_lexical_score with no query tokens → 0.0 (line 131)."""
        chunk = _chunk()
        score = _lexical_score(query_tokens=set(), chunk=chunk)
        assert score == 0.0
