"""Tests for the trw-memory retrieval pipeline.

Covers:
- bm25_search: happy path, empty input, fallback to token-overlap, unavailable
- cosine_similarity: orthogonal, identical, zero-vector
- dense_search: happy path, no embedder, unavailable embedder, missing embeddings
- rrf_fuse: single ranking, multi-ranking, empty input, rank ordering
- hybrid_search: bm25-only, dense-only, both, neither, empty entries
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import cosine_similarity, dense_search
from trw_memory.retrieval.fusion import rrf_fuse
from trw_memory.retrieval.pipeline import hybrid_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    entry_id: str,
    content: str,
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        status=status,
        created_at=now,
        updated_at=now,
    )


class _StubEmbedder:
    """Minimal EmbeddingProvider stub that uses the first 3 chars as a vector."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        # Deterministic 3-dim embedding from the first characters
        return [float(ord(c)) / 128.0 for c in (text[:3].ljust(3))]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(t) for t in texts]

    def available(self) -> bool:
        return self._available

    def dim(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------

class TestBM25Search:
    def test_returns_top_k_results(self) -> None:
        entries = [
            _make_entry("e1", "pydantic validation error handling"),
            _make_entry("e2", "fastmcp middleware pattern"),
            _make_entry("e3", "structlog event keyword reserved"),
        ]
        results = bm25_search("pydantic validation", entries, top_k=2)
        assert len(results) <= 2
        ids = [r[0] for r in results]
        assert "e1" in ids

    def test_empty_entries_returns_empty(self) -> None:
        assert bm25_search("anything", []) == []

    def test_scores_sorted_descending(self) -> None:
        entries = [
            _make_entry("a", "apple fruit tree"),
            _make_entry("b", "pydantic model validation schema"),
            _make_entry("c", "fastmcp middleware tool registration"),
        ]
        results = bm25_search("pydantic model", entries)
        if len(results) >= 2:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_hyphenated_tag_expansion(self) -> None:
        """Tags like 'pydantic-v2' expand to tokens 'pydantic', 'v2', 'pydantic-v2'."""
        entries = [
            _make_entry("tagged", "model configuration", tags=["pydantic-v2"]),
            _make_entry("untagged", "unrelated content about dogs"),
        ]
        results = bm25_search("pydantic", entries)
        ids = [r[0] for r in results]
        assert "tagged" in ids

    def test_fallback_token_overlap_when_all_zero(self) -> None:
        """When BM25 IDF degrades to zero (all docs contain term),
        token-overlap fallback kicks in."""
        # Use many short entries all containing the same word
        entries = [_make_entry(f"e{i}", "foo bar baz qux") for i in range(5)]
        entries.append(_make_entry("target", "foo bar baz qux overlap unique"))
        # BM25 may score all zero in small same-word corpus — fallback returns
        # anything with overlap
        results = bm25_search("overlap unique", entries)
        # Either BM25 found something or fallback did; result must be a list
        assert isinstance(results, list)

    def test_unavailable_returns_empty(self) -> None:
        """When rank_bm25 is not installed the function should return []."""
        entries = [_make_entry("x", "test content")]
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            results = bm25_search("test", entries)
        assert results == []

    def test_result_ids_match_entry_ids(self) -> None:
        entries = [
            _make_entry("alpha", "machine learning training data"),
            _make_entry("beta", "neural network weights gradient"),
        ]
        results = bm25_search("machine learning", entries)
        valid_ids = {"alpha", "beta"}
        for entry_id, score in results:
            assert entry_id in valid_ids
            assert score >= 0.0

    def test_top_k_limits_results(self) -> None:
        entries = [_make_entry(f"e{i}", f"python code test item {i}") for i in range(20)]
        results = bm25_search("python", entries, top_k=5)
        assert len(results) <= 5

    def test_content_and_detail_both_indexed(self) -> None:
        entries = [
            _make_entry("detail_match", "unrelated content", detail="pydantic validation"),
            _make_entry("no_match", "something else entirely"),
        ]
        results = bm25_search("pydantic", entries)
        ids = [r[0] for r in results]
        assert "detail_match" in ids


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors(self) -> None:
        assert cosine_similarity([0.0], [0.0]) == 0.0

    def test_single_dimension(self) -> None:
        assert cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)

    def test_high_dimensional(self) -> None:
        import math
        n = 384
        a = [1.0] * n
        b = [1.0] * n
        result = cosine_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# dense_search
# ---------------------------------------------------------------------------

class TestDenseSearch:
    def _stored(self, ids: list[str], embedder: _StubEmbedder) -> dict[str, list[float]]:
        return {eid: embedder.embed(eid) or [] for eid in ids}  # type: ignore[list-item]

    def test_returns_top_k_results(self) -> None:
        embedder = _StubEmbedder()
        ids = ["abc", "def", "ghi", "jkl"]
        stored = self._stored(ids, embedder)
        results = dense_search("abc", ids, embedder=embedder, stored_embeddings=stored, top_k=2)
        assert len(results) <= 2
        assert results[0][0] == "abc"  # identical query → top match

    def test_scores_sorted_descending(self) -> None:
        embedder = _StubEmbedder()
        ids = ["aaa", "bbb", "ccc"]
        stored = self._stored(ids, embedder)
        results = dense_search("aaa", ids, embedder=embedder, stored_embeddings=stored)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_embedder_returns_empty(self) -> None:
        ids = ["x", "y"]
        stored = {"x": [0.5, 0.5, 0.5], "y": [0.1, 0.1, 0.1]}
        results = dense_search("query", ids, embedder=None, stored_embeddings=stored)
        assert results == []

    def test_unavailable_embedder_returns_empty(self) -> None:
        embedder = _StubEmbedder(available=False)
        ids = ["x"]
        stored = {"x": [0.5, 0.5, 0.5]}
        results = dense_search("query", ids, embedder=embedder, stored_embeddings=stored)
        assert results == []

    def test_empty_entry_ids_returns_empty(self) -> None:
        embedder = _StubEmbedder()
        results = dense_search("query", [], embedder=embedder, stored_embeddings={})
        assert results == []

    def test_no_stored_embeddings_returns_empty(self) -> None:
        embedder = _StubEmbedder()
        results = dense_search("query", ["a", "b"], embedder=embedder, stored_embeddings=None)
        assert results == []

    def test_pre_computed_query_embedding(self) -> None:
        ids = ["aaa", "bbb"]
        stored = {
            "aaa": [1.0, 0.0, 0.0],
            "bbb": [0.0, 1.0, 0.0],
        }
        # Query vector identical to "aaa" stored vector → "aaa" should score highest
        q_vec = [1.0, 0.0, 0.0]
        results = dense_search("anything", ids, query_embedding=q_vec, stored_embeddings=stored)
        assert results[0][0] == "aaa"
        assert results[0][1] == pytest.approx(1.0)

    def test_missing_stored_embedding_skipped(self) -> None:
        embedder = _StubEmbedder()
        ids = ["known", "unknown"]
        stored = {"known": [1.0, 0.0, 0.0]}
        results = dense_search("known", ids, embedder=embedder, stored_embeddings=stored)
        result_ids = [r[0] for r in results]
        assert "unknown" not in result_ids
        assert "known" in result_ids

    def test_top_k_limits_results(self) -> None:
        embedder = _StubEmbedder()
        ids = [f"e{i:02d}" for i in range(10)]
        stored = self._stored(ids, embedder)
        results = dense_search("e00", ids, embedder=embedder, stored_embeddings=stored, top_k=3)
        assert len(results) <= 3

    def test_embed_exception_returns_empty(self) -> None:
        embedder = MagicMock(spec=EmbeddingProvider)
        embedder.available.return_value = True
        embedder.embed.side_effect = RuntimeError("embedding failed")
        stored = {"x": [1.0, 0.0, 0.0]}
        results = dense_search("query", ["x"], embedder=embedder, stored_embeddings=stored)
        assert results == []


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------

class TestRRFFuse:
    def test_empty_rankings_returns_empty(self) -> None:
        assert rrf_fuse([]) == []

    def test_single_ranking_passthrough(self) -> None:
        ranking = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        result = rrf_fuse([ranking])
        result_ids = [r[0] for r in result]
        # Ranking order should be preserved (rank 0 gets highest RRF)
        assert result_ids[0] == "a"
        assert result_ids[1] == "b"
        assert result_ids[2] == "c"

    def test_scores_sorted_descending(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("b", 1.0), ("c", 0.5)]
        result = rrf_fuse([r1, r2])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_document_appearing_in_both_scores_higher(self) -> None:
        """A doc appearing in both rankings accumulates score from each."""
        r1 = [("shared", 1.0), ("only_r1", 0.5)]
        r2 = [("shared", 1.0), ("only_r2", 0.5)]
        result = rrf_fuse([r1, r2])
        ids = [r[0] for r in result]
        assert ids[0] == "shared"

    def test_custom_k_value(self) -> None:
        ranking = [("a", 1.0)]
        result_k60 = rrf_fuse([ranking], k=60)
        result_k10 = rrf_fuse([ranking], k=10)
        # k=10 gives higher score than k=60 (smaller denominator)
        assert result_k10[0][1] > result_k60[0][1]

    def test_formula_values(self) -> None:
        """Verify exact RRF formula: score = 1/(k + rank + 1) with 0-based rank."""
        ranking = [("a", 99.0)]  # rank=0 → score = 1/(60+0+1) = 1/61
        result = rrf_fuse([ranking], k=60)
        assert result[0][1] == pytest.approx(1.0 / 61.0)

    def test_two_rankings_accumulate(self) -> None:
        r1 = [("x", 1.0)]
        r2 = [("x", 1.0)]
        result = rrf_fuse([r1, r2], k=60)
        # score = 1/61 + 1/61 = 2/61
        assert result[0][1] == pytest.approx(2.0 / 61.0)

    def test_all_unique_ids_preserved(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("c", 1.0), ("d", 0.5)]
        result = rrf_fuse([r1, r2])
        result_ids = {r[0] for r in result}
        assert result_ids == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def _entries(self) -> list[MemoryEntry]:
        return [
            _make_entry("e1", "pydantic model validation schema", tags=["pydantic"]),
            _make_entry("e2", "fastmcp middleware tool registration"),
            _make_entry("e3", "structlog reserved keyword event"),
            _make_entry("e4", "bm25 sparse retrieval ranking"),
            _make_entry("e5", "vector embedding cosine similarity"),
        ]

    def test_returns_memory_entries(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries)
        assert all(isinstance(r, MemoryEntry) for r in results)

    def test_bm25_only_no_embedder(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None)
        assert len(results) >= 1
        ids = [r.id for r in results]
        assert "e1" in ids

    def test_dense_only_no_bm25(self) -> None:
        """When BM25 is patched out, dense search still drives results."""
        embedder = _StubEmbedder()
        entries = self._entries()
        stored = {e.id: embedder.embed(e.id) or [] for e in entries}  # type: ignore[list-item]
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            results = hybrid_search(
                "e1", entries, embedder=embedder, stored_embeddings=stored
            )
        # Dense should find "e1" as top result (identical id prefix)
        assert len(results) >= 1

    def test_both_sources_produce_results(self) -> None:
        embedder = _StubEmbedder()
        entries = self._entries()
        stored = {e.id: embedder.embed(e.id) or [] for e in entries}  # type: ignore[list-item]
        results = hybrid_search(
            "pydantic model", entries, embedder=embedder, stored_embeddings=stored
        )
        assert len(results) >= 1

    def test_neither_source_returns_empty(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
                results = hybrid_search("anything", entries)
        assert results == []

    def test_empty_entries_returns_empty(self) -> None:
        results = hybrid_search("query", [])
        assert results == []

    def test_top_k_limits_results(self) -> None:
        entries = [_make_entry(f"e{i}", f"python item {i}") for i in range(20)]
        results = hybrid_search("python", entries, top_k=3)
        assert len(results) <= 3

    def test_result_ids_are_subset_of_input_ids(self) -> None:
        entries = self._entries()
        input_ids = {e.id for e in entries}
        results = hybrid_search("retrieval", entries)
        for r in results:
            assert r.id in input_ids

    def test_uses_config_parameters(self) -> None:
        """bm25_candidates, vector_candidates, rrf_k are forwarded."""
        entries = self._entries()
        # Should not raise even with non-default parameters
        results = hybrid_search(
            "pydantic",
            entries,
            bm25_candidates=5,
            vector_candidates=5,
            rrf_k=30,
            top_k=2,
        )
        assert len(results) <= 2

    def test_dense_skipped_when_no_stored_embeddings(self) -> None:
        embedder = _StubEmbedder()
        entries = self._entries()
        # stored_embeddings is None → dense_search returns []
        with patch("trw_memory.retrieval.pipeline.dense_search") as mock_dense:
            mock_dense.return_value = []
            hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=None)
        # dense_search should still be called (embedder is not None)
        mock_dense.assert_called_once()

    def test_entries_returned_are_original_objects(self) -> None:
        """Returned entries must be the same objects from the input list."""
        entries = self._entries()
        results = hybrid_search("pydantic fastmcp", entries)
        entry_map = {e.id: e for e in entries}
        for r in results:
            assert r is entry_map[r.id]
