"""Dense retrieval and cosine similarity tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.retrieval.dense import cosine_similarity, dense_search

from ._test_retrieval_support import StubEmbedder, stored_embeddings_for


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vector = [1.0, 2.0, 3.0]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors(self) -> None:
        assert cosine_similarity([0.0], [0.0]) == 0.0

    def test_single_dimension(self) -> None:
        assert cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)

    def test_high_dimensional(self) -> None:
        n = 384
        assert cosine_similarity([1.0] * n, [1.0] * n) == pytest.approx(1.0, abs=1e-6)


class TestDenseSearch:
    def test_returns_top_k_results(self) -> None:
        embedder = StubEmbedder()
        ids = ["abc", "def", "ghi", "jkl"]
        stored = stored_embeddings_for(ids, embedder)
        results = dense_search("abc", ids, embedder=embedder, stored_embeddings=stored, top_k=2)
        assert len(results) <= 2
        assert results[0][0] == "abc"

    def test_scores_sorted_descending(self) -> None:
        embedder = StubEmbedder()
        ids = ["aaa", "bbb", "ccc"]
        stored = stored_embeddings_for(ids, embedder)
        results = dense_search("aaa", ids, embedder=embedder, stored_embeddings=stored)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_embedder_returns_empty(self) -> None:
        stored = {"x": [0.5, 0.5, 0.5], "y": [0.1, 0.1, 0.1]}
        results = dense_search("query", ["x", "y"], embedder=None, stored_embeddings=stored)
        assert results == []

    def test_unavailable_embedder_returns_empty(self) -> None:
        embedder = StubEmbedder(available=False)
        results = dense_search("query", ["x"], embedder=embedder, stored_embeddings={"x": [0.5, 0.5, 0.5]})
        assert results == []

    def test_empty_entry_ids_returns_empty(self) -> None:
        embedder = StubEmbedder()
        results = dense_search("query", [], embedder=embedder, stored_embeddings={})
        assert results == []

    def test_no_stored_embeddings_returns_empty(self) -> None:
        embedder = StubEmbedder()
        results = dense_search("query", ["a", "b"], embedder=embedder, stored_embeddings=None)
        assert results == []

    def test_pre_computed_query_embedding(self) -> None:
        ids = ["aaa", "bbb"]
        stored = {
            "aaa": [1.0, 0.0, 0.0],
            "bbb": [0.0, 1.0, 0.0],
        }
        results = dense_search("anything", ids, query_embedding=[1.0, 0.0, 0.0], stored_embeddings=stored)
        assert results[0][0] == "aaa"
        assert results[0][1] == pytest.approx(1.0)

    def test_missing_stored_embedding_skipped(self) -> None:
        embedder = StubEmbedder()
        ids = ["known", "unknown"]
        stored = {"known": [1.0, 0.0, 0.0]}
        results = dense_search("known", ids, embedder=embedder, stored_embeddings=stored)
        result_ids = [entry_id for entry_id, _ in results]
        assert "unknown" not in result_ids
        assert "known" in result_ids

    def test_top_k_limits_results(self) -> None:
        embedder = StubEmbedder()
        ids = [f"e{i:02d}" for i in range(10)]
        stored = stored_embeddings_for(ids, embedder)
        results = dense_search("e00", ids, embedder=embedder, stored_embeddings=stored, top_k=3)
        assert len(results) <= 3

    def test_embed_exception_returns_empty(self) -> None:
        embedder = MagicMock(spec=EmbeddingProvider)
        embedder.available.return_value = True
        embedder.embed.side_effect = RuntimeError("embedding failed")
        results = dense_search("query", ["x"], embedder=embedder, stored_embeddings={"x": [1.0, 0.0, 0.0]})
        assert results == []
