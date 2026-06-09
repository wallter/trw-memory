"""Dense retrieval and cosine similarity tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock

import pytest
import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.retrieval.dense import cosine_similarity, dense_search

from ._test_retrieval_support import StubEmbedder, stored_embeddings_for

# Distinctive marker that must never surface in any log event — stands in for
# secrets / proprietary memory contents that real recall queries may carry.
SENSITIVE_QUERY = "SENSITIVE-sk_live_DEADBEEF-customer-pii-marker"


def _captured_text(events: Sequence[Mapping[str, object]]) -> str:
    """Flatten captured structlog events into a single searchable string."""
    return " ".join(f"{key}={value!r}" for event in events for key, value in event.items())


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


class TestDenseSearchTelemetryPrivacy:
    """Recall query text must never reach the logs (it may carry secrets)."""

    def test_embed_failed_logs_structure_not_query(self) -> None:
        embedder = MagicMock(spec=EmbeddingProvider)
        embedder.available.return_value = True
        embedder.embed.side_effect = RuntimeError("provider unavailable")

        with structlog.testing.capture_logs() as events:
            results = dense_search(
                SENSITIVE_QUERY,
                ["x"],
                embedder=embedder,
                stored_embeddings={"x": [1.0, 0.0, 0.0]},
            )

        assert results == []
        failures = [e for e in events if e.get("event") == "dense_search_embed_failed"]
        assert len(failures) == 1
        failure = failures[0]
        # Structural telemetry retained...
        assert failure["query_chars"] == len(SENSITIVE_QUERY)
        assert failure["candidates"] == 1
        assert failure["error_class"] == "RuntimeError"
        # ...but the raw query (or any substring of it) never leaks.
        assert SENSITIVE_QUERY not in _captured_text(events)
        assert "query" not in failure

    def test_complete_logs_structure_not_query(self) -> None:
        embedder = StubEmbedder()
        ids = ["aaa", "bbb"]
        stored = stored_embeddings_for(ids, embedder)

        with structlog.testing.capture_logs() as events:
            results = dense_search(
                SENSITIVE_QUERY,
                ids,
                embedder=embedder,
                stored_embeddings=stored,
            )

        assert results  # behaviour unchanged: real results still returned
        completes = [e for e in events if e.get("event") == "dense_search_complete"]
        assert len(completes) == 1
        complete = completes[0]
        assert complete["query_chars"] == len(SENSITIVE_QUERY)
        assert complete["candidates"] == len(ids)
        assert SENSITIVE_QUERY not in _captured_text(events)
        assert "query" not in complete
