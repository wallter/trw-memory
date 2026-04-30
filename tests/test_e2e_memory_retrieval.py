"""E2E retrieval tests for trw-memory."""

from __future__ import annotations

import pytest

from tests.conftest import make_entry
from trw_memory.exceptions import DimensionMismatchError


class TestHybridRetrieval:
    """Section 3 of E2E plan: BM25, dense, hybrid pipeline."""

    def test_bm25_keyword_search(self) -> None:
        """3.1 — BM25 ranks entries with query keyword overlap highest."""
        pytest.importorskip("rank_bm25")
        from trw_memory.retrieval.bm25 import bm25_search

        entries = [
            make_entry(
                entry_id="e1",
                content="pydantic v2 model validation with ConfigDict",
                tags=["pydantic"],
            ),
            make_entry(
                entry_id="e2",
                content="SQLAlchemy ORM session management",
                tags=["sqlalchemy"],
            ),
            make_entry(
                entry_id="e3",
                content="pydantic field validators and custom types",
                tags=["pydantic"],
            ),
        ]
        results = bm25_search("pydantic validation", entries, top_k=3)
        result_ids = [entry_id for entry_id, _ in results]
        assert "e1" in result_ids
        assert "e3" in result_ids
        if "e2" in result_ids:
            assert result_ids.index("e2") == len(result_ids) - 1

    def test_dense_cosine_similarity(self) -> None:
        """3.4 — Dense search returns results ordered by cosine similarity."""
        from trw_memory.retrieval.dense import cosine_similarity

        vector_a = [1.0, 0.0, 0.0]
        vector_b = [1.0, 0.0, 0.0]
        assert cosine_similarity(vector_a, vector_b) == pytest.approx(1.0)

        vector_c = [0.0, 1.0, 0.0]
        assert cosine_similarity(vector_a, vector_c) == pytest.approx(0.0)

        with pytest.raises(DimensionMismatchError, match="3 vs 2"):
            cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0])

    def test_hybrid_pipeline_bm25_only(self) -> None:
        """3.9 — Hybrid pipeline degrades to BM25-only when embedder is None."""
        pytest.importorskip("rank_bm25")
        from trw_memory.retrieval.pipeline import hybrid_search

        entries = [
            make_entry(entry_id="e1", content="pydantic validation patterns"),
            make_entry(entry_id="e2", content="react component lifecycle hooks"),
        ]
        results = hybrid_search(
            query="pydantic validation",
            entries=entries,
            embedder=None,
            top_k=5,
        )
        assert len(results) >= 1
        assert results[0].id == "e1"


class TestRRFFusion:
    """Verify the RRF fusion function directly."""

    def test_rrf_fuse_combines_rankings(self) -> None:
        """3.7 — RRF fusion combines BM25 and dense rankings."""
        from trw_memory.retrieval.fusion import rrf_fuse

        bm25_ranking = [("e1", 5.0), ("e2", 3.0), ("e3", 1.0)]
        dense_ranking = [("e2", 0.95), ("e1", 0.80), ("e4", 0.70)]

        fused = rrf_fuse([bm25_ranking, dense_ranking], k=60)
        fused_ids = [entry_id for entry_id, _ in fused]
        assert "e1" in fused_ids[:3]
        assert "e2" in fused_ids[:3]
        assert set(fused_ids) == {"e1", "e2", "e3", "e4"}

    def test_rrf_fuse_empty_input(self) -> None:
        """3.10 — RRF fusion with empty rankings returns empty."""
        from trw_memory.retrieval.fusion import rrf_fuse

        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []
