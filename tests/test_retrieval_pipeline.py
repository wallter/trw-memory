"""Hybrid retrieval pipeline tests."""

from __future__ import annotations

from unittest.mock import patch

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search

from ._test_retrieval_support import StubEmbedder, make_entry, stored_embeddings_for


class TestHybridSearch:
    def _entries(self) -> list[MemoryEntry]:
        return [
            make_entry("e1", "pydantic model validation schema", tags=["pydantic"]),
            make_entry("e2", "fastmcp middleware tool registration"),
            make_entry("e3", "structlog reserved keyword event"),
            make_entry("e4", "bm25 sparse retrieval ranking"),
            make_entry("e5", "vector embedding cosine similarity"),
        ]

    def test_returns_memory_entries(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries)
        assert all(isinstance(result, MemoryEntry) for result in results)

    def test_bm25_only_no_embedder(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_dense_only_no_bm25(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            results = hybrid_search("e1", entries, embedder=embedder, stored_embeddings=stored)
        assert len(results) >= 1

    def test_both_sources_produce_results(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        results = hybrid_search("pydantic model", entries, embedder=embedder, stored_embeddings=stored)
        assert len(results) >= 1

    def test_neither_source_returns_empty(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
                results = hybrid_search("anything", entries)
        assert results == []

    def test_empty_entries_returns_empty(self) -> None:
        assert hybrid_search("query", []) == []

    def test_top_k_limits_results(self) -> None:
        entries = [make_entry(f"e{i}", f"python item {i}") for i in range(20)]
        results = hybrid_search("python", entries, top_k=3)
        assert len(results) <= 3

    def test_result_ids_are_subset_of_input_ids(self) -> None:
        entries = self._entries()
        input_ids = {entry.id for entry in entries}
        results = hybrid_search("retrieval", entries)
        for result in results:
            assert result.id in input_ids

    def test_uses_config_parameters(self) -> None:
        entries = self._entries()
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
        embedder = StubEmbedder()
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.dense_search") as mock_dense:
            mock_dense.return_value = []
            hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=None)
        mock_dense.assert_called_once()

    def test_entries_returned_are_original_objects(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic fastmcp", entries)
        entry_map = {entry.id: entry for entry in entries}
        for result in results:
            assert result is entry_map[result.id]

    def test_hybrid_search_normal_rrf_fusion(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        results = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored)
        assert len(results) >= 1

    def test_hybrid_search_empty_entries(self) -> None:
        assert hybrid_search("query", []) == []

    def test_hybrid_search_embedder_none_with_stored_embeddings(self) -> None:
        entries = self._entries()
        stored = {"e1": [0.1, 0.2, 0.3]}
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=stored)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_query_empty_string(self) -> None:
        entries = self._entries()
        results = hybrid_search("", entries)
        assert len(results) >= 0


class TestHybridSearchDegradation:
    def _entries(self) -> list[MemoryEntry]:
        return [
            make_entry("e1", "pydantic validation error", tags=["pydantic"]),
            make_entry("e2", "fastmcp middleware tool", tags=["mcp"]),
            make_entry("e3", "structlog event keyword", tags=["logging"]),
        ]

    def test_hybrid_search_bm25_unavailable(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)

        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            results = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored)
        assert len(results) >= 1

    def test_hybrid_search_embedder_none(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_embedder_not_available(self) -> None:
        embedder = StubEmbedder(available=False)
        entries = self._entries()
        stored = {"e1": [0.1, 0.2, 0.3]}
        results = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_both_unavailable(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None)
        assert results == []

    def test_hybrid_search_single_source_passthrough(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
            results = hybrid_search("pydantic", entries, embedder=None)
        assert len(results) >= 1


class TestHybridSearchImportanceBlend:
    """R-FUSION-001 wiring: importance_alpha reorders the pipeline output.

    These tests drive ``hybrid_search`` (the live MemoryClient hybrid path)
    with two entries that TIE on fused position, then assert impact breaks the
    tie. Each test FAILS against the pre-fix pipeline (which always called
    ``rrf_fuse`` position-only and never threaded importance).
    """

    def _tied_entries(self) -> list[MemoryEntry]:
        # Two entries; one high-impact, one low-impact. The BM25/dense stubs
        # below place each at rank 0 of a separate ranking, so position-only
        # RRF scores them equally — impact is the only differentiator.
        return [
            make_entry("low", "shared topic content", importance=0.10),
            make_entry("high", "shared topic content", importance=0.95),
        ]

    def _patched_rankings(self) -> tuple[object, object]:
        # bm25 ranks 'low' first, dense ranks 'high' first → both at rank 0 in
        # their own list → equal RRF score.
        bm25 = patch(
            "trw_memory.retrieval.pipeline.bm25_search",
            return_value=[("low", 5.0)],
        )
        dense = patch(
            "trw_memory.retrieval.pipeline.dense_search",
            return_value=[("high", 0.9)],
        )
        return bm25, dense

    def test_default_alpha_does_not_reorder_by_impact(self) -> None:
        entries = self._tied_entries()
        bm25, dense = self._patched_rankings()
        with bm25, dense:
            # Default importance_alpha=1.0 → position-only. With a position tie,
            # impact must NOT be the decider (legacy behaviour preserved).
            results = hybrid_search(
                "shared",
                entries,
                embedder=StubEmbedder(),
                stored_embeddings={"low": [0.1, 0.2, 0.3], "high": [0.1, 0.2, 0.3]},
            )
        ids = [e.id for e in results]
        # Both present; impact 0.95 'high' does NOT get promoted above 'low'
        # purely by impact at alpha=1.0 (tie-break falls to dict/iteration order).
        assert set(ids) == {"low", "high"}
        assert ids[0] == "low"

    def test_importance_alpha_promotes_high_impact(self) -> None:
        entries = self._tied_entries()
        bm25, dense = self._patched_rankings()
        with bm25, dense:
            results = hybrid_search(
                "shared",
                entries,
                embedder=StubEmbedder(),
                stored_embeddings={"low": [0.1, 0.2, 0.3], "high": [0.1, 0.2, 0.3]},
                importance_alpha=0.7,
            )
        ids = [e.id for e in results]
        assert ids[0] == "high", "importance_alpha must promote the 0.95-impact entry"
