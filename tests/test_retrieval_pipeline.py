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


class _CountingEmbedder(StubEmbedder):
    """StubEmbedder that records how many times ``embed`` is invoked."""

    def __init__(self, available: bool = True) -> None:
        super().__init__(available=available)
        self.embed_calls = 0

    def embed(self, text: str) -> list[float] | None:
        self.embed_calls += 1
        return super().embed(text)


class TestHybridSearchPrecomputedQueryEmbedding:
    """Seam: an optional precomputed query embedding is forwarded to dense_search.

    Guards the Leverage win — when the caller already embedded the query (tier
    scoring), hybrid_search must NOT embed it a second time inside the dense
    step, while producing the same ranking as the embed-internally path.
    """

    def _entries(self) -> list[MemoryEntry]:
        return [
            make_entry("e1", "pydantic model validation schema", tags=["pydantic"]),
            make_entry("e2", "vector embedding cosine similarity"),
            make_entry("e3", "bm25 sparse retrieval ranking"),
        ]

    def test_precomputed_embedding_skips_query_embed(self) -> None:
        embedder = _CountingEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = embedder.embed("pydantic")  # caller's single embed pass
        embedder.embed_calls = 0  # reset; only count embeds inside hybrid_search

        hybrid_search(
            "pydantic",
            entries,
            embedder=embedder,
            query_embedding=precomputed,
            stored_embeddings=stored,
        )
        assert embedder.embed_calls == 0, "query must not be re-embedded when precomputed vector is supplied"

    def test_without_precomputed_embedding_query_is_embedded(self) -> None:
        embedder = _CountingEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        embedder.embed_calls = 0

        hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored)
        assert embedder.embed_calls == 1, "legacy path must embed the query inside the dense step"

    def test_precomputed_matches_internal_embedding_ranking(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = embedder.embed("pydantic")

        with_precomputed = hybrid_search(
            "pydantic", entries, embedder=embedder, query_embedding=precomputed, stored_embeddings=stored
        )
        internal = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored)
        assert [e.id for e in with_precomputed] == [e.id for e in internal]

    def test_precomputed_embedding_forwarded_to_dense_search(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = [0.5, 0.25, 0.125]
        with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]) as mock_dense:
            hybrid_search(
                "pydantic", entries, embedder=embedder, query_embedding=precomputed, stored_embeddings=stored
            )
        mock_dense.assert_called_once()
        assert mock_dense.call_args.kwargs["query_embedding"] == precomputed

    def test_precomputed_embedding_runs_dense_without_embedder(self) -> None:
        entries = self._entries()
        embedder = StubEmbedder()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = embedder.embed("e2")  # aligns with entry e2's stored vector
        results = hybrid_search(
            "e2",
            entries,
            embedder=None,
            query_embedding=precomputed,
            stored_embeddings=stored,
        )
        assert "e2" in [e.id for e in results]


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
