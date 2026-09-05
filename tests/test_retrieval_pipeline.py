"""Hybrid retrieval pipeline tests."""

from __future__ import annotations

from unittest.mock import patch

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search

from ._test_retrieval_support import StubEmbedder, make_entry, stored_embeddings_for
from ._test_scope_support import DEFAULT_SCOPE


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
        results = hybrid_search("pydantic", entries, scope=DEFAULT_SCOPE)
        assert all(isinstance(result, MemoryEntry) for result in results)

    def test_bm25_only_no_embedder(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None, scope=DEFAULT_SCOPE)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_dense_only_no_bm25(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            results = hybrid_search("e1", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert len(results) >= 1

    def test_both_sources_produce_results(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        results = hybrid_search(
            "pydantic model", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE
        )
        assert len(results) >= 1

    def test_neither_source_returns_empty(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[]):
            with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
                results = hybrid_search("anything", entries, scope=DEFAULT_SCOPE)
        assert results == []

    def test_empty_entries_returns_empty(self) -> None:
        assert hybrid_search("query", [], scope=DEFAULT_SCOPE) == []

    def test_top_k_limits_results(self) -> None:
        entries = [make_entry(f"e{i}", f"python item {i}") for i in range(20)]
        results = hybrid_search("python", entries, top_k=3, scope=DEFAULT_SCOPE)
        assert len(results) <= 3

    def test_result_ids_are_subset_of_input_ids(self) -> None:
        entries = self._entries()
        input_ids = {entry.id for entry in entries}
        results = hybrid_search("retrieval", entries, scope=DEFAULT_SCOPE)
        for result in results:
            assert result.id in input_ids

    def test_uses_config_parameters(self) -> None:
        entries = self._entries()
        results = hybrid_search(
            "pydantic", entries, bm25_candidates=5, vector_candidates=5, rrf_k=30, top_k=2, scope=DEFAULT_SCOPE
        )
        assert len(results) <= 2

    def test_direct_default_rrf_k_tracks_runtime_config(self) -> None:
        entries = self._entries()
        captured: dict[str, int] = {}

        def fake_rrf_fuse(
            _rankings: list[list[tuple[str, float]]],
            *,
            k: int,
            importances: dict[str, float] | None = None,
            alpha: float = 1.0,
        ) -> list[tuple[str, float]]:
            assert importances is None
            assert alpha == 1.0
            captured["k"] = k
            return [("e1", 1.0)]

        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[("e1", 1.0)]):
            with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[("e2", 1.0)]):
                with patch("trw_memory.retrieval.pipeline.rrf_fuse", side_effect=fake_rrf_fuse):
                    results = hybrid_search(
                        "pydantic", entries, stored_embeddings={"e2": [0.1, 0.2, 0.3]}, scope=DEFAULT_SCOPE
                    )

        assert captured["k"] == MemoryConfig().rrf_k == 5
        assert [entry.id for entry in results] == ["e1"]

    def test_direct_default_recency_halflife_tracks_runtime_config(self) -> None:
        entries = self._entries()
        captured: dict[str, float | None] = {}

        def fake_recency_rank(
            _entries: list[MemoryEntry],
            *,
            halflife_days: float,
            now: object | None = None,
        ) -> list[tuple[str, float]]:
            captured["halflife_days"] = halflife_days
            captured["now"] = now
            return [("e1", 1.0)]

        with patch("trw_memory.retrieval.pipeline.bm25_search", return_value=[("e1", 1.0)]):
            with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
                with patch("trw_memory.retrieval.pipeline.recency_rank", side_effect=fake_recency_rank):
                    results = hybrid_search("recent pydantic", entries, recency_weight=0.3, scope=DEFAULT_SCOPE)

        assert captured["halflife_days"] == MemoryConfig().recall_recency_halflife_days == 14.0
        assert captured["now"] is None
        assert [entry.id for entry in results] == ["e1"]

    def test_dense_skipped_when_no_stored_embeddings(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.dense_search") as mock_dense:
            mock_dense.return_value = []
            result = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=None, scope=DEFAULT_SCOPE)
        mock_dense.assert_called_once()
        assert result

    def test_entries_returned_are_original_objects(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic fastmcp", entries, scope=DEFAULT_SCOPE)
        entry_map = {entry.id: entry for entry in entries}
        for result in results:
            assert result is entry_map[result.id]

    def test_hybrid_search_normal_rrf_fusion(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([entry.id for entry in entries], embedder)
        results = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert len(results) >= 1

    def test_hybrid_search_empty_entries(self) -> None:
        assert hybrid_search("query", [], scope=DEFAULT_SCOPE) == []

    def test_hybrid_search_embedder_none_with_stored_embeddings(self) -> None:
        entries = self._entries()
        stored = {"e1": [0.1, 0.2, 0.3]}
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_query_empty_string(self) -> None:
        entries = self._entries()
        results = hybrid_search("", entries, scope=DEFAULT_SCOPE)
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
            results = hybrid_search(
                "pydantic", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE
            )
        assert len(results) >= 1

    def test_hybrid_search_embedder_none(self) -> None:
        entries = self._entries()
        results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None, scope=DEFAULT_SCOPE)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_embedder_not_available(self) -> None:
        embedder = StubEmbedder(available=False)
        entries = self._entries()
        stored = {"e1": [0.1, 0.2, 0.3]}
        results = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert len(results) >= 1
        assert "e1" in [entry.id for entry in results]

    def test_hybrid_search_both_unavailable(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            results = hybrid_search("pydantic", entries, embedder=None, stored_embeddings=None, scope=DEFAULT_SCOPE)
        assert results == []

    def test_hybrid_search_single_source_passthrough(self) -> None:
        entries = self._entries()
        with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]):
            results = hybrid_search("pydantic", entries, embedder=None, scope=DEFAULT_SCOPE)
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
            scope=DEFAULT_SCOPE,
        )
        assert embedder.embed_calls == 0, "query must not be re-embedded when precomputed vector is supplied"

    def test_without_precomputed_embedding_query_is_embedded(self) -> None:
        embedder = _CountingEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        embedder.embed_calls = 0

        hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert embedder.embed_calls == 1, "legacy path must embed the query inside the dense step"

    def test_precomputed_matches_internal_embedding_ranking(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = embedder.embed("pydantic")

        with_precomputed = hybrid_search(
            "pydantic",
            entries,
            embedder=embedder,
            query_embedding=precomputed,
            stored_embeddings=stored,
            scope=DEFAULT_SCOPE,
        )
        internal = hybrid_search("pydantic", entries, embedder=embedder, stored_embeddings=stored, scope=DEFAULT_SCOPE)
        assert [e.id for e in with_precomputed] == [e.id for e in internal]

    def test_precomputed_embedding_forwarded_to_dense_search(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = [0.5, 0.25, 0.125]
        with patch("trw_memory.retrieval.pipeline.dense_search", return_value=[]) as mock_dense:
            hybrid_search(
                "pydantic",
                entries,
                embedder=embedder,
                query_embedding=precomputed,
                stored_embeddings=stored,
                scope=DEFAULT_SCOPE,
            )
        mock_dense.assert_called_once()
        assert mock_dense.call_args.kwargs["query_embedding"] == precomputed

    def test_precomputed_embedding_runs_dense_without_embedder(self) -> None:
        entries = self._entries()
        embedder = StubEmbedder()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        precomputed = embedder.embed("e2")  # aligns with entry e2's stored vector
        results = hybrid_search(
            "e2", entries, embedder=None, query_embedding=precomputed, stored_embeddings=stored, scope=DEFAULT_SCOPE
        )
        assert "e2" in [e.id for e in results]


class TestHybridSearchFusionModes:
    """fusion_mode parameter wiring."""

    def _entries(self) -> list[MemoryEntry]:
        return [
            make_entry("m1", "python asyncio event loop"),
            make_entry("m2", "structlog structured logging"),
            make_entry("m3", "pydantic data validation"),
        ]

    def test_combmax_fusion_returns_results(self) -> None:
        embedder = StubEmbedder()
        entries = self._entries()
        stored = stored_embeddings_for([e.id for e in entries], embedder)
        results = hybrid_search(
            "asyncio pydantic",
            entries,
            embedder=embedder,
            stored_embeddings=stored,
            fusion_mode="combmax",
            scope=DEFAULT_SCOPE,
        )
        assert len(results) >= 1
        assert all(e.id in {"m1", "m2", "m3"} for e in results)

    def test_unknown_fusion_mode_falls_back_to_rrf(self) -> None:
        entries = self._entries()
        results = hybrid_search("asyncio", entries, fusion_mode="bogus_mode", scope=DEFAULT_SCOPE)
        assert len(results) >= 1

    def test_rerank_true_called_with_mock(self) -> None:
        """rerank=True wires cross_encode_rerank into the pipeline."""
        from unittest.mock import patch

        entries = self._entries()
        sentinel = [entries[0]]

        with patch(
            "trw_memory.retrieval.reranker.cross_encode_rerank",
            return_value=sentinel,
        ) as mock_rerank:
            results = hybrid_search("asyncio", entries, rerank=True, rerank_candidates=5, scope=DEFAULT_SCOPE)

        mock_rerank.assert_called_once()
        assert results[0].id == entries[0].id

    def test_rerank_query_override_passed_to_reranker(self) -> None:
        """rerank_query overrides the search query for the cross-encoder."""
        from unittest.mock import patch

        entries = self._entries()

        with patch(
            "trw_memory.retrieval.reranker.cross_encode_rerank",
            return_value=[entries[0]],
        ) as mock_rerank:
            hybrid_search(
                "asyncio",  # matches entry content so rerank path is reached
                entries,
                rerank=True,
                rerank_query="original full query",
                scope=DEFAULT_SCOPE,
            )

        mock_rerank.assert_called_once()
        call_kwargs = mock_rerank.call_args
        assert call_kwargs[0][0] == "original full query"


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
                scope=DEFAULT_SCOPE,
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
                scope=DEFAULT_SCOPE,
            )
        ids = [e.id for e in results]
        assert ids[0] == "high", "importance_alpha must promote the 0.95-impact entry"
