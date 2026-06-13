"""BM25 retrieval tests."""

from __future__ import annotations

from unittest.mock import patch

from trw_memory.retrieval.bm25 import bm25_search

from ._test_retrieval_support import make_entry


class TestBM25Search:
    def test_returns_top_k_results(self) -> None:
        entries = [
            make_entry("e1", "pydantic validation error handling"),
            make_entry("e2", "fastmcp middleware pattern"),
            make_entry("e3", "structlog event keyword reserved"),
        ]
        results = bm25_search("pydantic validation", entries, top_k=2)
        assert len(results) <= 2
        assert "e1" in [entry_id for entry_id, _ in results]

    def test_empty_entries_returns_empty(self) -> None:
        assert bm25_search("anything", []) == []

    def test_scores_sorted_descending(self) -> None:
        entries = [
            make_entry("a", "apple fruit tree"),
            make_entry("b", "pydantic model validation schema"),
            make_entry("c", "fastmcp middleware tool registration"),
        ]
        results = bm25_search("pydantic model", entries)
        if len(results) >= 2:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_hyphenated_tag_expansion(self) -> None:
        entries = [
            make_entry("tagged", "model configuration", tags=["pydantic-v2"]),
            make_entry("untagged", "unrelated content about dogs"),
        ]
        results = bm25_search("pydantic", entries)
        assert "tagged" in [entry_id for entry_id, _ in results]

    def test_fallback_token_overlap_when_all_zero(self) -> None:
        entries = [make_entry(f"e{i}", "foo bar baz qux") for i in range(5)]
        entries.append(make_entry("target", "foo bar baz qux overlap unique"))
        results = bm25_search("overlap unique", entries)
        assert isinstance(results, list)

    def test_unavailable_returns_empty(self) -> None:
        entries = [make_entry("x", "test content")]
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            results = bm25_search("test", entries)
        assert results == []

    def test_result_ids_match_entry_ids(self) -> None:
        entries = [
            make_entry("alpha", "machine learning training data"),
            make_entry("beta", "neural network weights gradient"),
        ]
        results = bm25_search("machine learning", entries)
        valid_ids = {"alpha", "beta"}
        for entry_id, score in results:
            assert entry_id in valid_ids
            assert score >= 0.0

    def test_top_k_limits_results(self) -> None:
        entries = [make_entry(f"e{i}", f"python code test item {i}") for i in range(20)]
        results = bm25_search("python", entries, top_k=5)
        assert len(results) <= 5

    def test_content_and_detail_both_indexed(self) -> None:
        entries = [
            make_entry("detail_match", "unrelated content", detail="pydantic validation"),
            make_entry("no_match", "something else entirely"),
        ]
        results = bm25_search("pydantic", entries)
        assert "detail_match" in [entry_id for entry_id, _ in results]

    def test_query_hyphen_expansion_matches_split_tag_tokens(self) -> None:
        # Document: tagged "pydantic-v2" → indexed as ["pydantic-v2", "pydantic", "v2"]
        # Query: "pydantic-v2" → must expand to ["pydantic-v2", "pydantic", "v2"]
        # so it matches on the expanded tag tokens too.
        entries = [
            make_entry("tagged", "model configuration", tags=["pydantic-v2"]),
            make_entry("unrelated", "something about javascript and npm"),
        ]
        results = bm25_search("pydantic-v2 schema", entries)
        ids = [entry_id for entry_id, _ in results]
        assert "tagged" in ids
        assert ids[0] == "tagged"  # tagged entry should rank first

