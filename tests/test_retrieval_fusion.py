"""Reciprocal rank fusion tests."""

from __future__ import annotations

import pytest

from trw_memory.retrieval.fusion import rrf_fuse


class TestRRFFuse:
    def test_empty_rankings_returns_empty(self) -> None:
        assert rrf_fuse([]) == []

    def test_single_ranking_passthrough(self) -> None:
        ranking = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        result = rrf_fuse([ranking])
        result_ids = [entry_id for entry_id, _ in result]
        assert result_ids[0] == "a"
        assert result_ids[1] == "b"
        assert result_ids[2] == "c"

    def test_scores_sorted_descending(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("b", 1.0), ("c", 0.5)]
        result = rrf_fuse([r1, r2])
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_document_appearing_in_both_scores_higher(self) -> None:
        r1 = [("shared", 1.0), ("only_r1", 0.5)]
        r2 = [("shared", 1.0), ("only_r2", 0.5)]
        result = rrf_fuse([r1, r2])
        assert [entry_id for entry_id, _ in result][0] == "shared"

    def test_custom_k_value(self) -> None:
        ranking = [("a", 1.0)]
        result_k60 = rrf_fuse([ranking], k=60)
        result_k10 = rrf_fuse([ranking], k=10)
        assert result_k10[0][1] > result_k60[0][1]

    def test_formula_values(self) -> None:
        ranking = [("a", 99.0)]
        result = rrf_fuse([ranking], k=60)
        assert result[0][1] == pytest.approx(1.0 / 61.0)

    def test_two_rankings_accumulate(self) -> None:
        r1 = [("x", 1.0)]
        r2 = [("x", 1.0)]
        result = rrf_fuse([r1, r2], k=60)
        assert result[0][1] == pytest.approx(2.0 / 61.0)

    def test_all_unique_ids_preserved(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("c", 1.0), ("d", 0.5)]
        result = rrf_fuse([r1, r2])
        assert {entry_id for entry_id, _ in result} == {"a", "b", "c", "d"}
