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
        assert next(entry_id for entry_id, _ in result) == "shared"

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


class TestRRFImportanceBlend:
    """R-FUSION-001: importance-aware fusion changes the output order.

    Each test below would FAIL against the pre-fix position-only ``rrf_fuse``
    (which discarded importance entirely) because the asserted order is driven
    by impact, not rank position.
    """

    def test_default_alpha_is_position_only_legacy(self) -> None:
        # alpha defaults to 1.0 → importances ignored → bit-for-bit legacy.
        ranking = [("a", 1.0), ("b", 0.5)]
        baseline = rrf_fuse([ranking], k=60)
        with_imp_default = rrf_fuse([ranking], k=60, importances={"a": 0.0, "b": 1.0})
        assert with_imp_default == baseline  # importance not applied at alpha=1.0

    def test_importance_breaks_position_tie(self) -> None:
        # Two SEPARATE single-element rankings → both 'lo' and 'hi' land at
        # rank 0 in their own list, so pure-position RRF gives them the SAME
        # score and order is arbitrary. With importance blending, the high-
        # impact entry MUST come first.
        r1 = [("lo", 1.0)]
        r2 = [("hi", 1.0)]
        position_only = rrf_fuse([r1, r2], k=60)
        # Confirm the precondition: equal RRF scores (the tie the fix targets).
        scores = dict(position_only)
        assert scores["lo"] == pytest.approx(scores["hi"])

        blended = rrf_fuse(
            [r1, r2],
            k=60,
            importances={"lo": 0.2, "hi": 0.95},
            alpha=0.7,
        )
        ordered = [eid for eid, _ in blended]
        assert ordered[0] == "hi", "high-impact entry must rank first after blend"

    def test_importance_overturns_position_when_close(self) -> None:
        # 'lo' is ranked slightly ahead by position (rank 0 vs rank 1) but 'hi'
        # is far higher impact. With alpha=0.5 the impact gap (0.95 vs 0.05)
        # must overturn the one-position lead.
        ranking = [("lo", 1.0), ("hi", 0.9)]
        position_only = [eid for eid, _ in rrf_fuse([ranking], k=60)]
        assert position_only == ["lo", "hi"], "precondition: position favours 'lo'"

        blended = rrf_fuse(
            [ranking],
            k=60,
            importances={"lo": 0.05, "hi": 0.95},
            alpha=0.5,
        )
        ordered = [eid for eid, _ in blended]
        assert ordered[0] == "hi", "impact gap must overturn the 1-position lead"

    def test_alpha_one_keeps_position_even_with_importances(self) -> None:
        ranking = [("lo", 1.0), ("hi", 0.9)]
        ordered = [
            eid
            for eid, _ in rrf_fuse(
                [ranking],
                k=60,
                importances={"lo": 0.05, "hi": 0.95},
                alpha=1.0,
            )
        ]
        assert ordered == ["lo", "hi"], "alpha=1.0 must ignore importance"
