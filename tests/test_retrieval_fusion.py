"""Reciprocal rank fusion tests."""

from __future__ import annotations

import pytest

from trw_memory.retrieval.fusion import blend_recency, combmax_fuse, rrf_fuse


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

    def test_k_invalid_logs_warning_and_falls_back(self) -> None:
        """rrf_fuse with k<1 logs a warning and uses k=60 as fallback."""
        import structlog.testing

        ranking = [("a", 1.0), ("b", 0.5)]
        with structlog.testing.capture_logs() as logs:
            result = rrf_fuse([ranking], k=0)
        assert len(result) >= 1
        warning_events = [log["event"] for log in logs if log.get("log_level") == "warning"]
        assert "rrf_k_invalid" in warning_events


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


class TestCombmaxFuse:
    """CombMAX fusion — max reciprocal-rank per document across all rankings."""

    def test_empty_rankings_returns_empty(self) -> None:
        assert combmax_fuse([]) == []

    def test_single_ranking_passthrough(self) -> None:
        ranking = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        result = combmax_fuse([ranking])
        result_ids = [entry_id for entry_id, _ in result]
        assert result_ids == ["a", "b", "c"]

    def test_scores_sorted_descending(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("b", 1.0), ("c", 0.5)]
        result = combmax_fuse([r1, r2])
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_unique_ids_preserved(self) -> None:
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("c", 1.0), ("d", 0.5)]
        result = combmax_fuse([r1, r2])
        assert {entry_id for entry_id, _ in result} == {"a", "b", "c", "d"}

    def test_combmax_uses_max_not_sum(self) -> None:
        # 'x' appears in both rankings at rank 0.  RRF-sum would double the
        # score; CombMAX should NOT — score must equal a single-ranking value.
        r1 = [("x", 1.0)]
        r2 = [("x", 1.0)]
        result = combmax_fuse([r1, r2])
        assert len(result) == 1
        expected_single = 1.0 / (60 + 1)
        assert result[0][1] == pytest.approx(expected_single)

    def test_formula_values(self) -> None:
        ranking = [("a", 99.0)]
        result = combmax_fuse([ranking])
        assert result[0][1] == pytest.approx(1.0 / 61.0)

    def test_hard_tail_recall_pattern(self) -> None:
        # Both 'lo' and 'hi' are rank-0 champions in their respective single-
        # element lists.  CombMAX assigns each the same peak score.  RRF-sum
        # would also assign equal scores here, but CombMAX must not double-
        # count a document that appears in both (tested separately).
        r1 = [("only_r1", 1.0), ("shared", 0.5)]
        r2 = [("only_r2", 1.0), ("shared", 0.5)]
        result = combmax_fuse([r1, r2])
        ids = [entry_id for entry_id, _ in result]
        # 'only_r1' and 'only_r2' are rank-0 in each list → max score == rank-0 score
        # 'shared' is rank-1 in both → max score == rank-1 score (lower)
        assert ids[0] in {"only_r1", "only_r2"}
        assert ids[1] in {"only_r1", "only_r2"}
        assert ids[2] == "shared"

    def test_document_in_multiple_lists_keeps_best_rank(self) -> None:
        # 'a' is rank-0 in r1 and rank-1 in r2.  CombMAX must keep rank-0 score.
        r1 = [("a", 1.0), ("b", 0.5)]
        r2 = [("c", 1.0), ("a", 0.5)]
        scores = dict(combmax_fuse([r1, r2]))
        rank0_score = 1.0 / (60 + 1)
        assert scores["a"] == pytest.approx(rank0_score)

    def test_k_invalid_logs_warning_and_falls_back(self) -> None:
        """combmax_fuse with k<1 logs a warning and uses k=60 as fallback."""
        import structlog.testing

        ranking = [("a", 1.0), ("b", 0.5)]
        with structlog.testing.capture_logs() as logs:
            result = combmax_fuse([ranking], k=0)
        assert len(result) >= 1
        warning_events = [log["event"] for log in logs if log.get("log_level") == "warning"]
        assert "combmax_k_invalid" in warning_events


class TestBlendRecency:
    """blend_recency — linear interpolation of relevance and recency scores.

    final = (1 - w) * normalised_relevance + w * recency_score
    """

    def test_zero_weight_returns_fused_unchanged(self) -> None:
        fused = [("a", 1.0), ("b", 0.5)]
        recency = [("b", 1.0), ("a", 0.2)]
        result = blend_recency(fused, recency_results=recency, recency_weight=0.0)
        assert result is fused  # unchanged (same object, no blending applied)

    def test_empty_recency_returns_fused_unchanged(self) -> None:
        fused = [("a", 1.0), ("b", 0.5)]
        result = blend_recency(fused, recency_results=[], recency_weight=0.5)
        assert result is fused

    def test_blend_moves_recent_entries_up(self) -> None:
        # 'a' leads on relevance, but 'b' is the most recent. A high recency
        # weight must promote 'b' above 'a'.
        fused = [("a", 1.0), ("b", 0.5)]
        recency = [("b", 1.0), ("a", 0.0)]
        relevance_only = [eid for eid, _ in fused]
        assert relevance_only == ["a", "b"], "precondition: relevance favours 'a'"

        blended = blend_recency(fused, recency_results=recency, recency_weight=0.9)
        ordered = [eid for eid, _ in blended]
        assert ordered[0] == "b", "high recency weight must promote the recent entry"

    def test_blend_appends_recency_only_entries(self) -> None:
        # 'c' appears only in the recency list — it must still surface in the
        # blended output (appended), not be dropped.
        fused = [("a", 1.0), ("b", 0.5)]
        recency = [("c", 1.0), ("a", 0.3)]
        blended = blend_recency(fused, recency_results=recency, recency_weight=0.5)
        ids = {eid for eid, _ in blended}
        assert ids == {"a", "b", "c"}, "recency-only entry must be appended"

    def test_full_recency_weight_sorts_by_recency(self) -> None:
        # weight=1.0 → relevance contributes nothing → order follows recency.
        fused = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        recency = [("c", 1.0), ("b", 0.6), ("a", 0.2)]
        blended = blend_recency(fused, recency_results=recency, recency_weight=1.0)
        ordered = [eid for eid, _ in blended]
        assert ordered == ["c", "b", "a"]

    def test_tie_break_by_original_relevance_rank(self) -> None:
        # 'a' and 'b' get identical blended scores: equal normalised relevance
        # (both share rel_max via... constructed below) and equal recency. The
        # tie must break by original relevance rank ('a' before 'b').
        fused = [("a", 1.0), ("b", 1.0)]
        recency = [("a", 0.5), ("b", 0.5)]
        blended = blend_recency(fused, recency_results=recency, recency_weight=0.5)
        scores = dict(blended)
        assert scores["a"] == pytest.approx(scores["b"]), "precondition: scores tie"
        ordered = [eid for eid, _ in blended]
        assert ordered == ["a", "b"], "tie must break by original relevance rank"


class TestHybridSearchFusionMode:
    """Verify hybrid_search respects the fusion_mode parameter."""

    def test_combmax_mode_returns_entries(self) -> None:
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.retrieval.pipeline import hybrid_search

        entries = [MemoryEntry(id=f"e{i}", content=f"entry {i}", namespace="default") for i in range(3)]
        result = hybrid_search("entry", entries, fusion_mode="combmax")
        # Must return MemoryEntry objects, not crash
        assert all(isinstance(e, MemoryEntry) for e in result)

    def test_unknown_fusion_mode_emits_warning_and_falls_back(self) -> None:
        # Supply synthetic rankings directly to trigger the fallback branch.
        # hybrid_search calls rrf_fuse/combmax_fuse only when rankings is
        # non-empty; we monkeypatch bm25_search to return a non-empty list so
        # the branch is reached.
        from unittest.mock import patch

        import structlog.testing

        from trw_memory.models.memory import MemoryEntry
        from trw_memory.retrieval.pipeline import hybrid_search

        entries = [MemoryEntry(id=f"e{i}", content=f"entry {i}", namespace="default") for i in range(3)]
        fake_bm25 = [("e0", 1.0), ("e1", 0.5)]
        with (
            patch("trw_memory.retrieval.pipeline.bm25_search", return_value=fake_bm25),
            structlog.testing.capture_logs() as logs,
        ):
            result = hybrid_search("entry", entries, fusion_mode="bogus_mode")
        warning_events = [log["event"] for log in logs if log.get("log_level") == "warning"]
        assert "hybrid_search_unknown_fusion_mode" in warning_events
        assert all(isinstance(e, MemoryEntry) for e in result)
