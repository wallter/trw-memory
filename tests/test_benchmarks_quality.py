"""Quality metric and quality benchmark tests for the benchmark suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.bench_quality import (
    QualityBenchmark,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from benchmarks.corpus import create_golden_set
from benchmarks.runner import check_thresholds, run_benchmarks


class TestQualityMetrics:
    """Unit tests for IR quality metric functions."""

    def test_precision_at_k_perfect(self) -> None:
        """All retrieved results relevant -> precision = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c", "d"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_at_k_half(self) -> None:
        """Half of retrieved results relevant -> precision = 0.5."""
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_precision_at_k_none(self) -> None:
        """No relevant results in top-K -> precision = 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_precision_at_k_zero_k(self) -> None:
        """k=0 returns 0.0."""
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_recall_at_k_perfect(self) -> None:
        """All relevant docs found -> recall = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_recall_at_k_partial(self) -> None:
        """Only 1 of 2 relevant found -> recall = 0.5."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_recall_at_k_none_relevant(self) -> None:
        """No relevant docs -> recall = 0.0."""
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        """Relevant doc at rank 1 -> RR = 1.0."""
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_reciprocal_rank_third(self) -> None:
        """Relevant doc at rank 3 -> RR = 1/3."""
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_not_found(self) -> None:
        """No relevant doc -> RR = 0.0."""
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_mean_reciprocal_rank(self) -> None:
        """MRR over multiple queries."""
        data = [
            (["a", "b"], {"a"}),
            (["x", "a"], {"a"}),
            (["x", "y", "a"], {"a"}),
        ]
        expected = (1.0 + 0.5 + 1 / 3) / 3
        assert mean_reciprocal_rank(data) == pytest.approx(expected)

    def test_mean_reciprocal_rank_empty(self) -> None:
        """MRR of empty list is 0.0."""
        assert mean_reciprocal_rank([]) == 0.0

    def test_ndcg_at_k_perfect(self) -> None:
        """All relevant docs ranked first -> NDCG = 1.0."""
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_ndcg_at_k_imperfect(self) -> None:
        """Relevant doc at rank 2 -> NDCG < 1.0."""
        retrieved = ["x", "a"]
        relevant = {"a"}
        result = ndcg_at_k(retrieved, relevant, 2)
        assert 0.0 < result < 1.0

    def test_ndcg_at_k_none(self) -> None:
        """No relevant docs -> NDCG = 0.0."""
        retrieved = ["x", "y"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_ndcg_at_k_empty_relevant(self) -> None:
        """Empty relevant set -> NDCG = 0.0."""
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0

    def test_ndcg_at_k_zero_k(self) -> None:
        """k=0 returns 0.0."""
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0


class TestQualityBenchmarkIntegration:
    """Integration test for the full quality benchmark pipeline."""

    def test_quality_benchmark_runs(self, tmp_path: Path) -> None:
        """QualityBenchmark.run() produces expected metrics."""
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)

        bench = QualityBenchmark(
            golden_set_path=golden_path,
            db_dir=tmp_path / "quality",
        )
        results = bench.run()

        assert "precision_at_5" in results
        assert "recall_at_10" in results
        assert "mrr" in results
        assert "ndcg_at_10" in results
        assert "total_queries" in results

        for key in ("precision_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
            assert 0.0 <= results[key] <= 1.0, f"{key} = {results[key]}"

    def test_run_benchmarks_meets_thresholds_with_bundled_fixtures(self, tmp_path: Path) -> None:
        """Bundled benchmark fixtures clear the default threshold gate."""
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)

        report = run_benchmarks(
            sizes=[100],
            golden_set_path=golden_path,
        )

        assert check_thresholds(report) == []
