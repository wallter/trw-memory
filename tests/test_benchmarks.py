"""Tests for the benchmark suite infrastructure.

Covers:
- Corpus generation (determinism, size, content validity)
- Golden set loading and validation
- Dedup set loading and validation
- Quality metric calculations (precision, recall, MRR, NDCG)
- Latency benchmark execution (small scale)
- Throughput benchmark execution (small scale)
- Memory benchmark execution (small scale)
- Runner report format
- Comparison mode
- Threshold checking
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Benchmark classes
from benchmarks.bench_latency import LatencyBenchmark
from benchmarks.bench_memory import MemoryBenchmark

# Quality metrics
from benchmarks.bench_quality import (
    QualityBenchmark,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from benchmarks.bench_throughput import ThroughputBenchmark

# Corpus / fixture generation
from benchmarks.corpus import (
    create_dedup_set,
    create_golden_set,
    generate_corpus,
    generate_query_set,
)

# Runner
from benchmarks.runner import (
    check_thresholds,
    compare_reports,
    format_report,
    run_benchmarks,
)

# ====================================================================
# Corpus generation tests
# ====================================================================


class TestCorpusGeneration:
    """Tests for the synthetic corpus generator."""

    def test_generate_corpus_exact_size(self) -> None:
        """generate_corpus(100) produces exactly 100 entries."""
        corpus = generate_corpus(100)
        assert len(corpus) == 100

    def test_generate_corpus_deterministic(self) -> None:
        """Same seed produces identical corpus on repeated calls."""
        c1 = generate_corpus(50, seed=42)
        c2 = generate_corpus(50, seed=42)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.content == b.content
            assert a.tags == b.tags
            assert a.importance == b.importance

    def test_generate_corpus_different_seeds(self) -> None:
        """Different seeds produce different corpora."""
        c1 = generate_corpus(50, seed=1)
        c2 = generate_corpus(50, seed=2)
        # IDs should differ since seed is part of hash input
        ids_1 = {e.id for e in c1}
        ids_2 = {e.id for e in c2}
        assert ids_1 != ids_2

    def test_generate_corpus_valid_fields(self) -> None:
        """Generated entries have valid MemoryEntry fields."""
        corpus = generate_corpus(20)
        for entry in corpus:
            assert entry.id.startswith("M-")
            assert len(entry.id) == 10  # "M-" + 8 hex chars
            assert len(entry.content) > 10
            assert len(entry.detail) > 0
            assert len(entry.tags) >= 2
            assert 0.1 <= entry.importance <= 1.0
            assert entry.namespace == "benchmark"
            assert entry.source == "agent"  # "synthetic" coerced to "agent" by source validator
            assert entry.recurrence >= 1

    def test_generate_corpus_large(self) -> None:
        """Corpus generation works at 1000 entries."""
        corpus = generate_corpus(1000)
        assert len(corpus) == 1000
        # All IDs unique
        ids = [e.id for e in corpus]
        assert len(set(ids)) == 1000

    def test_generate_query_set_count(self) -> None:
        """generate_query_set produces the requested number of queries."""
        corpus = generate_corpus(100)
        queries = generate_query_set(corpus, num_queries=30)
        assert len(queries) == 30

    def test_generate_query_set_structure(self) -> None:
        """Each query has the expected keys."""
        corpus = generate_corpus(100)
        queries = generate_query_set(corpus, num_queries=10)
        for q in queries:
            assert "query" in q
            assert "expected_ids" in q
            assert "expected_tags" in q
            assert isinstance(q["query"], str)
            assert isinstance(q["expected_ids"], list)
            assert isinstance(q["expected_tags"], list)

    def test_generate_query_set_deterministic(self) -> None:
        """Same corpus and seed produce identical query sets."""
        corpus = generate_corpus(100, seed=7)
        q1 = generate_query_set(corpus, num_queries=10, seed=7)
        q2 = generate_query_set(corpus, num_queries=10, seed=7)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a["query"] == b["query"]


# ====================================================================
# Golden set and dedup set fixture tests
# ====================================================================


class TestFixtures:
    """Tests for golden set and dedup set fixture generation and loading."""

    def test_golden_set_generation(self, tmp_path: Path) -> None:
        """create_golden_set writes 50 entries to JSON."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "entries" in data
        assert len(data["entries"]) == 50

    def test_golden_set_entry_structure(self, tmp_path: Path) -> None:
        """Each golden entry has required fields."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        data = json.loads(out.read_text())
        for entry in data["entries"]:
            assert "id" in entry
            assert "content" in entry
            assert "tags" in entry
            assert "importance" in entry
            assert "queries" in entry
            assert entry["id"].startswith("golden-")
            assert len(entry["queries"]) >= 2
            for q in entry["queries"]:
                assert "query" in q
                assert "relevant" in q

    def test_golden_set_unique_ids(self, tmp_path: Path) -> None:
        """All golden entry IDs are unique."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        data = json.loads(out.read_text())
        ids = [e["id"] for e in data["entries"]]
        assert len(set(ids)) == 50

    def test_dedup_set_generation(self, tmp_path: Path) -> None:
        """create_dedup_set writes 30 pairs to JSON."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "pairs" in data
        assert len(data["pairs"]) == 30

    def test_dedup_set_pair_structure(self, tmp_path: Path) -> None:
        """Each dedup pair has required fields."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        data = json.loads(out.read_text())
        for pair in data["pairs"]:
            assert "id" in pair
            assert "entry_a" in pair
            assert "entry_b" in pair
            assert "expected_duplicate" in pair
            assert "content" in pair["entry_a"]
            assert "tags" in pair["entry_a"]
            assert "content" in pair["entry_b"]
            assert "tags" in pair["entry_b"]
            assert isinstance(pair["expected_duplicate"], bool)

    def test_dedup_set_has_both_labels(self, tmp_path: Path) -> None:
        """Dedup set contains both true and false duplicate pairs."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        data = json.loads(out.read_text())
        true_count = sum(1 for p in data["pairs"] if p["expected_duplicate"])
        false_count = sum(1 for p in data["pairs"] if not p["expected_duplicate"])
        assert true_count > 0, "Should have true duplicate pairs"
        assert false_count > 0, "Should have false duplicate pairs"
        assert true_count + false_count == 30


# ====================================================================
# Quality metric unit tests
# ====================================================================


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
            (["a", "b"], {"a"}),  # RR = 1.0
            (["x", "a"], {"a"}),  # RR = 0.5
            (["x", "y", "a"], {"a"}),  # RR = 1/3
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


# ====================================================================
# Latency benchmark tests
# ====================================================================


class TestLatencyBenchmark:
    """Integration tests for latency benchmarks at small scale."""

    def test_latency_run_100(self, tmp_path: Path) -> None:
        """LatencyBenchmark.run([100]) completes with expected keys."""
        bench = LatencyBenchmark(db_dir=tmp_path / "latency")
        results = bench.run([100])

        assert "recall_100" in results
        assert "store_100" in results

        recall = results["recall_100"]
        assert "p50_ms" in recall
        assert "p95_ms" in recall
        assert "p99_ms" in recall
        assert "mean_ms" in recall
        assert recall["count"] == 100.0

        store = results["store_100"]
        assert "entries_per_sec" in store
        assert store["entries_per_sec"] > 0


# ====================================================================
# Throughput benchmark tests
# ====================================================================


class TestThroughputBenchmark:
    """Integration tests for throughput benchmarks at small scale."""

    def test_throughput_run_100(self, tmp_path: Path) -> None:
        """ThroughputBenchmark.run([100]) completes with expected keys."""
        bench = ThroughputBenchmark(db_dir=tmp_path / "throughput")
        results = bench.run([100])

        assert "write_100" in results
        assert "read_100" in results

        write = results["write_100"]
        assert "entries_per_sec" in write
        assert write["entries_per_sec"] > 0
        assert write["entry_count"] == 100.0

        read = results["read_100"]
        assert "queries_per_sec" in read
        assert read["queries_per_sec"] > 0


# ====================================================================
# Memory benchmark tests
# ====================================================================


class TestMemoryBenchmark:
    """Integration tests for memory benchmarks at small scale."""

    def test_memory_run_100(self, tmp_path: Path) -> None:
        """MemoryBenchmark.run([100]) completes with expected keys."""
        bench = MemoryBenchmark(db_dir=tmp_path / "memory")
        results = bench.run([100])

        assert "memory_100" in results
        mem = results["memory_100"]
        assert "db_size_bytes" in mem
        assert "db_size_mb" in mem
        assert "per_1000_db_mb" in mem
        assert "entry_count" in mem
        assert mem["entry_count"] == 100.0
        assert mem["db_size_bytes"] > 0


# ====================================================================
# Runner tests
# ====================================================================


class TestRunner:
    """Integration tests for the benchmark runner."""

    def test_run_benchmarks_produces_report(self, tmp_path: Path) -> None:
        """run_benchmarks(sizes=[100]) produces a complete report."""
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)

        report = run_benchmarks(
            output_path=tmp_path / "report.json",
            sizes=[100],
            golden_set_path=golden_path,
        )

        assert "timestamp" in report
        assert "sizes" in report
        assert report["sizes"] == [100]
        assert "suites" in report
        assert "latency" in report["suites"]
        assert "quality" in report["suites"]
        assert "throughput" in report["suites"]
        assert "memory" in report["suites"]

        # Verify file was written
        assert (tmp_path / "report.json").exists()

    def test_run_benchmarks_no_output_file(self, tmp_path: Path) -> None:
        """run_benchmarks works without an output path."""
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)

        report = run_benchmarks(
            sizes=[100],
            golden_set_path=golden_path,
        )
        assert "suites" in report

    def test_compare_reports_deltas(self) -> None:
        """compare_reports detects differences between two reports."""
        previous: dict[str, object] = {
            "timestamp": "2025-01-01T00:00:00Z",
            "suites": {
                "quality": {
                    "precision_at_5": 0.90,
                    "recall_at_10": 0.80,
                    "mrr": 0.75,
                },
            },
        }
        current: dict[str, object] = {
            "timestamp": "2025-01-02T00:00:00Z",
            "suites": {
                "quality": {
                    "precision_at_5": 0.85,
                    "recall_at_10": 0.82,
                    "mrr": 0.70,
                },
            },
        }
        comparison = compare_reports(current, previous)

        assert "deltas" in comparison
        assert "quality" in comparison["deltas"]
        quality_deltas = comparison["deltas"]["quality"]
        assert "precision_at_5" in quality_deltas
        assert quality_deltas["precision_at_5"]["delta"] == pytest.approx(-0.05, abs=0.001)

    def test_compare_reports_regressions(self) -> None:
        """compare_reports flags regressions when metrics drop >10%."""
        previous: dict[str, object] = {
            "timestamp": "2025-01-01T00:00:00Z",
            "suites": {
                "quality": {
                    "precision_at_5": 0.90,
                    "recall_at_10": 0.80,
                },
            },
        }
        current: dict[str, object] = {
            "timestamp": "2025-01-02T00:00:00Z",
            "suites": {
                "quality": {
                    "precision_at_5": 0.70,  # -22% -> regression
                    "recall_at_10": 0.79,  # -1.25% -> no regression
                },
            },
        }
        comparison = compare_reports(current, previous)

        regressions = comparison["regressions"]
        assert len(regressions) >= 1
        regression_metrics = [r["metric"] for r in regressions]
        assert "quality.precision_at_5" in regression_metrics

    def test_check_thresholds_pass(self) -> None:
        """check_thresholds returns empty list when all thresholds pass."""
        report: dict[str, object] = {
            "suites": {
                "latency": {
                    "recall_100": {"p95_ms": 10.0},
                    "recall_1000": {"p95_ms": 50.0},
                },
                "quality": {
                    "precision_at_5": 0.95,
                    "recall_at_10": 0.85,
                    "mrr": 0.80,
                    "ndcg_at_10": 0.75,
                },
                "throughput": {
                    "write_100": {"entries_per_sec": 1000.0},
                    "read_100": {"queries_per_sec": 500.0},
                },
                "memory": {
                    "memory_100": {"per_1000_rss_mb": 5.0, "per_1000_db_mb": 1.0},
                },
            },
        }
        failures = check_thresholds(report)
        assert failures == []

    def test_check_thresholds_violations(self) -> None:
        """check_thresholds detects violations."""
        report: dict[str, object] = {
            "suites": {
                "latency": {
                    "recall_100": {"p95_ms": 999.0},  # threshold: 100
                },
                "quality": {
                    "precision_at_5": 0.10,  # threshold: 0.80
                    "recall_at_10": 0.95,
                    "mrr": 0.95,
                    "ndcg_at_10": 0.95,
                },
                "throughput": {},
                "memory": {},
            },
        }
        failures = check_thresholds(report)
        assert len(failures) >= 2

        violation_suites = {f["suite"] for f in failures}
        assert "latency" in violation_suites
        assert "quality" in violation_suites

    def test_format_report_text(self) -> None:
        """format_report produces readable text output."""
        report: dict[str, object] = {
            "timestamp": "2025-01-01T00:00:00Z",
            "sizes": [100],
            "suites": {
                "latency": {
                    "recall_100": {"p50_ms": 5.0, "p95_ms": 15.0},
                },
                "quality": {
                    "precision_at_5": 0.90,
                },
            },
        }
        text = format_report(report, human_readable=True)
        assert "trw-memory Benchmark Report" in text
        assert "LATENCY" in text
        assert "QUALITY" in text
        assert "precision_at_5" in text

    def test_format_report_json(self) -> None:
        """format_report with human_readable=False produces valid JSON."""
        report: dict[str, object] = {
            "timestamp": "2025-01-01T00:00:00Z",
            "sizes": [100],
            "suites": {},
        }
        text = format_report(report, human_readable=False)
        parsed = json.loads(text)
        assert parsed["timestamp"] == "2025-01-01T00:00:00Z"

    def test_golden_set_bundled_fixture_exists(self) -> None:
        """The bundled golden_set.json fixture exists and has 50 entries."""
        fixture_path = Path(__file__).parent.parent / "benchmarks" / "fixtures" / "golden_set.json"
        assert fixture_path.exists(), f"Missing: {fixture_path}"
        data = json.loads(fixture_path.read_text())
        assert len(data["entries"]) == 50

    def test_dedup_set_bundled_fixture_exists(self) -> None:
        """The bundled dedup_set.json fixture exists and has 30 pairs."""
        fixture_path = Path(__file__).parent.parent / "benchmarks" / "fixtures" / "dedup_set.json"
        assert fixture_path.exists(), f"Missing: {fixture_path}"
        data = json.loads(fixture_path.read_text())
        assert len(data["pairs"]) == 30


# ====================================================================
# Quality benchmark integration test
# ====================================================================


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

        # Metrics should be between 0 and 1
        for key in ("precision_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
            assert 0.0 <= results[key] <= 1.0, f"{key} = {results[key]}"
