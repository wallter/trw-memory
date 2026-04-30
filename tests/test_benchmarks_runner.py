"""Runner and bundled fixture tests for the benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.corpus import create_golden_set
from benchmarks.runner import (
    check_thresholds,
    compare_reports,
    format_report,
    run_benchmarks,
)


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
                    "precision_at_5": 0.70,
                    "recall_at_10": 0.79,
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
                    "recall_100": {"p95_ms": 999.0},
                },
                "quality": {
                    "precision_at_5": 0.10,
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
