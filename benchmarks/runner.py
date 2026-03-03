"""Benchmark CLI runner -- runs all benchmarks, produces JSON report, supports comparison.

Usage:
    python -m benchmarks.runner --sizes 100 1000 --format text
    python -m benchmarks.runner -o report.json --sizes 100
    python -m benchmarks.runner --compare previous_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.bench_latency import LATENCY_THRESHOLDS, LatencyBenchmark
from benchmarks.bench_memory import MEMORY_THRESHOLDS, MemoryBenchmark
from benchmarks.bench_quality import QUALITY_THRESHOLDS, QualityBenchmark
from benchmarks.bench_throughput import THROUGHPUT_THRESHOLDS, ThroughputBenchmark


def run_benchmarks(
    output_path: Path | None = None,
    sizes: list[int] | None = None,
    golden_set_path: Path | None = None,
) -> dict[str, Any]:
    """Run all benchmark suites and produce a report.

    Args:
        output_path: If provided, write JSON report to this path.
        sizes: Corpus sizes to benchmark (default: [100, 1000]).
        golden_set_path: Path to golden set fixture. Defaults to
            benchmarks/fixtures/golden_set.json.

    Returns:
        Complete benchmark report as a dict.
    """
    sizes = sizes or [100, 1000]

    if golden_set_path is None:
        golden_set_path = (
            Path(__file__).parent / "fixtures" / "golden_set.json"
        )

    with tempfile.TemporaryDirectory(prefix="trw-bench-") as tmp:
        tmp_dir = Path(tmp)

        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sizes": sizes,
            "suites": {},
        }

        # Latency benchmarks
        latency = LatencyBenchmark(db_dir=tmp_dir / "latency")
        report["suites"]["latency"] = latency.run(sizes)

        # Quality benchmarks (uses golden set, size-independent)
        if golden_set_path.exists():
            quality = QualityBenchmark(
                golden_set_path=golden_set_path,
                db_dir=tmp_dir / "quality",
            )
            report["suites"]["quality"] = quality.run()
        else:
            report["suites"]["quality"] = {"error": "golden_set.json not found"}

        # Throughput benchmarks
        throughput = ThroughputBenchmark(db_dir=tmp_dir / "throughput")
        report["suites"]["throughput"] = throughput.run(sizes)

        # Memory benchmarks
        memory = MemoryBenchmark(db_dir=tmp_dir / "memory")
        report["suites"]["memory"] = memory.run(sizes)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    return report


def compare_reports(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    """Compare two benchmark reports, showing deltas and regressions.

    For each numeric metric present in both reports, computes:
    - absolute delta
    - percentage change
    - regression flag (True if the metric worsened beyond 10%)

    Args:
        current: The new benchmark report.
        previous: The baseline benchmark report.

    Returns:
        Comparison report with per-metric deltas.
    """
    comparison: dict[str, Any] = {
        "current_timestamp": current.get("timestamp", "unknown"),
        "previous_timestamp": previous.get("timestamp", "unknown"),
        "deltas": {},
        "regressions": [],
    }

    # Metrics where LOWER is better
    lower_is_better = {
        "p50_ms", "p95_ms", "p99_ms", "mean_ms", "total_ms",
        "total_sec", "rss_delta_kb", "db_size_bytes", "db_size_mb",
        "per_1000_rss_mb", "per_1000_db_mb",
    }

    for suite_name in current.get("suites", {}):
        cur_suite = current["suites"].get(suite_name, {})
        prev_suite = previous.get("suites", {}).get(suite_name, {})

        if not prev_suite or not isinstance(cur_suite, dict):
            continue

        suite_deltas: dict[str, Any] = {}

        # Handle flat dict (quality) and nested dict (latency, throughput, memory)
        if suite_name == "quality":
            _compare_flat(
                cur_suite, prev_suite, suite_deltas,
                comparison["regressions"], suite_name, lower_is_better,
            )
        else:
            for bench_key in cur_suite:
                if bench_key not in prev_suite:
                    continue
                cur_bench = cur_suite[bench_key]
                prev_bench = prev_suite[bench_key]
                if isinstance(cur_bench, dict) and isinstance(prev_bench, dict):
                    key_deltas: dict[str, Any] = {}
                    _compare_flat(
                        cur_bench, prev_bench, key_deltas,
                        comparison["regressions"],
                        f"{suite_name}.{bench_key}", lower_is_better,
                    )
                    suite_deltas[bench_key] = key_deltas

        comparison["deltas"][suite_name] = suite_deltas

    return comparison


def _compare_flat(
    current: dict[str, Any],
    previous: dict[str, Any],
    deltas: dict[str, Any],
    regressions: list[dict[str, Any]],
    prefix: str,
    lower_is_better: set[str],
) -> None:
    """Compare flat metric dicts and populate deltas/regressions."""
    for key in current:
        if key not in previous:
            continue
        cur_val = current[key]
        prev_val = previous[key]
        if not isinstance(cur_val, (int, float)) or not isinstance(prev_val, (int, float)):
            continue

        delta = float(cur_val) - float(prev_val)
        pct_change = (
            (delta / float(prev_val)) * 100 if float(prev_val) != 0 else 0.0
        )

        # Determine if this is a regression
        is_regression = False
        if key in lower_is_better:
            # Higher is worse
            is_regression = pct_change > 10.0
        else:
            # Higher is better (throughput, precision, recall, etc.)
            is_regression = pct_change < -10.0

        deltas[key] = {
            "current": cur_val,
            "previous": prev_val,
            "delta": round(delta, 4),
            "pct_change": round(pct_change, 2),
            "regression": is_regression,
        }

        if is_regression:
            regressions.append({
                "metric": f"{prefix}.{key}",
                "current": cur_val,
                "previous": prev_val,
                "pct_change": round(pct_change, 2),
            })


def format_report(report: dict[str, Any], human_readable: bool = True) -> str:
    """Format a benchmark report for display.

    Args:
        report: Benchmark report dict (from run_benchmarks or compare_reports).
        human_readable: If True, produce formatted text. If False, produce JSON.

    Returns:
        Formatted string representation.
    """
    if not human_readable:
        return json.dumps(report, indent=2)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("trw-memory Benchmark Report")
    lines.append("=" * 60)

    timestamp = report.get("timestamp", report.get("current_timestamp", ""))
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")

    sizes = report.get("sizes")
    if sizes:
        lines.append(f"Sizes: {sizes}")

    lines.append("")

    suites = report.get("suites", {})
    for suite_name, suite_data in suites.items():
        lines.append(f"--- {suite_name.upper()} ---")

        if isinstance(suite_data, dict):
            if suite_name == "quality":
                # Flat metrics
                for key, val in suite_data.items():
                    lines.append(f"  {key}: {val}")
            else:
                # Nested: each key is a sub-benchmark
                for bench_key, bench_data in suite_data.items():
                    lines.append(f"  [{bench_key}]")
                    if isinstance(bench_data, dict):
                        for key, val in bench_data.items():
                            lines.append(f"    {key}: {val}")
                    else:
                        lines.append(f"    {bench_data}")

        lines.append("")

    # Show comparison deltas if present
    deltas = report.get("deltas", {})
    if deltas:
        lines.append("--- COMPARISON DELTAS ---")
        for suite_name, suite_deltas in deltas.items():
            lines.append(f"  [{suite_name}]")
            if isinstance(suite_deltas, dict):
                for key, val in suite_deltas.items():
                    if isinstance(val, dict) and "delta" in val:
                        lines.append(
                            f"    {key}: {val['current']} "
                            f"(delta: {val['delta']:+.4f}, "
                            f"{val['pct_change']:+.2f}%)"
                        )
                    elif isinstance(val, dict):
                        # Nested sub-benchmark
                        lines.append(f"    [{key}]")
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, dict) and "delta" in sub_val:
                                lines.append(
                                    f"      {sub_key}: {sub_val['current']} "
                                    f"(delta: {sub_val['delta']:+.4f}, "
                                    f"{sub_val['pct_change']:+.2f}%)"
                                )
        lines.append("")

    regressions = report.get("regressions", [])
    if regressions:
        lines.append("--- REGRESSIONS ---")
        for reg in regressions:
            lines.append(
                f"  {reg['metric']}: {reg['previous']} -> {reg['current']} "
                f"({reg['pct_change']:+.2f}%)"
            )
        lines.append("")

    return "\n".join(lines)


def check_thresholds(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Check benchmark results against defined thresholds.

    Checks latency p95 values, quality metrics, throughput rates,
    and memory per-1000-entry ratios against their respective thresholds.

    Args:
        report: Complete benchmark report from run_benchmarks.

    Returns:
        List of threshold violation dicts. Empty list means all passed.
    """
    failures: list[dict[str, Any]] = []
    suites = report.get("suites", {})

    # Check latency thresholds
    latency = suites.get("latency", {})
    for threshold_key, threshold_val in LATENCY_THRESHOLDS.items():
        # Parse: "recall_100_p95_ms" -> bench="recall_100", metric="p95_ms"
        parts = threshold_key.rsplit("_p", 1)
        if len(parts) != 2:
            continue
        bench_key = parts[0]
        metric_key = "p" + parts[1]

        bench_data = latency.get(bench_key, {})
        if isinstance(bench_data, dict):
            actual = bench_data.get(metric_key)
            if actual is not None and float(actual) > threshold_val:
                failures.append({
                    "suite": "latency",
                    "benchmark": bench_key,
                    "metric": metric_key,
                    "actual": actual,
                    "threshold": threshold_val,
                    "violation": "exceeds_maximum",
                })

    # Check quality thresholds (higher is better)
    quality = suites.get("quality", {})
    if isinstance(quality, dict):
        for metric_key, threshold_val in QUALITY_THRESHOLDS.items():
            actual = quality.get(metric_key)
            if actual is not None and float(actual) < threshold_val:
                failures.append({
                    "suite": "quality",
                    "benchmark": "golden_set",
                    "metric": metric_key,
                    "actual": actual,
                    "threshold": threshold_val,
                    "violation": "below_minimum",
                })

    # Check throughput thresholds (higher is better)
    throughput = suites.get("throughput", {})
    for bench_key, bench_data in throughput.items():
        if not isinstance(bench_data, dict):
            continue
        for metric_key, threshold_val in THROUGHPUT_THRESHOLDS.items():
            # Match metric: "write_entries_per_sec" -> "entries_per_sec" in write_* benchmarks
            if metric_key.startswith("write_") and bench_key.startswith("write_"):
                actual_key = metric_key.replace("write_", "")
                actual = bench_data.get(actual_key)
                if actual is not None and float(actual) < threshold_val:
                    failures.append({
                        "suite": "throughput",
                        "benchmark": bench_key,
                        "metric": actual_key,
                        "actual": actual,
                        "threshold": threshold_val,
                        "violation": "below_minimum",
                    })
            elif metric_key.startswith("read_") and bench_key.startswith("read_"):
                actual_key = metric_key.replace("read_", "")
                actual = bench_data.get(actual_key)
                if actual is not None and float(actual) < threshold_val:
                    failures.append({
                        "suite": "throughput",
                        "benchmark": bench_key,
                        "metric": actual_key,
                        "actual": actual,
                        "threshold": threshold_val,
                        "violation": "below_minimum",
                    })

    # Check memory thresholds
    memory = suites.get("memory", {})
    for bench_key, bench_data in memory.items():
        if not isinstance(bench_data, dict):
            continue
        for threshold_key, threshold_val in MEMORY_THRESHOLDS.items():
            if threshold_key == "rss_per_1000_entries_mb":
                actual = bench_data.get("per_1000_rss_mb")
                if actual is not None and float(actual) > threshold_val:
                    failures.append({
                        "suite": "memory",
                        "benchmark": bench_key,
                        "metric": "per_1000_rss_mb",
                        "actual": actual,
                        "threshold": threshold_val,
                        "violation": "exceeds_maximum",
                    })
            elif threshold_key == "db_per_1000_entries_mb":
                actual = bench_data.get("per_1000_db_mb")
                if actual is not None and float(actual) > threshold_val:
                    failures.append({
                        "suite": "memory",
                        "benchmark": bench_key,
                        "metric": "per_1000_db_mb",
                        "actual": actual,
                        "threshold": threshold_val,
                        "violation": "exceeds_maximum",
                    })

    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the benchmark runner.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code: 0 if all thresholds pass, 1 if any violation.
    """
    parser = argparse.ArgumentParser(
        prog="trw-memory-bench",
        description="trw-memory benchmark runner",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Previous report to compare against",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Corpus sizes to benchmark",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args(argv)

    report = run_benchmarks(args.output, args.sizes)

    if args.compare and args.compare.exists():
        prev = json.loads(args.compare.read_text(encoding="utf-8"))
        comparison = compare_reports(report, prev)
        print(format_report(comparison, human_readable=(args.format == "text")))
    else:
        print(format_report(report, human_readable=(args.format == "text")))

    failures = check_thresholds(report)
    if failures:
        print(f"\n{len(failures)} threshold violation(s) found!")
        for f in failures:
            print(
                f"  [{f['suite']}/{f['benchmark']}] {f['metric']}: "
                f"{f['actual']} (threshold: {f['threshold']}, {f['violation']})"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
