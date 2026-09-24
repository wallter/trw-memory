"""Quality metric and quality benchmark tests for the benchmark suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.bench_latency import LATENCY_THRESHOLDS
from benchmarks.bench_memory import MEMORY_THRESHOLDS, _get_rss_kb
from benchmarks.bench_quality import (
    QualityBenchmark,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from benchmarks.bench_throughput import THROUGHPUT_THRESHOLDS
from benchmarks.corpus import create_golden_set
from benchmarks.runner import check_thresholds, run_benchmarks
from tests._timing import assert_budget

# The threshold gate includes wall-clock throughput SLOs (read_queries_per_sec
# >= 100) measured against whatever else the box is doing right now. Verified
# 2026-09-03: a standalone run (no pytest parallelism) failed with
# read_100=88.1 qps while the shared dev box's load average was 24 on 12
# cores (concurrent agent sessions per this repo's shared-tree convention),
# then passed seconds later once load dropped -- not test-order state leakage.
# Retrying keeps the assertion honest: a genuine regression drags every
# attempt over budget, so it still fails; only transient contention is
# absorbed.
_MAX_THRESHOLD_ATTEMPTS = 3


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

    def test_bundled_fixtures_meet_quality_thresholds(self, tmp_path: Path) -> None:
        """Retrieval quality on the bundled golden set clears its thresholds (deterministic, gating)."""
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)
        quality = QualityBenchmark(golden_set_path=golden_path, db_dir=tmp_path / "quality").run()

        assert check_thresholds({"suites": {"quality": quality}}) == []

    # Timing gates measure the shipped runtime, not coverage tracing overhead.
    @pytest.mark.no_cover
    @pytest.mark.requires_local_timing
    def test_bundled_fixtures_meet_latency_and_throughput_budgets(self, tmp_path: Path) -> None:
        """Latency p95 and throughput on the bundled fixtures stay within budget (host-resource).

        See ``_MAX_THRESHOLD_ATTEMPTS`` above for why this retries: a genuine regression drags
        every attempt over budget, transient contention does not.
        """
        golden_path = tmp_path / "golden.json"
        create_golden_set(golden_path)

        suites: dict[str, Any] = {}
        for _attempt in range(_MAX_THRESHOLD_ATTEMPTS):
            suites = run_benchmarks(sizes=[100], golden_set_path=golden_path)["suites"]
            timing = {"latency": suites["latency"], "throughput": suites["throughput"]}
            if not check_thresholds({"suites": timing}):
                break

        for bench, metrics in suites["latency"].items():
            for key, limit in LATENCY_THRESHOLDS.items():
                prefix, _, metric = key.rpartition("_p")
                if prefix == bench and isinstance(metrics, dict) and f"p{metric}" in metrics:
                    assert_budget(f"latency.{bench}.p{metric}", float(metrics[f"p{metric}"]), limit, "ms")
        for bench, metrics in suites["throughput"].items():
            for key, limit in THROUGHPUT_THRESHOLDS.items():
                kind, _, metric = key.partition("_")
                if bench.startswith(kind + "_") and isinstance(metrics, dict) and metric in metrics:
                    assert_budget(f"throughput.{bench}.{metric}", float(metrics[metric]), limit, "ops/s", at_least=True)

    @pytest.mark.requires_local_timing
    def test_rss_per_1000_entries_within_budget_in_a_fresh_process(self) -> None:
        """RSS growth per 1,000 stored entries, measured where earlier tests cannot inflate the peak."""
        measured = _measure_rss_in_subprocess(size=1000)

        assert_budget(
            "rss_per_1000_entries",
            measured["per_1000_rss_mb"],
            MEMORY_THRESHOLDS["rss_per_1000_entries_mb"],
            "MB",
        )


class TestRssSubprocess:
    """PRD-QUAL-141-FR03: the RSS measurement runs in a fresh process, and its failures are visible."""

    def test_subprocess_failure_propagates_with_its_stderr(self) -> None:
        with pytest.raises(RuntimeError, match="boom-from-child"):
            _run_measurement("raise SystemExit('boom-from-child')")

    def test_child_does_not_inherit_the_parents_peak(self) -> None:
        ballast = bytearray(256 * 1024 * 1024)  # raise THIS process's peak RSS by ~256 MB
        ballast[::4096] = b"x" * len(ballast[::4096])
        parent_peak_kb = _get_rss_kb()

        child = _run_measurement(
            "from benchmarks.bench_memory import _get_rss_kb; print(json.dumps({'kb': _get_rss_kb()}))"
        )

        assert child["kb"] < parent_peak_kb - 128 * 1024, (child["kb"], parent_peak_kb)
        del ballast

    def test_linux_reads_the_peak_of_this_program_not_the_one_exec_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Linux ``ru_maxrss`` survives ``execve`` (the kernel carries the replaced
        program's peak into it), so a spawned child reports its PARENT's peak; ``VmHWM``
        belongs to the new address space. Found by the Linux replay leg, 2026-09-23."""
        import benchmarks.bench_memory as bench_memory

        status = tmp_path / "status"
        status.write_text(
            "Name:\tpython\nVmPeak:\t 900000 kB\nVmHWM:\t    9088 kB\nVmRSS:\t 9000 kB\n", encoding="utf-8"
        )
        monkeypatch.setattr(bench_memory.sys, "platform", "linux")
        monkeypatch.setattr(bench_memory, "_PROC_STATUS", status, raising=False)

        assert _get_rss_kb() == 9088


@pytest.mark.parametrize(
    "status",
    [
        None,  # no /proc
        b"Name:\tpython\nVmRSS:\t 9000 kB\n",  # no VmHWM line
        b"VmHWM:\t lots kB\n",  # malformed value
        b"VmHWM:\n",  # value missing
        b"Name:\t\xff\xfeproc\nVmHWM:\t 9088 kB\n",  # non-ASCII process name: still read
    ],
    ids=["no-proc", "no-vmhwm", "malformed", "empty", "non-ascii-name"],
)
def test_linux_peak_rss_falls_back_to_ru_maxrss_only_when_vmhwm_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: bytes | None
) -> None:
    import benchmarks.bench_memory as bench_memory

    path = tmp_path / "status"
    if status is not None:
        path.write_bytes(status)
    monkeypatch.setattr(bench_memory.sys, "platform", "linux")
    monkeypatch.setattr(bench_memory, "_PROC_STATUS", path)
    monkeypatch.setattr(bench_memory._resource, "getrusage", lambda _who: type("U", (), {"ru_maxrss": 4242})())

    expected = 9088 if status is not None and b"9088" in status else 4242
    assert _get_rss_kb() == expected


def test_macos_peak_rss_converts_ru_maxrss_bytes_to_kilobytes(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.bench_memory as bench_memory

    monkeypatch.setattr(bench_memory.sys, "platform", "darwin")
    monkeypatch.setattr(bench_memory._resource, "getrusage", lambda _who: type("U", (), {"ru_maxrss": 5 * 1024})())

    assert _get_rss_kb() == 5


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _run_measurement(snippet: str) -> dict[str, Any]:
    """Run ``snippet`` in a fresh interpreter rooted at the package; return its last stdout line as JSON."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(_PACKAGE_ROOT), str(_PACKAGE_ROOT / "src"), os.environ.get("PYTHONPATH")])
        ),
    }
    proc = subprocess.run(
        [sys.executable, "-c", f"import json\n{snippet}"],
        cwd=_PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"measurement subprocess exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _measure_rss_in_subprocess(size: int) -> dict[str, Any]:
    return _run_measurement(
        "import tempfile, pathlib\n"
        "from benchmarks.bench_memory import MemoryBenchmark\n"
        "with tempfile.TemporaryDirectory() as tmp:\n"
        f"    print(json.dumps(MemoryBenchmark(db_dir=pathlib.Path(tmp)).run([{size}])['memory_{size}']))"
    )
