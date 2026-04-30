"""Performance benchmark smoke tests for latency, throughput, and memory."""

from __future__ import annotations

from pathlib import Path

from benchmarks.bench_latency import LatencyBenchmark
from benchmarks.bench_memory import MemoryBenchmark
from benchmarks.bench_throughput import ThroughputBenchmark


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
