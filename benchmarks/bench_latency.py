"""Latency benchmarks for trw-memory operations.

Measures p50/p95/p99 latency for:
- recall (search) at 100/1000/10000 entries
- store individual entries
- store with sequential writes

All timing uses time.perf_counter() for sub-millisecond precision.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from benchmarks.corpus import generate_corpus, generate_query_set
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _percentile(data: list[float], pct: float) -> float:
    """Return the pct-th percentile from sorted data.

    Args:
        data: Unsorted list of float values.
        pct: Percentile as a fraction (0.0-1.0).

    Returns:
        The value at the given percentile.
    """
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


class LatencyBenchmark:
    """Run latency benchmarks at different corpus sizes.

    Attributes:
        db_dir: Directory for temporary benchmark databases.
        results: Collected benchmark results keyed by operation_size.
    """

    def __init__(self, db_dir: Path) -> None:
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict[str, float]] = {}

    def run(self, sizes: list[int] | None = None) -> dict[str, dict[str, float]]:
        """Run all latency benchmarks across specified corpus sizes.

        Args:
            sizes: List of corpus sizes (default: [100, 1000, 10000]).

        Returns:
            Dict mapping benchmark name to latency statistics.
        """
        sizes = sizes or [100, 1000, 10000]
        for size in sizes:
            self.results[f"recall_{size}"] = self._bench_recall(size)
            self.results[f"store_{size}"] = self._bench_store(size)
        return self.results

    def _bench_recall(self, size: int) -> dict[str, float]:
        """Benchmark recall (search) latency at given corpus size.

        Stores the corpus, then measures search latency over 100 queries.

        Args:
            size: Number of entries in the corpus.

        Returns:
            Dict with p50_ms, p95_ms, p99_ms, mean_ms, count.
        """
        db_path = self.db_dir / f"bench_recall_{size}.db"
        backend = SQLiteBackend(db_path=db_path, dim=384)

        try:
            corpus = generate_corpus(size)
            for entry in corpus:
                backend.store(entry)

            queries = generate_query_set(corpus, num_queries=100)
            latencies: list[float] = []

            for q in queries:
                query_str = str(q["query"])
                start = time.perf_counter()
                backend.search(query_str, top_k=10, namespace="benchmark")
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)

            return {
                "p50_ms": round(statistics.median(latencies), 3),
                "p95_ms": round(_percentile(latencies, 0.95), 3),
                "p99_ms": round(_percentile(latencies, 0.99), 3),
                "mean_ms": round(statistics.mean(latencies), 3),
                "count": float(len(latencies)),
            }
        finally:
            backend.close()

    def _bench_store(self, size: int) -> dict[str, float]:
        """Benchmark store latency for individual entry writes.

        Args:
            size: Number of entries to store.

        Returns:
            Dict with p50_ms, p95_ms, p99_ms, mean_ms, total_ms, entries_per_sec.
        """
        db_path = self.db_dir / f"bench_store_{size}.db"
        backend = SQLiteBackend(db_path=db_path, dim=384)

        try:
            corpus = generate_corpus(size)
            latencies: list[float] = []

            for entry in corpus:
                start = time.perf_counter()
                backend.store(entry)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            total_ms = sum(latencies)
            return {
                "p50_ms": round(statistics.median(latencies), 3),
                "p95_ms": round(_percentile(latencies, 0.95), 3),
                "p99_ms": round(_percentile(latencies, 0.99), 3),
                "mean_ms": round(statistics.mean(latencies), 3),
                "total_ms": round(total_ms, 3),
                "entries_per_sec": round(
                    len(latencies) / (total_ms / 1000) if total_ms > 0 else 0, 2
                ),
            }
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Thresholds for regression detection
# ---------------------------------------------------------------------------

LATENCY_THRESHOLDS: dict[str, float] = {
    "recall_100_p95_ms": 100.0,
    "recall_1000_p95_ms": 200.0,
    "recall_10000_p95_ms": 1000.0,
    "store_100_p95_ms": 50.0,
    "store_1000_p95_ms": 50.0,
    "store_10000_p95_ms": 100.0,
}
