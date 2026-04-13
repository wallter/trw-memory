"""Throughput benchmarks -- write entries/sec, read queries/sec.

Measures sustained throughput for bulk operations at different corpus sizes.
Unlike latency benchmarks (which measure individual operation timing),
throughput benchmarks measure aggregate operations-per-second including
any overhead from connection handling, commits, and index maintenance.
"""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks._retrieval import search_backend_entries
from benchmarks.corpus import generate_corpus, generate_query_set
from trw_memory.storage.sqlite_backend import SQLiteBackend


class ThroughputBenchmark:
    """Run throughput benchmarks at different corpus sizes.

    Measures:
    - Write throughput: entries stored per second (sequential writes)
    - Read throughput: search queries executed per second

    Attributes:
        db_dir: Directory for temporary benchmark databases.
        results: Collected benchmark results keyed by operation_size.
    """

    def __init__(self, db_dir: Path) -> None:
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict[str, float]] = {}

    def run(
        self, sizes: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """Run all throughput benchmarks across specified corpus sizes.

        Args:
            sizes: List of corpus sizes (default: [100, 1000, 10000]).

        Returns:
            Dict mapping benchmark name to throughput statistics.
        """
        sizes = sizes or [100, 1000, 10000]
        for size in sizes:
            self.results[f"write_{size}"] = self._bench_write_throughput(size)
            self.results[f"read_{size}"] = self._bench_read_throughput(size)
        return self.results

    def _bench_write_throughput(self, size: int) -> dict[str, float]:
        """Benchmark write throughput at given corpus size.

        Measures total time to store `size` entries sequentially,
        then calculates entries-per-second.

        Args:
            size: Number of entries to store.

        Returns:
            Dict with entries_per_sec, total_sec, entry_count.
        """
        db_path = self.db_dir / f"throughput_write_{size}.db"
        backend = SQLiteBackend(db_path=db_path, dim=384)

        try:
            corpus = generate_corpus(size)

            start = time.perf_counter()
            for entry in corpus:
                backend.store(entry)
            total_sec = time.perf_counter() - start

            entries_per_sec = size / total_sec if total_sec > 0 else 0.0

            return {
                "entries_per_sec": round(entries_per_sec, 2),
                "total_sec": round(total_sec, 4),
                "entry_count": float(size),
            }
        finally:
            backend.close()

    def _bench_read_throughput(self, size: int) -> dict[str, float]:
        """Benchmark read (search) throughput at given corpus size.

        Stores the corpus first, then measures how many search queries
        can be executed per second.

        Args:
            size: Number of entries in the corpus.

        Returns:
            Dict with queries_per_sec, total_sec, query_count.
        """
        db_path = self.db_dir / f"throughput_read_{size}.db"
        backend = SQLiteBackend(db_path=db_path, dim=384)

        try:
            corpus = generate_corpus(size)
            for entry in corpus:
                backend.store(entry)

            queries = generate_query_set(corpus, num_queries=200)

            start = time.perf_counter()
            for q in queries:
                query_str = str(q["query"])
                search_backend_entries(
                    backend,
                    query_str,
                    namespace="benchmark",
                    candidate_limit=size,
                    top_k=10,
                )
            total_sec = time.perf_counter() - start

            query_count = len(queries)
            queries_per_sec = (
                query_count / total_sec if total_sec > 0 else 0.0
            )

            return {
                "queries_per_sec": round(queries_per_sec, 2),
                "total_sec": round(total_sec, 4),
                "query_count": float(query_count),
            }
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THROUGHPUT_THRESHOLDS: dict[str, float] = {
    "write_entries_per_sec": 500.0,
    "read_queries_per_sec": 100.0,
}
