"""Memory profiling -- RSS usage, SQLite DB size, index growth.

Measures resource consumption at different corpus sizes to detect
memory leaks and ensure storage efficiency stays within bounds.

Note: RSS measurement uses resource.getrusage on Unix/WSL.
On platforms where this is unavailable, RSS is estimated via
os.path.getsize of the database file as a proxy.
"""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.corpus import generate_corpus
from trw_memory.storage.sqlite_backend import SQLiteBackend

# Attempt to import resource (Unix only, available on WSL)
try:
    import resource as _resource

    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False


def _get_rss_kb() -> int:
    """Return current process RSS in kilobytes.

    Uses resource.getrusage on Unix/WSL. Returns 0 if unavailable.
    """
    if _HAS_RESOURCE:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        return int(usage.ru_maxrss)
    return 0


def _get_file_size_bytes(path: Path) -> int:
    """Return file size in bytes, or 0 if the file does not exist."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


class MemoryBenchmark:
    """Run memory profiling benchmarks at different corpus sizes.

    Measures:
    - RSS (resident set size) before and after loading entries
    - SQLite database file size on disk
    - Per-1000-entry resource ratios for regression detection

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
        """Run all memory benchmarks across specified corpus sizes.

        Args:
            sizes: List of corpus sizes (default: [100, 1000, 10000]).

        Returns:
            Dict mapping benchmark name to memory statistics.
        """
        sizes = sizes or [100, 1000, 10000]
        for size in sizes:
            self.results[f"memory_{size}"] = self._measure(size)
        return self.results

    def _measure(self, size: int) -> dict[str, float]:
        """Measure memory consumption for storing `size` entries.

        Args:
            size: Number of entries to store.

        Returns:
            Dict with rss_before_kb, rss_after_kb, rss_delta_kb,
            db_size_bytes, db_size_mb, per_1000_rss_mb, per_1000_db_mb.
        """
        db_path = self.db_dir / f"memory_bench_{size}.db"

        rss_before = _get_rss_kb()

        backend = SQLiteBackend(db_path=db_path, dim=384)
        try:
            corpus = generate_corpus(size)
            for entry in corpus:
                backend.store(entry)
        finally:
            backend.close()

        rss_after = _get_rss_kb()
        rss_delta = rss_after - rss_before
        db_size_bytes = _get_file_size_bytes(db_path)
        db_size_mb = db_size_bytes / (1024 * 1024)

        # Per-1000-entry ratios
        scale = max(size / 1000, 0.001)
        per_1000_rss_mb = (rss_delta / 1024) / scale if rss_delta > 0 else 0.0
        per_1000_db_mb = db_size_mb / scale

        return {
            "rss_before_kb": float(rss_before),
            "rss_after_kb": float(rss_after),
            "rss_delta_kb": float(rss_delta),
            "db_size_bytes": float(db_size_bytes),
            "db_size_mb": round(db_size_mb, 4),
            "per_1000_rss_mb": round(per_1000_rss_mb, 4),
            "per_1000_db_mb": round(per_1000_db_mb, 4),
            "entry_count": float(size),
        }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MEMORY_THRESHOLDS: dict[str, float] = {
    "rss_per_1000_entries_mb": 50.0,
    "db_per_1000_entries_mb": 10.0,
}
