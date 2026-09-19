"""Anomaly scoring and anomaly-stats persistence for the runtime store path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

- ``score_anomaly`` — score a candidate against its namespace's reference
  window (active, non-quarantined, non-canary; see ``_anomaly_reference``,
  which keeps that window current incrementally) via
  ``poisoning.score_series_anomaly``.
- ``shared_anomaly_reference`` — scope in which every entry scored against
  one namespace reuses a single reference view (``bulk_store``).
- ``write_anomaly_stats`` — persist the rolling-window stats to
  ``anomaly_stats.yaml`` next to the quarantine root, at most once per
  ``_STATS_WRITE_INTERVAL_S`` or ``_STATS_WRITE_MAX_DEFERRED`` calls per file;
  ``flush_anomaly_stats`` writes what is pending (client close, interpreter exit).
- ``AnomalyStats`` / ``build_anomaly_stats`` / ``series_stats`` — re-exported
  from ``_anomaly_reference``.

Extracted as PRD-DIST-245 Phase 3 batch 102.
"""

from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._anomaly_reference import (
    _REFERENCE_FETCH_LIMIT as _REFERENCE_FETCH_LIMIT,
)
from trw_memory.security._anomaly_reference import (
    _ROLLING_WINDOW as _ROLLING_WINDOW,
)
from trw_memory.security._anomaly_reference import (
    AnomalyStats as AnomalyStats,
)
from trw_memory.security._anomaly_reference import (
    ReferenceView,
    reference_view,
)
from trw_memory.security._anomaly_reference import (
    build_anomaly_stats as build_anomaly_stats,
)
from trw_memory.security._anomaly_reference import (
    series_stats as series_stats,
)
from trw_memory.security.poisoning import score_series_anomaly
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import write_yaml

logger = structlog.get_logger(__name__)

# anomaly_stats.yaml is an observability snapshot (nothing in the store path
# reads it back), so a single-row writer need not rewrite it on every store.
_STATS_WRITE_INTERVAL_S = 5.0
_STATS_WRITE_MAX_DEFERRED = 100
_MAX_TRACKED_STATS_FILES = 256


@dataclass
class _SharedReference:
    views: dict[tuple[int, str], ReferenceView] = field(default_factory=dict)


_SHARED_REFERENCE: ContextVar[_SharedReference | None] = ContextVar("trw_memory_shared_anomaly_reference", default=None)


@dataclass
class _StatsFile:
    last_write: float = float("-inf")
    pending: AnomalyStats | None = None
    deferred: int = 0


_STATS_FILES: dict[Path, _StatsFile] = {}
# Held across the write so two threads cannot land an older snapshot last.
_STATS_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


@contextmanager
def shared_anomaly_reference() -> Iterator[None]:
    """Within this block, score every entry of a namespace against ONE reference view.

    Only correct while nothing is written to the scored namespaces inside the
    block: ``bulk_store`` scores every row before it persists any, so each row
    already saw the same reference window.
    """
    token = _SHARED_REFERENCE.set(_SharedReference())
    try:
        yield
    finally:
        _SHARED_REFERENCE.reset(token)


def score_anomaly(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    config: MemoryConfig,
) -> tuple[tuple[str, float] | None, AnomalyStats]:
    shared = _SHARED_REFERENCE.get()
    key = (id(backend), entry.namespace)
    view = shared.views.get(key) if shared is not None else None
    if view is None:
        view = reference_view(entry.namespace, backend)
        if shared is not None:
            shared.views[key] = view
    anomaly = score_series_anomaly(
        entry, lengths=view.lengths, tag_counts=view.tag_counts, z_threshold=config.poisoning_z_threshold
    )
    return anomaly, view.stats


def write_anomaly_stats(config: MemoryConfig, stats: AnomalyStats) -> None:
    """Persist *stats* now, or defer them when this file was written moments ago.

    The first call per file writes immediately. Later calls write when
    ``_STATS_WRITE_INTERVAL_S`` passed since the last write or
    ``_STATS_WRITE_MAX_DEFERRED`` calls were deferred; otherwise the newest
    stats stay pending for the next due call or :func:`flush_anomaly_stats`.
    Writes are atomic (temp file + rename), so a concurrent reader or another
    process's writer never sees a torn file; the last rename wins.
    """
    global _ATEXIT_REGISTERED
    path = Path(config.quarantine_path).parent / "anomaly_stats.yaml"
    with _STATS_LOCK:
        if path not in _STATS_FILES and len(_STATS_FILES) >= _MAX_TRACKED_STATS_FILES:
            for idle in [known for known, tracked in _STATS_FILES.items() if tracked.pending is None]:
                del _STATS_FILES[idle]  # forgetting an idle file only makes its next write immediate
        state = _STATS_FILES.setdefault(path, _StatsFile())
        now = time.monotonic()
        state.deferred += 1
        if now - state.last_write < _STATS_WRITE_INTERVAL_S and state.deferred < _STATS_WRITE_MAX_DEFERRED:
            state.pending = stats
            if not _ATEXIT_REGISTERED:
                atexit.register(_flush_at_exit)
                _ATEXIT_REGISTERED = True
            return
        state.pending, state.deferred, state.last_write = None, 0, now
        _write_stats_file(path, stats)


def flush_anomaly_stats(config: MemoryConfig | None = None) -> None:
    """Write pending anomaly stats: *config*'s file, or every file when ``None``."""
    only = None if config is None else Path(config.quarantine_path).parent / "anomaly_stats.yaml"
    with _STATS_LOCK:
        for path, state in _STATS_FILES.items():
            if state.pending is None or (only is not None and path != only):
                continue
            stats, state.pending, state.deferred, state.last_write = state.pending, None, 0, time.monotonic()
            _write_stats_file(path, stats)


def _flush_at_exit() -> None:
    try:
        flush_anomaly_stats()
    except (
        StorageError
    ):  # trw-fail-silent-allow: interpreter exit; the stats are an observability snapshot the next store rewrites
        logger.warning("anomaly_stats_flush_failed", op="anomaly_stats", outcome="skipped_at_exit", exc_info=True)


def _write_stats_file(path: Path, stats: AnomalyStats) -> None:
    payload: dict[str, object] = {
        "version": "1.0",
        "updated": datetime.now(timezone.utc).isoformat(),
        "sample_count": stats.sample_count,
        "dimensions": stats.dimensions,
    }
    write_yaml(path, payload)
