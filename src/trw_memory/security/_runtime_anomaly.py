"""Anomaly-stats helpers for the runtime store path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

Per-namespace rolling-window anomaly statistics + z-score scoring used
by the runtime intake gate.

- ``AnomalyStats`` — frozen dataclass: ``sample_count`` and per-
  dimension ``{mean, std_dev}`` payload.
- ``score_anomaly`` — pull a clean reference set (active +
  non-canary), build rolling stats, then score the candidate via
  ``poisoning.score_entry_anomaly``.
- ``build_anomaly_stats`` — compute mean/std_dev across entry
  length / tag-count / importance dimensions.
- ``series_stats`` — per-series mean + std_dev helper.
- ``write_anomaly_stats`` — persist the rolling-window stats to
  ``anomaly_stats.yaml`` next to the quarantine root.

Extracted as PRD-DIST-245 Phase 3 batch 102.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.poisoning import score_entry_anomaly
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import write_yaml

logger = structlog.get_logger(__name__)

# Rolling-window size used for per-namespace anomaly statistics.
_ROLLING_WINDOW = 100
# Over-fetch buffer: list_entries already returns updated_at DESC, so the
# rolling window is the first _ROLLING_WINDOW clean rows. We fetch 2x the
# window so the small set of in-store filtered rows (system canaries — capped
# at 5 by canary_injection_rate; legacy quarantined rows are kept in a
# SEPARATE quarantine store) can be skipped without dropping below the window.
# This caps per-write deserialization at 2x the window instead of the prior
# fixed 1,000 full MemoryEntry objects (sliced to 100 and the rest discarded).
_REFERENCE_FETCH_LIMIT = _ROLLING_WINDOW * 2


@dataclass(frozen=True)
class AnomalyStats:
    """Rolling anomaly statistics persisted alongside the quarantine store."""

    sample_count: int
    dimensions: dict[str, dict[str, float]]


def score_anomaly(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    config: MemoryConfig,
) -> tuple[tuple[str, float] | None, AnomalyStats]:
    # ACTIVE-only: retire/obsolete/archived entries no longer represent normal
    # write behaviour — including them skews mean/std and corrupts z-scores.
    reference_entries = backend.list_entries(
        namespace=entry.namespace,
        status=MemoryStatus.ACTIVE,
        limit=_REFERENCE_FETCH_LIMIT,
    )
    clean_reference = [
        candidate
        for candidate in reference_entries
        if candidate.metadata.get("quarantined") != "true" and candidate.metadata.get("system_canary") != "true"
    ]
    clean_reference.sort(key=lambda candidate: candidate.updated_at, reverse=True)
    rolling = clean_reference[:_ROLLING_WINDOW]
    stats = build_anomaly_stats(rolling)
    anomaly_reference = [candidate for candidate in rolling if (candidate.content + candidate.detail).strip()]
    anomaly = score_entry_anomaly(entry, anomaly_reference, z_threshold=config.poisoning_z_threshold)
    return anomaly, stats


def build_anomaly_stats(entries: list[MemoryEntry]) -> AnomalyStats:
    if not entries:
        return AnomalyStats(sample_count=0, dimensions={})
    lengths = [float(len((entry.content + entry.detail).encode("utf-8"))) for entry in entries]
    tag_counts = [float(len(entry.tags)) for entry in entries]
    importances = [float(entry.importance) for entry in entries]
    return AnomalyStats(
        sample_count=len(entries),
        dimensions={
            "entry_length": series_stats(lengths),
            "tag_count": series_stats(tag_counts),
            "importance": series_stats(importances),
        },
    )


def series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std_dev": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std_dev": math.sqrt(variance)}


def write_anomaly_stats(config: MemoryConfig, stats: AnomalyStats) -> None:
    stats_path = Path(config.quarantine_path).parent / "anomaly_stats.yaml"
    payload: dict[str, object] = {
        "version": "1.0",
        "updated": datetime.now(timezone.utc).isoformat(),
        "sample_count": stats.sample_count,
        "dimensions": stats.dimensions,
    }
    write_yaml(stats_path, payload)
