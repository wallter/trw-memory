"""Memory poisoning detection using statistical anomaly analysis.

Uses 3-sigma (z-score) thresholds to detect three classes of anomaly:
- **Frequency spikes**: too many entries created in a short time window
- **Size anomalies**: entries whose content is much larger than average
- **Pattern anomalies**: repetitive or formulaic content across entries
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict

from trw_memory.models.memory import MemoryEntry


class AnomalyType(str, Enum):
    """Categories of memory poisoning anomalies."""

    FREQUENCY_SPIKE = "frequency_spike"
    SIZE_ANOMALY = "size_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"


class AnomalyResult(BaseModel):
    """A single anomaly detection result."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    entry_id: str
    anomaly_type: AnomalyType
    z_score: float
    detail: str


class PoisoningDetector:
    """Detect memory poisoning via 3-sigma statistical analysis.

    Args:
        z_threshold: Number of standard deviations above the mean to
            flag as anomalous.  Defaults to ``3.0`` (99.7% confidence).
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self._z_threshold = z_threshold

    def check_frequency(
        self,
        entries: list[MemoryEntry],
        window_minutes: int = 60,
    ) -> list[AnomalyResult]:
        """Detect frequency spikes (too many entries per time window).

        Buckets entries by their ``created_at`` timestamp into windows
        of *window_minutes* and flags windows where the count exceeds
        the z-score threshold.

        Args:
            entries: All entries to analyze.
            window_minutes: Size of each time bucket in minutes.

        Returns:
            Anomaly results for entries in over-populated windows.
        """
        if len(entries) < 2 or window_minutes <= 0:
            return []

        # Bucket entries by time window
        buckets: dict[int, list[MemoryEntry]] = {}
        for entry in entries:
            ts = entry.created_at
            # Bucket key = minutes since epoch, floored to window
            epoch_minutes = int(ts.timestamp() / 60)
            bucket_key = epoch_minutes // window_minutes
            buckets.setdefault(bucket_key, []).append(entry)

        if len(buckets) < 2:
            return []

        # Compute mean and std of bucket sizes
        counts = [len(b) for b in buckets.values()]
        mean, std = _mean_std(counts)
        if std == 0:
            return []

        results: list[AnomalyResult] = []
        for _bucket_key, bucket_entries in buckets.items():
            count = len(bucket_entries)
            z = (count - mean) / std
            if z >= self._z_threshold:
                for entry in bucket_entries:
                    results.append(
                        AnomalyResult(
                            entry_id=entry.id,
                            anomaly_type=AnomalyType.FREQUENCY_SPIKE,
                            z_score=round(z, 2),
                            detail=(
                                f"{count} entries in window "
                                f"(mean={mean:.1f}, std={std:.1f})"
                            ),
                        )
                    )
        return results

    def check_size(
        self,
        entries: list[MemoryEntry],
    ) -> list[AnomalyResult]:
        """Detect size anomalies (entries much larger than average).

        Uses the combined length of ``content`` + ``detail`` as the
        size metric.

        Args:
            entries: All entries to analyze.

        Returns:
            Anomaly results for oversized entries.
        """
        if len(entries) < 2:
            return []

        sizes = [len(e.content) + len(e.detail) for e in entries]
        mean, std = _mean_std(sizes)
        if std == 0:
            return []

        results: list[AnomalyResult] = []
        for entry, size in zip(entries, sizes, strict=False):
            z = (size - mean) / std
            if z >= self._z_threshold:
                results.append(
                    AnomalyResult(
                        entry_id=entry.id,
                        anomaly_type=AnomalyType.SIZE_ANOMALY,
                        z_score=round(z, 2),
                        detail=(
                            f"size={size} chars "
                            f"(mean={mean:.1f}, std={std:.1f})"
                        ),
                    )
                )
        return results

    def check_patterns(
        self,
        entries: list[MemoryEntry],
    ) -> list[AnomalyResult]:
        """Detect repetitive or formulaic content patterns.

        Flags entries whose ``content`` is duplicated more than the
        z-score threshold standard deviations above the mean duplication
        rate.

        Args:
            entries: All entries to analyze.

        Returns:
            Anomaly results for entries with suspicious repetition.
        """
        if len(entries) < 2:
            return []

        # Count occurrences of each content string
        content_counts: Counter[str] = Counter(e.content for e in entries)

        # If all content is unique, no patterns to detect
        counts = list(content_counts.values())
        mean, std = _mean_std(counts)
        if std == 0:
            return []

        # Find content strings with anomalous repetition
        flagged_contents: set[str] = set()
        for content, count in content_counts.items():
            z = (count - mean) / std
            if z >= self._z_threshold:
                flagged_contents.add(content)

        results: list[AnomalyResult] = []
        for entry in entries:
            if entry.content in flagged_contents:
                count = content_counts[entry.content]
                z = (count - mean) / std
                results.append(
                    AnomalyResult(
                        entry_id=entry.id,
                        anomaly_type=AnomalyType.PATTERN_ANOMALY,
                        z_score=round(z, 2),
                        detail=(
                            f"content repeated {count} times "
                            f"(mean={mean:.1f}, std={std:.1f})"
                        ),
                    )
                )
        return results

    def analyze(
        self,
        entries: list[MemoryEntry],
    ) -> list[AnomalyResult]:
        """Run all anomaly checks and return combined results.

        Args:
            entries: All entries to analyze.

        Returns:
            All anomalies found across frequency, size, and pattern checks.
        """
        results: list[AnomalyResult] = []
        results.extend(self.check_frequency(entries))
        results.extend(self.check_size(entries))
        results.extend(self.check_patterns(entries))
        return results


def quarantine_entry(entry: MemoryEntry) -> MemoryEntry:
    """Move *entry* to quarantine by setting metadata flags.

    Sets ``metadata["quarantined"]`` to ``"true"`` and
    ``metadata["quarantined_at"]`` to the current UTC timestamp.

    Args:
        entry: The entry to quarantine.

    Returns:
        A new :class:`MemoryEntry` with quarantine metadata applied.
    """
    now = datetime.now(timezone.utc).isoformat()
    new_metadata = dict(entry.metadata)
    new_metadata["quarantined"] = "true"
    new_metadata["quarantined_at"] = now
    return entry.model_copy(update={"metadata": new_metadata})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_std(values: list[int] | list[float]) -> tuple[float, float]:
    """Compute mean and population standard deviation.

    Args:
        values: Numeric values.

    Returns:
        Tuple of ``(mean, std_dev)``.  Returns ``(0.0, 0.0)`` for
        empty input.
    """
    if not values:
        return (0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return (mean, math.sqrt(variance))
