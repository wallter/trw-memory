"""Tests for trw_memory.security.poisoning — memory poisoning detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.security.poisoning import (
    AnomalyResult,
    AnomalyType,
    PoisoningDetector,
    quarantine_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "M-001",
    content: str = "Normal content",
    detail: str = "",
    created_at: datetime | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        created_at=created_at or datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def _make_entries_spread(
    count: int,
    interval_minutes: int = 120,
    content: str = "Normal content",
) -> list[MemoryEntry]:
    """Create *count* entries spread evenly across time."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        _make_entry(
            entry_id=f"M-{i:03d}",
            content=f"{content} #{i}",
            created_at=base + timedelta(minutes=i * interval_minutes),
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# PoisoningDetector.check_frequency tests
# ---------------------------------------------------------------------------


class TestCheckFrequency:
    """Tests for frequency spike detection."""

    def test_normal_rate_no_anomalies(self) -> None:
        """Evenly spread entries produce no frequency anomalies."""
        entries = _make_entries_spread(20, interval_minutes=120)
        detector = PoisoningDetector(z_threshold=3.0)
        results = detector.check_frequency(entries, window_minutes=60)
        assert results == []

    def test_spike_detected(self) -> None:
        """A burst of entries in one window is flagged."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Create 20 entries: 1 per hour for 19 hours, then 15 in the same
        # window (spike)
        entries: list[MemoryEntry] = []
        for i in range(19):
            entries.append(
                _make_entry(
                    entry_id=f"M-{i:03d}",
                    content=f"Normal #{i}",
                    created_at=base + timedelta(hours=i),
                )
            )
        # Spike: 15 entries in 5 minutes in the same window
        spike_time = base + timedelta(hours=20)
        for j in range(15):
            entries.append(
                _make_entry(
                    entry_id=f"M-spike-{j:03d}",
                    content=f"Spike #{j}",
                    created_at=spike_time + timedelta(minutes=j),
                )
            )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_frequency(entries, window_minutes=60)
        assert len(results) > 0
        assert all(
            r.anomaly_type == AnomalyType.FREQUENCY_SPIKE for r in results
        )

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot produce frequency anomalies."""
        entries = [_make_entry()]
        detector = PoisoningDetector()
        results = detector.check_frequency(entries)
        assert results == []

    def test_empty_entries_no_anomalies(self) -> None:
        """Empty input produces no anomalies."""
        detector = PoisoningDetector()
        results = detector.check_frequency([])
        assert results == []


# ---------------------------------------------------------------------------
# PoisoningDetector.check_size tests
# ---------------------------------------------------------------------------


class TestCheckSize:
    """Tests for size anomaly detection."""

    def test_normal_sizes_no_anomalies(self) -> None:
        """Entries of similar size produce no anomalies."""
        entries = [
            _make_entry(entry_id=f"M-{i}", content="A" * 100)
            for i in range(20)
        ]
        detector = PoisoningDetector(z_threshold=3.0)
        results = detector.check_size(entries)
        assert results == []

    def test_huge_entry_detected(self) -> None:
        """An entry much larger than average is flagged."""
        entries = [
            _make_entry(entry_id=f"M-{i}", content="A" * 100)
            for i in range(20)
        ]
        # Add one giant entry
        entries.append(
            _make_entry(entry_id="M-giant", content="B" * 10000)
        )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_size(entries)
        assert len(results) >= 1
        giant_results = [r for r in results if r.entry_id == "M-giant"]
        assert len(giant_results) == 1
        assert giant_results[0].anomaly_type == AnomalyType.SIZE_ANOMALY

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot be anomalous (no distribution)."""
        entries = [_make_entry(content="A" * 100)]
        detector = PoisoningDetector()
        results = detector.check_size(entries)
        assert results == []

    def test_detail_field_included_in_size(self) -> None:
        """Both content and detail contribute to size calculation."""
        entries = [
            _make_entry(entry_id=f"M-{i}", content="A" * 50, detail="B" * 50)
            for i in range(20)
        ]
        # Giant detail
        entries.append(
            _make_entry(
                entry_id="M-giant",
                content="A" * 50,
                detail="C" * 10000,
            )
        )
        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_size(entries)
        giant_results = [r for r in results if r.entry_id == "M-giant"]
        assert len(giant_results) == 1


# ---------------------------------------------------------------------------
# PoisoningDetector.check_patterns tests
# ---------------------------------------------------------------------------


class TestCheckPatterns:
    """Tests for repetitive content pattern detection."""

    def test_varied_content_no_anomalies(self) -> None:
        """Entries with unique content produce no pattern anomalies."""
        entries = [
            _make_entry(entry_id=f"M-{i}", content=f"Unique content #{i}")
            for i in range(20)
        ]
        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_patterns(entries)
        assert results == []

    def test_identical_content_detected(self) -> None:
        """Many entries with identical content are flagged."""
        entries: list[MemoryEntry] = []
        # 5 unique entries
        for i in range(5):
            entries.append(
                _make_entry(
                    entry_id=f"M-unique-{i}",
                    content=f"Unique #{i}",
                )
            )
        # 20 identical entries (poisoning attempt)
        for j in range(20):
            entries.append(
                _make_entry(
                    entry_id=f"M-dup-{j}",
                    content="INJECTED POISONED CONTENT",
                )
            )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_patterns(entries)
        assert len(results) > 0
        assert all(
            r.anomaly_type == AnomalyType.PATTERN_ANOMALY for r in results
        )
        # All flagged should be the duplicated content
        flagged_ids = {r.entry_id for r in results}
        assert all(eid.startswith("M-dup-") for eid in flagged_ids)

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot show pattern anomalies."""
        entries = [_make_entry()]
        detector = PoisoningDetector()
        results = detector.check_patterns(entries)
        assert results == []


# ---------------------------------------------------------------------------
# PoisoningDetector.analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for the combined analysis."""

    def test_analyze_runs_all_checks(self) -> None:
        """analyze() combines results from all three check methods."""
        detector = PoisoningDetector(z_threshold=2.0)
        # Normal entries: no anomalies
        entries = _make_entries_spread(10, interval_minutes=120)
        results = detector.analyze(entries)
        assert isinstance(results, list)

    def test_analyze_empty_input(self) -> None:
        """analyze() handles empty input gracefully."""
        detector = PoisoningDetector()
        results = detector.analyze([])
        assert results == []

    def test_analyze_returns_anomaly_results(self) -> None:
        """analyze() returns AnomalyResult instances."""
        # Create entries with both size and pattern anomalies
        entries: list[MemoryEntry] = []
        for i in range(10):
            entries.append(
                _make_entry(
                    entry_id=f"M-{i}",
                    content=f"Normal #{i}",
                )
            )
        # Add poisoned duplicates
        for j in range(15):
            entries.append(
                _make_entry(
                    entry_id=f"M-poison-{j}",
                    content="POISONED",
                )
            )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.analyze(entries)
        assert all(isinstance(r, AnomalyResult) for r in results)


# ---------------------------------------------------------------------------
# quarantine_entry tests
# ---------------------------------------------------------------------------


class TestQuarantineEntry:
    """Tests for quarantine metadata tagging."""

    def test_sets_quarantined_flag(self) -> None:
        """quarantine_entry sets metadata['quarantined'] to 'true'."""
        entry = _make_entry()
        quarantined = quarantine_entry(entry)
        assert quarantined.metadata["quarantined"] == "true"

    def test_sets_quarantined_at_timestamp(self) -> None:
        """quarantine_entry sets a timestamp in metadata."""
        entry = _make_entry()
        quarantined = quarantine_entry(entry)
        assert "quarantined_at" in quarantined.metadata
        # Should be an ISO format timestamp
        ts = quarantined.metadata["quarantined_at"]
        assert "T" in ts  # ISO format contains 'T'

    def test_preserves_existing_metadata(self) -> None:
        """Existing metadata keys are preserved."""
        entry = _make_entry(metadata={"custom": "value"})
        quarantined = quarantine_entry(entry)
        assert quarantined.metadata["custom"] == "value"
        assert quarantined.metadata["quarantined"] == "true"

    def test_returns_new_entry(self) -> None:
        """quarantine_entry returns a new entry, not mutating the original."""
        entry = _make_entry()
        quarantined = quarantine_entry(entry)
        assert "quarantined" not in entry.metadata
        assert quarantined.metadata["quarantined"] == "true"

    def test_preserves_all_other_fields(self) -> None:
        """Non-metadata fields remain unchanged."""
        entry = _make_entry(
            entry_id="M-preserve",
            content="Important content",
            detail="Detailed info",
        )
        quarantined = quarantine_entry(entry)
        assert quarantined.id == "M-preserve"
        assert quarantined.content == "Important content"
        assert quarantined.detail == "Detailed info"


# ---------------------------------------------------------------------------
# Custom z_threshold tests
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    """Tests for configurable z-score threshold."""

    def test_lower_threshold_catches_more(self) -> None:
        """A lower z_threshold flags more entries."""
        entries = [
            _make_entry(entry_id=f"M-{i}", content="A" * 100)
            for i in range(20)
        ]
        # Add moderately large entry (not extreme)
        entries.append(
            _make_entry(entry_id="M-medium", content="B" * 500)
        )

        strict = PoisoningDetector(z_threshold=3.0)
        relaxed = PoisoningDetector(z_threshold=1.0)

        strict_results = strict.check_size(entries)
        relaxed_results = relaxed.check_size(entries)

        assert len(relaxed_results) >= len(strict_results)
