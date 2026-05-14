"""Tests for trw_memory.security.poisoning — detector behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trw_memory.security.poisoning import (
    AnomalyResult,
    AnomalyType,
    PoisoningDetector,
    quarantine_entry,
)

from ._test_poisoning_support import make_entries_spread, make_entry


class TestCheckFrequency:
    """Tests for frequency spike detection."""

    def test_normal_rate_no_anomalies(self) -> None:
        """Evenly spread entries produce no frequency anomalies."""
        entries = make_entries_spread(20, interval_minutes=120)
        detector = PoisoningDetector(z_threshold=3.0)
        results = detector.check_frequency(entries, window_minutes=60)
        assert results == []

    def test_spike_detected(self) -> None:
        """A burst of entries in one window is flagged."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        entries = [
            make_entry(
                entry_id=f"M-{index:03d}",
                content=f"Normal #{index}",
                created_at=base + timedelta(hours=index),
            )
            for index in range(19)
        ]
        spike_time = base + timedelta(hours=20)
        entries.extend(
            make_entry(
                entry_id=f"M-spike-{index:03d}",
                content=f"Spike #{index}",
                created_at=spike_time + timedelta(minutes=index),
            )
            for index in range(15)
        )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_frequency(entries, window_minutes=60)
        assert len(results) > 0
        assert all(result.anomaly_type == AnomalyType.FREQUENCY_SPIKE for result in results)

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot produce frequency anomalies."""
        detector = PoisoningDetector()
        assert detector.check_frequency([make_entry()]) == []

    def test_empty_entries_no_anomalies(self) -> None:
        """Empty input produces no anomalies."""
        detector = PoisoningDetector()
        assert detector.check_frequency([]) == []


class TestCheckSize:
    """Tests for size anomaly detection."""

    def test_normal_sizes_no_anomalies(self) -> None:
        """Entries of similar size produce no anomalies."""
        entries = [make_entry(entry_id=f"M-{index}", content="A" * 100) for index in range(20)]
        detector = PoisoningDetector(z_threshold=3.0)
        assert detector.check_size(entries) == []

    def test_huge_entry_detected(self) -> None:
        """An entry much larger than average is flagged."""
        entries = [make_entry(entry_id=f"M-{index}", content="A" * 100) for index in range(20)]
        entries.append(make_entry(entry_id="M-giant", content="B" * 10000))

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_size(entries)
        giant_results = [result for result in results if result.entry_id == "M-giant"]
        assert len(results) >= 1
        assert len(giant_results) == 1
        assert giant_results[0].anomaly_type == AnomalyType.SIZE_ANOMALY

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot be anomalous (no distribution)."""
        detector = PoisoningDetector()
        assert detector.check_size([make_entry(content="A" * 100)]) == []

    def test_detail_field_included_in_size(self) -> None:
        """Both content and detail contribute to size calculation."""
        entries = [make_entry(entry_id=f"M-{index}", content="A" * 50, detail="B" * 50) for index in range(20)]
        entries.append(
            make_entry(
                entry_id="M-giant",
                content="A" * 50,
                detail="C" * 10000,
            )
        )
        detector = PoisoningDetector(z_threshold=2.0)
        giant_results = [result for result in detector.check_size(entries) if result.entry_id == "M-giant"]
        assert len(giant_results) == 1


class TestCheckPatterns:
    """Tests for repetitive content pattern detection."""

    def test_varied_content_no_anomalies(self) -> None:
        """Entries with unique content produce no pattern anomalies."""
        entries = [make_entry(entry_id=f"M-{index}", content=f"Unique content #{index}") for index in range(20)]
        detector = PoisoningDetector(z_threshold=2.0)
        assert detector.check_patterns(entries) == []

    def test_identical_content_detected(self) -> None:
        """Many entries with identical content are flagged."""
        entries = [make_entry(entry_id=f"M-unique-{index}", content=f"Unique #{index}") for index in range(5)]
        entries.extend(
            make_entry(
                entry_id=f"M-dup-{index}",
                content="INJECTED POISONED CONTENT",
            )
            for index in range(20)
        )

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.check_patterns(entries)
        flagged_ids = {result.entry_id for result in results}
        assert len(results) > 0
        assert all(result.anomaly_type == AnomalyType.PATTERN_ANOMALY for result in results)
        assert all(entry_id.startswith("M-dup-") for entry_id in flagged_ids)

    def test_single_entry_no_anomalies(self) -> None:
        """A single entry cannot show pattern anomalies."""
        detector = PoisoningDetector()
        assert detector.check_patterns([make_entry()]) == []


class TestAnalyze:
    """Tests for the combined analysis."""

    def test_analyze_runs_all_checks(self) -> None:
        """analyze() combines results from all three check methods."""
        detector = PoisoningDetector(z_threshold=2.0)
        entries = make_entries_spread(10, interval_minutes=120)
        results = detector.analyze(entries)
        assert isinstance(results, list)

    def test_analyze_empty_input(self) -> None:
        """analyze() handles empty input gracefully."""
        detector = PoisoningDetector()
        assert detector.analyze([]) == []

    def test_analyze_returns_anomaly_results(self) -> None:
        """analyze() returns AnomalyResult instances."""
        entries = [make_entry(entry_id=f"M-{index}", content=f"Normal #{index}") for index in range(10)]
        entries.extend(make_entry(entry_id=f"M-poison-{index}", content="POISONED") for index in range(15))

        detector = PoisoningDetector(z_threshold=2.0)
        results = detector.analyze(entries)
        assert all(isinstance(result, AnomalyResult) for result in results)


class TestQuarantineEntry:
    """Tests for quarantine metadata tagging."""

    def test_sets_quarantined_flag(self) -> None:
        """quarantine_entry sets metadata['quarantined'] to 'true'."""
        quarantined = quarantine_entry(make_entry())
        assert quarantined.metadata["quarantined"] == "true"

    def test_sets_quarantined_at_timestamp(self) -> None:
        """quarantine_entry sets a timestamp in metadata."""
        quarantined = quarantine_entry(make_entry())
        assert "quarantined_at" in quarantined.metadata
        assert "T" in quarantined.metadata["quarantined_at"]

    def test_preserves_existing_metadata(self) -> None:
        """Existing metadata keys are preserved."""
        quarantined = quarantine_entry(make_entry(metadata={"custom": "value"}))
        assert quarantined.metadata["custom"] == "value"
        assert quarantined.metadata["quarantined"] == "true"

    def test_returns_new_entry(self) -> None:
        """quarantine_entry returns a new entry, not mutating the original."""
        entry = make_entry()
        quarantined = quarantine_entry(entry)
        assert "quarantined" not in entry.metadata
        assert quarantined.metadata["quarantined"] == "true"

    def test_preserves_all_other_fields(self) -> None:
        """Non-metadata fields remain unchanged."""
        quarantined = quarantine_entry(
            make_entry(
                entry_id="M-preserve",
                content="Important content",
                detail="Detailed info",
            )
        )
        assert quarantined.id == "M-preserve"
        assert quarantined.content == "Important content"
        assert quarantined.detail == "Detailed info"


class TestCustomThreshold:
    """Tests for configurable z-score threshold."""

    def test_lower_threshold_catches_more(self) -> None:
        """A lower z_threshold flags more entries."""
        entries = [make_entry(entry_id=f"M-{index}", content="A" * 100) for index in range(20)]
        entries.append(make_entry(entry_id="M-medium", content="B" * 500))

        strict = PoisoningDetector(z_threshold=3.0)
        relaxed = PoisoningDetector(z_threshold=1.0)

        strict_results = strict.check_size(entries)
        relaxed_results = relaxed.check_size(entries)

        assert len(relaxed_results) >= len(strict_results)
