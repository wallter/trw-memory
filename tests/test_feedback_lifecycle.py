"""Tests for PRD-CORE-132: Impact-Driven Learning Lifecycle & Feedback Mechanism.

Covers:
- FR01: recall_count, helpful_count, unhelpful_count fields on MemoryEntry
- FR02: record_recall_access increments recall_count
- FR03: Feedback parameter in tools (tested in trw-mcp)
- FR04: feedback_decay_score dynamic scoring algorithm
- Schema migration: ALTER TABLE ADD COLUMN with defaults
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.lifecycle._recall import record_recall_access
from trw_memory.lifecycle.scoring import entry_utility, recall_frequency_decay_score
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# FR01: Enhanced Memory Schema
# ---------------------------------------------------------------------------


class TestMemoryEntryFeedbackFields:
    """Verify new fields exist with correct defaults."""

    def test_defaults_are_zero(self) -> None:
        entry = MemoryEntry(id="test-1", content="test entry")
        assert entry.recall_count == 0

    def test_fields_accept_values(self) -> None:
        entry = MemoryEntry(
            id="test-2",
            content="test entry",
            recall_count=10,
        )
        assert entry.recall_count == 10

    def test_to_dict_includes_feedback_fields(self) -> None:
        entry = MemoryEntry(id="test-3", content="test", recall_count=5)
        d = entry.to_dict()
        assert d["recall_count"] == 5

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryEntry(id="test-4", content="test", recall_count=-1)


# ---------------------------------------------------------------------------
# Schema migration: SQLite column addition
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """Verify SQLite columns are created and data round-trips correctly."""

    def test_columns_exist_in_fresh_db(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="m-1", content="test migration")
            backend.store(entry)
            loaded = backend.get("m-1", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 0
        finally:
            backend.close()

    def test_recall_count_round_trip(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="m-2", content="test round trip")
            backend.store(entry)
            backend.update("m-2", recall_count=15, namespace="default")
            loaded = backend.get("m-2", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 15
        finally:
            backend.close()

    def test_migration_preserves_existing_data(self, tmp_path: Path) -> None:
        """Simulate upgrading an existing DB: data should survive migration."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        try:
            entry = MemoryEntry(id="m-3", content="original content", importance=0.8)
            backend.store(entry)
        finally:
            backend.close()

        # Re-open — ensure_schema runs migration again (idempotent)
        backend2 = SQLiteBackend(db_path)
        try:
            loaded = backend2.get("m-3", namespace="default")
            assert loaded is not None
            assert loaded.content == "original content"
            assert loaded.importance == 0.8
            assert loaded.recall_count == 0
        finally:
            backend2.close()


# ---------------------------------------------------------------------------
# FR02: Recall tracking — record_recall_access increments recall_count
# ---------------------------------------------------------------------------


class TestRecallTracking:
    """Verify record_recall_access increments recall_count."""

    def test_recall_access_increments_recall_count(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="r-1", content="recall test")
            backend.store(entry)

            record_recall_access(backend, ["r-1"], namespace="default")

            loaded = backend.get("r-1", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 1
            assert loaded.access_count == 1
        finally:
            backend.close()

    def test_multiple_recalls_increment(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="r-2", content="multi recall test")
            backend.store(entry)

            for _ in range(5):
                record_recall_access(backend, ["r-2"], namespace="default")

            loaded = backend.get("r-2", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 5
            assert loaded.access_count == 5
        finally:
            backend.close()

    def test_recall_dedup_within_single_call(self) -> None:
        """Duplicate IDs in a single call should only count once."""
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="r-3", content="dedup test")
            backend.store(entry)

            record_recall_access(backend, ["r-3", "r-3", "r-3"], namespace="default")

            loaded = backend.get("r-3", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 1
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# FR04: Dynamic scoring — recall_frequency_decay_score
# (PRD-CORE-293 FR02 renamed ``feedback_decay_score`` and dropped its dead
# ``helpful_count`` parameter — see module docstring in scoring.py)
# ---------------------------------------------------------------------------


class TestRecallFrequencyDecayScore:
    """Test the recall-frequency decay formula.

    Formula: importance * max(min_factor, 0.95 ** recall_count)

    PRD-CORE-244 FR11 added the ``min_factor`` floor (default 0.5) on the decay
    FACTOR — not the score — so a heavily-recalled entry decays no further than
    half its importance rather than toward zero. ``min_factor=0.0`` restores
    the pre-floor curve exactly.
    """

    def test_zero_recalls_returns_importance(self) -> None:
        """No recalls means no decay."""
        score = recall_frequency_decay_score(importance=0.8, recall_count=0)
        assert score == pytest.approx(0.8)

    def test_ten_recalls_decays(self) -> None:
        """10 recalls: moderate decay."""
        score = recall_frequency_decay_score(importance=0.8, recall_count=10)
        expected = 0.8 * (0.95**10)  # 0.8 * ~0.5987 = ~0.479
        assert score == pytest.approx(expected, rel=1e-4)

    def test_100_recalls_hits_floor(self) -> None:
        """100 recalls: the raw curve would be near-zero, but the default 0.5
        floor caps the decay FACTOR (PRD-CORE-244 FR11) — the score never
        drops below ``importance * min_factor``.
        """
        raw_factor = 0.95**100
        assert raw_factor < 0.5, "fixture must exercise the floor, not the raw curve"

        score = recall_frequency_decay_score(importance=0.8, recall_count=100)
        expected_floored = 0.8 * 0.5  # importance * default min_factor
        assert score == pytest.approx(expected_floored, rel=1e-6)
        assert score >= expected_floored

    def test_1000_recalls_floored_vs_unfloored(self) -> None:
        """1000 recalls: default floor holds the score at ``importance *
        min_factor``; ``min_factor=0.0`` restores the old, unbounded-below
        curve (near zero) exactly.
        """
        floored = recall_frequency_decay_score(importance=0.8, recall_count=1000)
        assert floored == pytest.approx(0.8 * 0.5, rel=1e-6)

        unfloored = recall_frequency_decay_score(importance=0.8, recall_count=1000, min_factor=0.0)
        assert unfloored == pytest.approx(0.0, abs=1e-10)
        assert unfloored < floored

    def test_clamped_to_01(self) -> None:
        """Score should never exceed 1.0 or go below 0.0."""
        score = recall_frequency_decay_score(importance=1.0, recall_count=0)
        assert 0.0 <= score <= 1.0
        score2 = recall_frequency_decay_score(importance=1.0, recall_count=10000)
        assert 0.0 <= score2 <= 1.0


class TestFeedbackInEntryUtility:
    """Verify recall-frequency decay is wired into entry_utility."""

    def test_entry_utility_with_recall_has_lower_utility(self) -> None:
        """Entry with high recall_count should have lower utility (recall-frequency decay)."""
        base_entry = {
            "importance": 0.8,
            "q_value": 0.5,
            "q_observations": 0,
            "recurrence": 1,
            "access_count": 0,
            "source": "agent",
            "recall_count": 0,
        }
        high_recall_entry = {**base_entry, "recall_count": 50}

        utility_base = entry_utility(base_entry)
        utility_high_recall = entry_utility(high_recall_entry)

        # High recall should have lower utility than no recall.
        assert utility_high_recall < utility_base
