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
from trw_memory.lifecycle.scoring import entry_utility, feedback_decay_score
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
        assert entry.helpful_count == 0
        assert entry.unhelpful_count == 0

    def test_fields_accept_values(self) -> None:
        entry = MemoryEntry(
            id="test-2",
            content="test entry",
            recall_count=10,
            helpful_count=3,
            unhelpful_count=1,
        )
        assert entry.recall_count == 10
        assert entry.helpful_count == 3
        assert entry.unhelpful_count == 1

    def test_to_dict_includes_feedback_fields(self) -> None:
        entry = MemoryEntry(id="test-3", content="test", recall_count=5, helpful_count=2, unhelpful_count=1)
        d = entry.to_dict()
        assert d["recall_count"] == 5
        assert d["helpful_count"] == 2
        assert d["unhelpful_count"] == 1

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryEntry(id="test-4", content="test", recall_count=-1)
        with pytest.raises(ValueError):
            MemoryEntry(id="test-5", content="test", helpful_count=-1)
        with pytest.raises(ValueError):
            MemoryEntry(id="test-6", content="test", unhelpful_count=-1)


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
            assert loaded.helpful_count == 0
            assert loaded.unhelpful_count == 0
        finally:
            backend.close()

    def test_feedback_counters_round_trip(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            entry = MemoryEntry(id="m-2", content="test round trip")
            backend.store(entry)
            backend.update("m-2", recall_count=15, helpful_count=5, unhelpful_count=2, namespace="default")
            loaded = backend.get("m-2", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 15
            assert loaded.helpful_count == 5
            assert loaded.unhelpful_count == 2
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

            record_recall_access(backend, ["r-1"])

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
                record_recall_access(backend, ["r-2"])

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

            record_recall_access(backend, ["r-3", "r-3", "r-3"])

            loaded = backend.get("r-3", namespace="default")
            assert loaded is not None
            assert loaded.recall_count == 1
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# FR04: Dynamic scoring — feedback_decay_score
# ---------------------------------------------------------------------------


class TestFeedbackDecayScore:
    """Test the feedback-aware decay formula.

    Formula: importance * max(min_factor, 0.95 ** (recall_count / max(1, helpful_count)))

    PRD-CORE-244 FR11 added the ``min_factor`` floor (default 0.5) on the decay
    FACTOR — not the score — so an unrated, heavily-recalled entry decays no
    further than half its importance rather than toward zero. ``min_factor=0.0``
    restores the pre-floor curve exactly.
    """

    def test_zero_recalls_returns_importance(self) -> None:
        """No recalls means no decay."""
        score = feedback_decay_score(importance=0.8, recall_count=0, helpful_count=0)
        assert score == pytest.approx(0.8)

    def test_recalls_with_no_helpful_decays(self) -> None:
        """10 recalls, 0 helpful: moderate decay."""
        score = feedback_decay_score(importance=0.8, recall_count=10, helpful_count=0)
        expected = 0.8 * (0.95**10)  # 0.8 * ~0.5987 = ~0.479
        assert score == pytest.approx(expected, rel=1e-4)

    def test_100_recalls_no_helpful_hits_floor(self) -> None:
        """100 recalls, 0 helpful: the raw curve would be near-zero, but the
        default 0.5 floor caps the decay FACTOR (PRD-CORE-244 FR11) — the
        score never drops below ``importance * min_factor``.
        """
        raw_factor = 0.95**100
        assert raw_factor < 0.5, "fixture must exercise the floor, not the raw curve"

        score = feedback_decay_score(importance=0.8, recall_count=100, helpful_count=0)
        expected_floored = 0.8 * 0.5  # importance * default min_factor
        assert score == pytest.approx(expected_floored, rel=1e-6)
        assert score >= expected_floored

    def test_1000_recalls_no_helpful_floored_vs_unfloored(self) -> None:
        """1000 recalls, 0 helpful: default floor holds the score at
        ``importance * min_factor``; ``min_factor=0.0`` restores the old,
        unbounded-below curve (near zero) exactly.
        """
        floored = feedback_decay_score(importance=0.8, recall_count=1000, helpful_count=0)
        assert floored == pytest.approx(0.8 * 0.5, rel=1e-6)

        unfloored = feedback_decay_score(importance=0.8, recall_count=1000, helpful_count=0, min_factor=0.0)
        assert unfloored == pytest.approx(0.0, abs=1e-10)
        assert unfloored < floored

    def test_helpful_counteracts_decay(self) -> None:
        """10 recalls, 10 helpful: exponent = 1, minimal decay."""
        score = feedback_decay_score(importance=0.8, recall_count=10, helpful_count=10)
        expected = 0.8 * (0.95**1)  # 0.8 * 0.95 = 0.76
        assert score == pytest.approx(expected, rel=1e-4)

    def test_high_helpful_ratio_preserves_score(self) -> None:
        """100 recalls, 100 helpful: exponent = 1, minimal decay."""
        score_100 = feedback_decay_score(importance=0.8, recall_count=100, helpful_count=100)
        score_10 = feedback_decay_score(importance=0.8, recall_count=10, helpful_count=10)
        assert score_100 == pytest.approx(score_10, rel=1e-4)

    def test_partial_helpful(self) -> None:
        """100 recalls, 10 helpful: exponent = 10, same as 10/0 case."""
        score = feedback_decay_score(importance=0.8, recall_count=100, helpful_count=10)
        expected = 0.8 * (0.95**10)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_clamped_to_01(self) -> None:
        """Score should never exceed 1.0 or go below 0.0."""
        score = feedback_decay_score(importance=1.0, recall_count=0, helpful_count=0)
        assert 0.0 <= score <= 1.0
        score2 = feedback_decay_score(importance=1.0, recall_count=10000, helpful_count=0)
        assert 0.0 <= score2 <= 1.0


class TestFeedbackInEntryUtility:
    """Verify feedback decay is wired into entry_utility."""

    def test_entry_utility_with_recall_no_helpful(self) -> None:
        """Entry with high recall_count but no helpful should have lower utility."""
        base_entry = {
            "importance": 0.8,
            "q_value": 0.5,
            "q_observations": 0,
            "recurrence": 1,
            "access_count": 0,
            "source": "agent",
            "recall_count": 0,
            "helpful_count": 0,
        }
        high_recall_entry = {**base_entry, "recall_count": 50}

        utility_base = entry_utility(base_entry)
        utility_high_recall = entry_utility(high_recall_entry)

        # High recall with no helpful should have lower utility
        assert utility_high_recall < utility_base

    def test_entry_utility_helpful_preserves_score(self) -> None:
        """Entry with balanced recall/helpful should preserve utility."""
        base_entry = {
            "importance": 0.8,
            "q_value": 0.5,
            "q_observations": 0,
            "recurrence": 1,
            "access_count": 0,
            "source": "agent",
            "recall_count": 50,
            "helpful_count": 50,
        }
        no_recall_entry = {**base_entry, "recall_count": 0, "helpful_count": 0}

        utility_balanced = entry_utility(base_entry)
        utility_none = entry_utility(no_recall_entry)

        # Balanced recall/helpful should be close to no-recall
        # (exponent = 1, so only 0.95^1 factor)
        assert abs(utility_balanced - utility_none) < 0.1
