"""Wave 12: targeted tests for uncovered branches in lifecycle/_recall.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trw_memory.lifecycle._recall import (
    rank_by_utility,
    record_recall_access,
    utility_based_prune_candidates,
)
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# rank_by_utility — empty-after-drop branch (line 113)
# ---------------------------------------------------------------------------


class TestRankByUtilityEdgePaths:
    def test_all_expired_returns_empty(self) -> None:
        """All entries expired → drop_expired empties list → early return."""
        expired_entry = {
            "id": "E-001",
            "content": "stale content",
            "expires": "2020-01-01T00:00:00+00:00",
        }
        result = rank_by_utility([expired_entry], query_tokens=["stale"], lambda_weight=0.5)
        assert result == []

    def test_empty_query_tokens_uses_wildcard_relevance(self) -> None:
        """Empty query_tokens → relevance = 1.0 for all entries (line 132)."""
        entries = [
            {"id": "E-001", "content": "alpha content", "tags": []},
            {"id": "E-002", "content": "beta content", "tags": []},
        ]
        result = rank_by_utility(entries, query_tokens=[], lambda_weight=0.0)
        # Both get relevance 1.0; order determined by utility (both near-zero → stable)
        assert len(result) == 2
        ids = {e["id"] for e in result}
        assert ids == {"E-001", "E-002"}

    def test_tags_not_a_list_is_handled(self) -> None:
        """Non-list tags are treated as empty string for relevance scoring."""
        entries = [{"id": "E-003", "content": "content", "tags": "not-a-list"}]
        result = rank_by_utility(entries, query_tokens=["content"], lambda_weight=0.0)
        assert len(result) == 1

    def test_empty_input_returns_empty(self) -> None:
        result = rank_by_utility([], query_tokens=["x"], lambda_weight=0.5)
        assert result == []


# ---------------------------------------------------------------------------
# record_recall_access — empty IDs early return (line 163)
# ---------------------------------------------------------------------------


class TestRecordRecallAccessEmptyIds:
    def test_empty_entry_ids_is_noop(self) -> None:
        """Empty entry_ids list → early return, no backend call."""
        backend = SQLiteBackend(Path(":memory:"))
        try:
            # No entries stored, empty list should not raise
            record_recall_access(backend, [])
        finally:
            backend.close()

    def test_explicit_accessed_at_is_forwarded(self) -> None:
        """accessed_at kwarg is passed through to increment_recall_access."""
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(MemoryEntry(id="R-001", content="test"))
            ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
            record_recall_access(backend, ["R-001"], accessed_at=ts)
            loaded = backend.get("R-001")
            assert loaded is not None
            assert loaded.recall_count == 1
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# utility_based_prune_candidates — full coverage (lines 201-272)
# ---------------------------------------------------------------------------


def _entry(
    *,
    id: str,
    status: str = "active",
    created_at: str = "2020-01-01",
    access_count: int = 0,
    recall_count: int = 0,
) -> dict[str, object]:
    return {
        "id": id,
        "content": f"content for {id}",
        "status": status,
        "created_at": created_at,
        "access_count": access_count,
        "recall_count": recall_count,
    }


class TestUtilityBasedPruneCandidates:
    def test_empty_entries_returns_empty(self) -> None:
        result = utility_based_prune_candidates([])
        assert result == []

    def test_resolved_status_is_candidate(self) -> None:
        """Status 'resolved' → immediately a cleanup candidate (Tier 1)."""
        result = utility_based_prune_candidates([_entry(id="P-001", status="resolved")])
        assert len(result) == 1
        assert result[0]["id"] == "P-001"
        assert result[0]["suggested_status"] == "resolved"

    def test_obsolete_status_is_candidate(self) -> None:
        """Status 'obsolete' → immediately a cleanup candidate (Tier 1)."""
        result = utility_based_prune_candidates([_entry(id="P-002", status="obsolete")])
        assert len(result) == 1
        assert result[0]["suggested_status"] == "obsolete"

    def test_low_utility_below_delete_threshold(self) -> None:
        """Utility < 0.05 → delete candidate (Tier 2)."""
        # Zero access_count, old date, no recalls → utility near zero
        result = utility_based_prune_candidates(
            [_entry(id="P-003", created_at="2018-01-01", access_count=0)],
            delete_threshold=1.0,  # force everything below threshold
        )
        assert len(result) == 1
        assert result[0]["id"] == "P-003"
        assert result[0]["suggested_status"] == "obsolete"

    def test_prune_threshold_with_old_entry(self) -> None:
        """Utility < prune_threshold and age > 14 days → obsolete candidate (Tier 3)."""
        result = utility_based_prune_candidates(
            [_entry(id="P-004", created_at="2020-01-01", access_count=0)],
            delete_threshold=0.0,  # nothing qualifies for delete
            prune_threshold=1.0,  # everything below this
        )
        assert len(result) == 1
        assert result[0]["id"] == "P-004"
        assert result[0]["suggested_status"] == "obsolete"

    def test_new_entry_below_prune_threshold_not_pruned(self) -> None:
        """Entry < 14 days old is excluded from Tier 3 pruning."""
        today = datetime.now(timezone.utc).date().isoformat()
        result = utility_based_prune_candidates(
            [_entry(id="P-005", created_at=today, access_count=0)],
            delete_threshold=0.0,
            prune_threshold=1.0,
        )
        assert result == []

    def test_duplicate_ids_deduplicated(self) -> None:
        """Same id appearing twice → only included once."""
        entry = _entry(id="P-006", status="resolved")
        result = utility_based_prune_candidates([entry, entry])
        assert len(result) == 1

    def test_active_high_utility_not_pruned(self) -> None:
        """Active entry with utility above both thresholds → not a candidate."""
        result = utility_based_prune_candidates(
            [_entry(id="P-007", created_at="2025-01-01", access_count=100)],
            delete_threshold=0.0,
            prune_threshold=0.0,
        )
        assert result == []

    def test_invalid_created_at_defaults_to_today(self) -> None:
        """Unparseable created_at falls back to today → age=0 → no Tier 3 prune."""
        result = utility_based_prune_candidates(
            [_entry(id="P-008", created_at="not-a-date", access_count=0)],
            delete_threshold=0.0,
            prune_threshold=1.0,
        )
        # age = 0 → fails age > 14 guard → not a candidate
        assert result == []

    def test_none_created_at_defaults_to_today(self) -> None:
        """None created_at → age_days=0 → not pruned by Tier 3."""
        entry = dict(_entry(id="P-009"))
        entry["created_at"] = None
        result = utility_based_prune_candidates(
            [entry],
            delete_threshold=0.0,
            prune_threshold=1.0,
        )
        assert result == []

    def test_reason_contains_threshold_values(self) -> None:
        """Tier 2 reason string includes threshold value."""
        result = utility_based_prune_candidates(
            [_entry(id="P-010", created_at="2018-01-01", access_count=0)],
            delete_threshold=1.0,
        )
        assert len(result) == 1
        reason = str(result[0]["reason"])
        assert "delete threshold" in reason
