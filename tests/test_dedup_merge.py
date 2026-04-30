"""Tests for merge_entries behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.lifecycle.dedup import merge_entries

from ._test_dedup_support import make_entry


class TestMergeEntries:
    def test_tags_are_unioned(self) -> None:
        existing = make_entry("e1", "content", tags=["a", "b"])
        new_entry = make_entry("e2", "new content", tags=["b", "c"])

        updated = merge_entries(existing, new_entry)
        assert set(updated.tags) == {"a", "b", "c"}

    def test_evidence_is_unioned(self) -> None:
        existing = make_entry("e1", "content", evidence=["ev1", "ev2"])
        new_entry = make_entry("e2", "new content", evidence=["ev2", "ev3"])

        updated = merge_entries(existing, new_entry)
        assert set(updated.evidence) == {"ev1", "ev2", "ev3"}

    def test_importance_takes_max(self) -> None:
        existing = make_entry("e1", "content", importance=0.6)
        new_entry = make_entry("e2", "new content", importance=0.9)

        updated = merge_entries(existing, new_entry)
        assert updated.importance == 0.9

    def test_importance_existing_wins_when_higher(self) -> None:
        existing = make_entry("e1", "content", importance=0.9)
        new_entry = make_entry("e2", "new content", importance=0.6)

        updated = merge_entries(existing, new_entry)
        assert updated.importance == 0.9

    def test_recurrence_incremented(self) -> None:
        existing = make_entry("e1", "content", recurrence=3)
        new_entry = make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.recurrence == 4

    def test_detail_appended_when_new_is_longer(self) -> None:
        existing = make_entry("e1", "content", detail="short")
        new_entry = make_entry("e2", "content", detail="much longer detail with more information")

        updated = merge_entries(existing, new_entry)
        assert "short" in updated.detail
        assert "much longer detail" in updated.detail
        assert "Merged from e2" in updated.detail

    def test_detail_unchanged_when_new_is_shorter(self) -> None:
        existing = make_entry("e1", "content", detail="original long detail string here")
        new_entry = make_entry("e2", "content", detail="tiny")

        updated = merge_entries(existing, new_entry)
        assert updated.detail == "original long detail string here"

    def test_detail_set_when_existing_is_empty(self) -> None:
        existing = make_entry("e1", "content", detail="")
        new_entry = make_entry("e2", "content", detail="new detail")

        updated = merge_entries(existing, new_entry)
        assert "new detail" in updated.detail

    def test_merged_from_tracks_new_entry_id(self) -> None:
        existing = make_entry("e1", "content")
        new_entry = make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert "e2" in updated.merged_from

    def test_merged_from_no_duplicate(self) -> None:
        existing = make_entry("e1", "content", merged_from=["e2"])
        new_entry = make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.merged_from.count("e2") == 1

    def test_updated_at_changes(self) -> None:
        import time

        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        existing = make_entry("e1", "content")
        existing = existing.model_copy(update={"updated_at": old_time})
        new_entry = make_entry("e2", "new content")

        time.sleep(0.01)
        updated = merge_entries(existing, new_entry)
        assert updated.updated_at > old_time

    def test_returns_updated_entry_with_same_id(self) -> None:
        existing = make_entry("e1", "content")
        new_entry = make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.id == "e1"

    def test_tags_order_preserved_existing_first(self) -> None:
        existing = make_entry("e1", "content", tags=["b", "a"])
        new_entry = make_entry("e2", "content", tags=["c", "a"])

        updated = merge_entries(existing, new_entry)
        assert updated.tags[0] == "b"
        assert updated.tags[1] == "a"
        assert "c" in updated.tags
