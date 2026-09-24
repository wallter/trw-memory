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

    def test_detail_appended_when_new_is_shorter(self) -> None:
        """Bug fix (learning L-bvnz): a shorter incoming detail used to be
        silently dropped. It must now always be appended under the audit header.
        """
        existing = make_entry("e1", "content", detail="original long detail string here")
        new_entry = make_entry("e2", "content", detail="tiny")

        updated = merge_entries(existing, new_entry)
        assert "original long detail string here" in updated.detail
        assert "tiny" in updated.detail
        assert "Merged from e2" in updated.detail

    def test_detail_not_duplicated_when_already_present_verbatim(self) -> None:
        """Re-merging the same content must not inflate detail (no double append)."""
        existing = make_entry("e1", "content", detail="original detail\n---\nMerged from e2 on 2020-01-01:\ntiny")
        new_entry = make_entry("e2", "content", detail="tiny")

        updated = merge_entries(existing, new_entry)
        assert updated.detail.count("tiny") == 1

    def test_differing_incoming_content_preserved_in_audit_header(self) -> None:
        """Bug fix (learning L-bvnz): new_entry.content was never looked at, so a
        differing incoming summary was always lost. It must now survive in the
        audit header even when the survivor's own content is unchanged.
        """
        existing = make_entry("e1", "existing summary", detail="")
        new_entry = make_entry("e2", "a completely different incoming summary", detail="")

        updated = merge_entries(existing, new_entry)
        assert updated.content == "existing summary"  # survivor keeps its own content
        assert "a completely different incoming summary" in updated.detail
        assert "Merged from e2" in updated.detail

    def test_matching_incoming_content_not_duplicated_in_header(self) -> None:
        """When incoming content matches the survivor's, no summary rides the header."""
        existing = make_entry("e1", "same summary", detail="some detail")
        new_entry = make_entry("e2", "same summary", detail="new stuff")

        updated = merge_entries(existing, new_entry)
        assert "same summary" not in updated.detail
        assert "new stuff" in updated.detail

    def test_confidence_takes_higher(self) -> None:
        from trw_memory.models.memory import Confidence

        existing = make_entry("e1", "content", confidence=Confidence.UNVERIFIED)
        new_entry = make_entry("e2", "content", confidence=Confidence.VERIFIED)

        updated = merge_entries(existing, new_entry)
        assert updated.confidence == Confidence.VERIFIED.value

    def test_confidence_existing_wins_when_higher(self) -> None:
        from trw_memory.models.memory import Confidence

        existing = make_entry("e1", "content", confidence=Confidence.HIGH)
        new_entry = make_entry("e2", "content", confidence=Confidence.LOW)

        updated = merge_entries(existing, new_entry)
        assert updated.confidence == Confidence.HIGH.value

    def test_type_upgrades_pattern_to_incident(self) -> None:
        from trw_memory.models.memory import MemoryType

        existing = make_entry("e1", "content", type=MemoryType.PATTERN)
        new_entry = make_entry("e2", "content", type=MemoryType.INCIDENT)

        updated = merge_entries(existing, new_entry)
        assert updated.type == MemoryType.INCIDENT.value

    def test_type_stays_incident_when_incoming_is_pattern(self) -> None:
        from trw_memory.models.memory import MemoryType

        existing = make_entry("e1", "content", type=MemoryType.INCIDENT)
        new_entry = make_entry("e2", "content", type=MemoryType.PATTERN)

        updated = merge_entries(existing, new_entry)
        assert updated.type == MemoryType.INCIDENT.value

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

    def test_merged_from_chained_merge_keeps_incoming_ancestry(self) -> None:
        """A merged-away entry that had already absorbed others passes its ancestry on."""
        existing = make_entry("e1", "content", merged_from=["e0", "e3"])
        new_entry = make_entry("e2", "new content", merged_from=["e3", "e4", "e1"])

        updated = merge_entries(existing, new_entry)
        assert updated.merged_from == ["e0", "e3", "e2", "e4"]

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


def test_multi_line_incoming_summary_rides_the_header_on_one_line() -> None:
    """A multi-line incoming summary is kept whole, flattened so the audit header stays one line."""
    existing = make_entry(entry_id="L-a", content="Original summary", detail="old detail")
    incoming = make_entry(entry_id="L-b", content="Different\nsummary  spanning\nlines", detail="")
    merged = merge_entries(existing, incoming)
    header = merged.detail.split("---\n", 1)[1]
    assert header.startswith("Merged from L-b on ")
    assert header.endswith(": Different summary spanning lines")
