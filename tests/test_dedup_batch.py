"""Tests for batch_dedup behavior."""

from __future__ import annotations

import math

from trw_memory.lifecycle.dedup import batch_dedup
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus

from ._test_dedup_support import StubEmbedder, make_entry


def test_survivor_merged_content_in_updated_entries() -> None:
    """P0 regression: merged survivor must appear in updated_entries.

    Bug: original_map was built AFTER the mutation loop, so
    ``current != orig`` always compared merged-vs-merged (False) and
    the merged survivor was silently dropped from updated_entries.
    Fix: snapshot original_map BEFORE the loop.

    Verifies that after a near-duplicate merge the SURVIVOR entry (e1)
    is present in updated_entries with the merged recurrence incremented.
    """
    embedder = StubEmbedder(available=True)
    # e1 and e2 are near-duplicates (similarity >= merge_threshold, < skip)
    sq = math.sqrt(1.0 - 0.81)  # yields cosine(e1, e2) ≈ 0.9
    embedder.set_vector("alpha entry ", [1.0, 0.0, 0.0])
    embedder.set_vector("alpha entry dupe ", [0.9, sq, 0.0])

    e1 = make_entry("survivor", "alpha entry", recurrence=1)
    e2 = make_entry("absorbed", "alpha entry dupe", recurrence=1)

    config = MemoryConfig(
        dedup_skip_threshold=0.95,
        dedup_merge_threshold=0.85,
    )
    result = batch_dedup([e1, e2], embedder, config=config)

    assert result["status"] == "completed"
    assert result["entries_merged"] >= 1

    updated_ids = {e.id for e in result["updated_entries"]}  # type: ignore[union-attr]
    # The survivor (e1) must be in updated_entries with its merged state.
    assert "survivor" in updated_ids, (
        "Merged survivor 'survivor' must appear in updated_entries — "
        "original_map was not snapshotted before the mutation loop"
    )
    # The survivor's recurrence should be incremented (merged e2 into it).
    survivor = next(e for e in result["updated_entries"] if e.id == "survivor")  # type: ignore[union-attr]
    assert survivor.recurrence > e1.recurrence, (
        "Survivor recurrence must be incremented after merge"
    )


class TestBatchDedup:
    def test_embedder_unavailable_returns_skipped(self) -> None:
        embedder = StubEmbedder(available=False)
        entries = [
            make_entry("e1", "content one"),
            make_entry("e2", "content two"),
        ]
        result = batch_dedup(entries, embedder)
        assert result["status"] == "skipped"
        assert "unavailable" in str(result.get("reason", "")).lower()

    def test_no_entries_returns_skipped(self) -> None:
        embedder = StubEmbedder(available=True)
        result = batch_dedup([], embedder)
        assert result["status"] == "skipped"

    def test_no_duplicates_all_unchanged(self) -> None:
        embedder = StubEmbedder(available=True)
        embedder.set_vector("content one detail", [1.0, 0.0, 0.0])
        embedder.set_vector("content two detail", [0.0, 1.0, 0.0])
        embedder.set_vector("content three detail", [0.0, 0.0, 1.0])

        entries = [
            make_entry("e1", "content one", detail="detail"),
            make_entry("e2", "content two", detail="detail"),
            make_entry("e3", "content three", detail="detail"),
        ]
        result = batch_dedup(entries, embedder)
        assert result["status"] == "completed"
        assert result["entries_merged"] == 0
        assert result["entries_scanned"] == 3

    def test_exact_duplicates_second_obsoleted(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)
        entries = [
            make_entry("e1", "same content"),
            make_entry("e2", "same content"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_merged"] >= 1

    def test_near_duplicates_merged(self) -> None:
        embedder = StubEmbedder(available=True)

        sq = math.sqrt(1.0 - 0.81)
        embedder.set_vector("entry one ", [1.0, 0.0, 0.0])
        embedder.set_vector("entry two ", [0.9, sq, 0.0])

        entries = [
            make_entry("e1", "entry one"),
            make_entry("e2", "entry two"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_merged"] >= 1

    def test_skips_non_active_entries(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)

        entries = [
            make_entry("e1", "same content"),
            make_entry("e2", "same content", status=MemoryStatus.OBSOLETE),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_scanned"] == 1
        assert result["entries_merged"] == 0

    def test_returns_correct_counts(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("dup content ", vec)
        embedder.set_vector("unique content ", [0.0, 1.0, 0.0])

        entries = [
            make_entry("e1", "dup content"),
            make_entry("e2", "dup content"),
            make_entry("e3", "unique content"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_scanned"] == 3
        assert result["entries_merged"] == 1
        assert result["entries_skipped"] == 1

    def test_returns_updated_entries(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)

        entries = [
            make_entry("e1", "same content"),
            make_entry("e2", "same content"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert "updated_entries" in result
        updated = result["updated_entries"]
        assert isinstance(updated, list)
