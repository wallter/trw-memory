"""Tests for batch_dedup behavior."""

from __future__ import annotations

import math

from trw_memory.lifecycle.dedup import batch_dedup
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus

from ._test_dedup_support import StubEmbedder, make_entry


class CountingEmbedder(StubEmbedder):
    """StubEmbedder that counts individual embed() vs embed_batch() calls."""

    def __init__(self, available: bool = True) -> None:
        super().__init__(available=available)
        self.embed_calls = 0
        self.embed_batch_calls = 0

    def embed(self, text: str) -> list[float] | None:
        self.embed_calls += 1
        return super().embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        self.embed_batch_calls += 1
        return super().embed_batch(texts)


def test_batch_dedup_uses_embed_batch_not_per_entry_embed() -> None:
    """batch_dedup must use embed_batch() for all entries in a single call.

    Previously the function called embedder.embed() per entry (N individual
    round-trips). The fix collects all active entry texts first and calls
    embed_batch() once for the whole active set, matching the documented
    'dedup batch embed' optimisation.

    Verify: embed_batch is called exactly once (for all N active entries
    together) regardless of how many active entries exist. The pre-fix code
    called embed_batch() N times (once per entry via embedder.embed), so
    the count was proportional to N. After the fix, the count is always 1.
    """
    embedder = CountingEmbedder(available=True)
    embedder.set_vector("alpha content ", [1.0, 0.0, 0.0])
    embedder.set_vector("beta content ", [0.0, 1.0, 0.0])
    embedder.set_vector("gamma content ", [0.0, 0.0, 1.0])

    entries = [
        make_entry("e1", "alpha content"),
        make_entry("e2", "beta content"),
        make_entry("e3", "gamma content"),
    ]

    batch_dedup(entries, embedder)

    # embed_batch must be called exactly once for all 3 active entries.
    # Pre-fix: embed() was called 3 times (once per entry), so embed_batch_calls
    # would have been 0 (the old code used embed(), not embed_batch()).
    # Post-fix: embed_batch_calls == 1, embed_calls == 0 from dedup.py itself
    # (StubEmbedder.embed_batch internally calls self.embed() per text, which
    # increments our embed_calls counter; that's expected provider behavior
    # and does NOT indicate a regression in dedup.py's call pattern).
    assert embedder.embed_batch_calls == 1, (
        f"Expected 1 embed_batch call for 3 active entries, got {embedder.embed_batch_calls}; "
        "batch_dedup must call embed_batch exactly once for the full active set"
    )


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
