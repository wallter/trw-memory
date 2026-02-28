"""Tests for lifecycle/dedup.py — semantic deduplication.

Covers:
- cosine_similarity: identical vectors, orthogonal, zero vectors, dimension mismatch
- check_duplicate: embedder unavailable → store, skip threshold, merge threshold,
  below merge → store, no existing entries, skips non-active entries
- merge_entries: tags union, evidence union, impact max, recurrence increment,
  detail append when new is longer, detail unchanged when new is shorter,
  merged_from tracking, updated_at change
- batch_dedup: embedder unavailable, no entries, exact duplicates obsoleted,
  near-duplicates merged, unchanged entries pass through
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.lifecycle.dedup import (
    DedupResult,
    check_duplicate,
    merge_entries,
    batch_dedup,
)
from trw_memory.retrieval.dense import cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    entry_id: str,
    content: str,
    detail: str = "",
    tags: list[str] | None = None,
    evidence: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    recurrence: int = 1,
    merged_from: list[str] | None = None,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        evidence=evidence or [],
        importance=importance,
        status=status,
        recurrence=recurrence,
        merged_from=merged_from or [],
        created_at=now,
        updated_at=now,
    )


class _StubEmbedder:
    """Minimal EmbeddingProvider stub with deterministic embeddings."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        # Map text -> fixed vector for deterministic testing
        self._vectors: dict[str, list[float]] = {}

    def set_vector(self, text: str, vector: list[float]) -> None:
        self._vectors[text] = vector

    def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        if text in self._vectors:
            return self._vectors[text]
        # Default: use first 3 chars
        return [float(ord(c)) / 128.0 for c in (text[:3].ljust(3))]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(t) for t in texts]

    def available(self) -> bool:
        return self._available

    def dim(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_empty_lists(self) -> None:
        assert cosine_similarity([], []) == 0.0

    def test_partial_similarity(self) -> None:
        # 45-degree angle → cos(45°) ≈ 0.707
        import math
        v = 1.0 / math.sqrt(2.0)
        a = [v, v, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(v, abs=1e-5)

    def test_unnormalized_vectors(self) -> None:
        # cos similarity handles unnormalized vectors
        a = [3.0, 0.0]
        b = [0.0, 4.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_duplicate
# ---------------------------------------------------------------------------

class TestCheckDuplicate:
    def test_embedder_unavailable_returns_store(self) -> None:
        embedder = _StubEmbedder(available=False)
        entries: list[MemoryEntry] = [_make_entry("e1", "some content")]
        result = check_duplicate("new content", entries, embedder)
        assert result == DedupResult("store", None, 0.0)

    def test_no_entries_returns_store(self) -> None:
        embedder = _StubEmbedder(available=True)
        result = check_duplicate("new content", [], embedder)
        assert result == DedupResult("store", None, 0.0)

    def test_skip_when_similarity_at_threshold(self) -> None:
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        # Entry text = content + " " + detail = "existing content "
        embedder.set_vector("existing content ", vec)
        # New text = content + " " + detail = "new content "
        embedder.set_vector("new content ", vec)

        entries = [_make_entry("e1", "existing content")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        assert result.action == "skip"
        assert result.existing_id == "e1"
        assert result.similarity == pytest.approx(1.0)

    def test_merge_when_similarity_between_thresholds(self) -> None:
        embedder = _StubEmbedder(available=True)
        import math
        # 90% cosine similarity (between 0.85 merge and 0.95 skip thresholds)
        sq = math.sqrt(1.0 - 0.81)
        vec_existing = [1.0, 0.0, 0.0]
        vec_new = [0.9, sq, 0.0]

        # Entry text = "existing entry " (content + " " + "")
        embedder.set_vector("existing entry ", vec_existing)
        # New text = "new entry " (content + " " + "")
        embedder.set_vector("new entry ", vec_new)

        entries = [_make_entry("e1", "existing entry")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new entry", entries, embedder, config=config)
        assert result.action == "merge"
        assert result.existing_id == "e1"
        assert result.similarity == pytest.approx(0.9, abs=1e-5)

    def test_store_when_similarity_below_merge(self) -> None:
        embedder = _StubEmbedder(available=True)
        # Orthogonal → similarity = 0.0
        # Entry text and new text use trailing space (content + " " + "")
        embedder.set_vector("completely different content ", [1.0, 0.0, 0.0])
        embedder.set_vector("existing entry with no overlap ", [0.0, 1.0, 0.0])

        entries = [_make_entry("e1", "existing entry with no overlap")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("completely different content", entries, embedder, config=config)
        assert result.action == "store"
        assert result.existing_id is None

    def test_skips_non_active_entries(self) -> None:
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("archived content ", vec)
        embedder.set_vector("new content ", vec)

        archived_entry = _make_entry("e1", "archived content", status=MemoryStatus.ARCHIVED)
        entries = [archived_entry]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        # Should store since archived entry is skipped
        assert result.action == "store"
        assert result.existing_id is None

    def test_picks_best_matching_entry(self) -> None:
        embedder = _StubEmbedder(available=True)
        import math
        # e1 has ~0.9 similarity, e2 has 1.0 similarity
        vec_new = [1.0, 0.0, 0.0]
        vec_e1 = [0.9, math.sqrt(0.19), 0.0]
        vec_e2 = [1.0, 0.0, 0.0]  # 1.0 similarity

        # Entry texts use trailing space (content + " " + "")
        embedder.set_vector("new content ", vec_new)
        embedder.set_vector("entry one ", vec_e1)
        embedder.set_vector("entry two ", vec_e2)

        entries = [
            _make_entry("e1", "entry one"),
            _make_entry("e2", "entry two"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        assert result.action == "skip"
        assert result.existing_id == "e2"

    def test_threshold_validation_merge_gte_skip_uses_defaults(self) -> None:
        embedder = _StubEmbedder(available=True)
        # Invalid config: merge >= skip
        config = MemoryConfig(
            dedup_skip_threshold=0.80,
            dedup_merge_threshold=0.90,  # merge > skip → invalid
        )
        # Should fall back to defaults (0.95/0.85)
        # With orthogonal vectors, action should be "store"
        embedder.set_vector("content a ", [1.0, 0.0, 0.0])
        embedder.set_vector("content b ", [0.0, 1.0, 0.0])
        entries = [_make_entry("e1", "content b")]
        result = check_duplicate("content a", entries, embedder, config=config)
        assert result.action == "store"

    def test_uses_content_plus_detail_for_embedding(self) -> None:
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        # The lookup key is "content detail" (content + " " + detail)
        embedder.set_vector("my content my detail", vec)
        embedder.set_vector("existing content existing detail", vec)

        entries = [_make_entry("e1", "existing content", detail="existing detail")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("my content", entries, embedder, config=config, detail="my detail")
        assert result.action == "skip"
        assert result.existing_id == "e1"


# ---------------------------------------------------------------------------
# merge_entries
# ---------------------------------------------------------------------------

class TestMergeEntries:
    def test_tags_are_unioned(self) -> None:
        existing = _make_entry("e1", "content", tags=["a", "b"])
        new_entry = _make_entry("e2", "new content", tags=["b", "c"])

        updated = merge_entries(existing, new_entry)
        assert set(updated.tags) == {"a", "b", "c"}

    def test_evidence_is_unioned(self) -> None:
        existing = _make_entry("e1", "content", evidence=["ev1", "ev2"])
        new_entry = _make_entry("e2", "new content", evidence=["ev2", "ev3"])

        updated = merge_entries(existing, new_entry)
        assert set(updated.evidence) == {"ev1", "ev2", "ev3"}

    def test_importance_takes_max(self) -> None:
        existing = _make_entry("e1", "content", importance=0.6)
        new_entry = _make_entry("e2", "new content", importance=0.9)

        updated = merge_entries(existing, new_entry)
        assert updated.importance == pytest.approx(0.9)

    def test_importance_existing_wins_when_higher(self) -> None:
        existing = _make_entry("e1", "content", importance=0.9)
        new_entry = _make_entry("e2", "new content", importance=0.6)

        updated = merge_entries(existing, new_entry)
        assert updated.importance == pytest.approx(0.9)

    def test_recurrence_incremented(self) -> None:
        existing = _make_entry("e1", "content", recurrence=3)
        new_entry = _make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.recurrence == 4

    def test_detail_appended_when_new_is_longer(self) -> None:
        existing = _make_entry("e1", "content", detail="short")
        new_entry = _make_entry("e2", "content", detail="much longer detail with more information")

        updated = merge_entries(existing, new_entry)
        assert "short" in updated.detail
        assert "much longer detail" in updated.detail
        assert "Merged from e2" in updated.detail

    def test_detail_unchanged_when_new_is_shorter(self) -> None:
        existing = _make_entry("e1", "content", detail="original long detail string here")
        new_entry = _make_entry("e2", "content", detail="tiny")

        updated = merge_entries(existing, new_entry)
        assert updated.detail == "original long detail string here"

    def test_detail_set_when_existing_is_empty(self) -> None:
        existing = _make_entry("e1", "content", detail="")
        new_entry = _make_entry("e2", "content", detail="new detail")

        updated = merge_entries(existing, new_entry)
        assert "new detail" in updated.detail

    def test_merged_from_tracks_new_entry_id(self) -> None:
        existing = _make_entry("e1", "content")
        new_entry = _make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert "e2" in updated.merged_from

    def test_merged_from_no_duplicate(self) -> None:
        existing = _make_entry("e1", "content", merged_from=["e2"])
        new_entry = _make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.merged_from.count("e2") == 1

    def test_updated_at_changes(self) -> None:
        import time
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        existing = _make_entry("e1", "content")
        existing = existing.model_copy(update={"updated_at": old_time})
        new_entry = _make_entry("e2", "new content")

        time.sleep(0.01)  # ensure time passes
        updated = merge_entries(existing, new_entry)
        assert updated.updated_at > old_time

    def test_returns_updated_entry_with_same_id(self) -> None:
        existing = _make_entry("e1", "content")
        new_entry = _make_entry("e2", "new content")

        updated = merge_entries(existing, new_entry)
        assert updated.id == "e1"

    def test_tags_order_preserved_existing_first(self) -> None:
        existing = _make_entry("e1", "content", tags=["b", "a"])
        new_entry = _make_entry("e2", "content", tags=["c", "a"])

        updated = merge_entries(existing, new_entry)
        # existing tags come first, then new-only tags
        assert updated.tags[0] == "b"
        assert updated.tags[1] == "a"
        assert "c" in updated.tags


# ---------------------------------------------------------------------------
# batch_dedup
# ---------------------------------------------------------------------------

class TestBatchDedup:
    def test_embedder_unavailable_returns_skipped(self) -> None:
        embedder = _StubEmbedder(available=False)
        entries: list[MemoryEntry] = [
            _make_entry("e1", "content one"),
            _make_entry("e2", "content two"),
        ]
        result = batch_dedup(entries, embedder)
        assert result["status"] == "skipped"
        assert "unavailable" in str(result.get("reason", "")).lower()

    def test_no_entries_returns_skipped(self) -> None:
        embedder = _StubEmbedder(available=True)
        result = batch_dedup([], embedder)
        assert result["status"] == "skipped"

    def test_no_duplicates_all_unchanged(self) -> None:
        embedder = _StubEmbedder(available=True)
        # Orthogonal vectors → no duplicates
        embedder.set_vector("content one detail", [1.0, 0.0, 0.0])
        embedder.set_vector("content two detail", [0.0, 1.0, 0.0])
        embedder.set_vector("content three detail", [0.0, 0.0, 1.0])

        entries = [
            _make_entry("e1", "content one", detail="detail"),
            _make_entry("e2", "content two", detail="detail"),
            _make_entry("e3", "content three", detail="detail"),
        ]
        result = batch_dedup(entries, embedder)
        assert result["status"] == "completed"
        assert result["entries_merged"] == 0
        assert result["entries_scanned"] == 3

    def test_exact_duplicates_second_obsoleted(self) -> None:
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)  # content + " " + "" = "same content "
        entries = [
            _make_entry("e1", "same content"),
            _make_entry("e2", "same content"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_merged"] >= 1

    def test_near_duplicates_merged(self) -> None:
        embedder = _StubEmbedder(available=True)
        import math
        # 0.9 similarity → merge (above merge threshold 0.85, below skip 0.95)
        sq = math.sqrt(1.0 - 0.81)
        embedder.set_vector("entry one ", [1.0, 0.0, 0.0])
        embedder.set_vector("entry two ", [0.9, sq, 0.0])

        entries = [
            _make_entry("e1", "entry one"),
            _make_entry("e2", "entry two"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        assert result["entries_merged"] >= 1

    def test_skips_non_active_entries(self) -> None:
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)

        entries = [
            _make_entry("e1", "same content"),
            _make_entry("e2", "same content", status=MemoryStatus.OBSOLETE),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert result["status"] == "completed"
        # e2 is already obsolete, should not be scanned
        assert result["entries_scanned"] == 1
        assert result["entries_merged"] == 0

    def test_returns_correct_counts(self) -> None:
        embedder = _StubEmbedder(available=True)
        # One exact duplicate pair, one unique
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("dup content ", vec)
        embedder.set_vector("unique content ", [0.0, 1.0, 0.0])

        entries = [
            _make_entry("e1", "dup content"),
            _make_entry("e2", "dup content"),
            _make_entry("e3", "unique content"),
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
        """batch_dedup should return list of updated entries."""
        embedder = _StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("same content ", vec)

        entries = [
            _make_entry("e1", "same content"),
            _make_entry("e2", "same content"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = batch_dedup(entries, embedder, config=config)
        assert "updated_entries" in result
        updated = result["updated_entries"]
        assert isinstance(updated, list)
