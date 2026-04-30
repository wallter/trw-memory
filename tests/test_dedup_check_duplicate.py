"""Tests for check_duplicate behavior."""

from __future__ import annotations

import math

import pytest

from trw_memory.lifecycle.dedup import DedupResult, check_duplicate
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus

from ._test_dedup_support import StubEmbedder, make_entry


class TestCheckDuplicate:
    def test_embedder_unavailable_returns_store(self) -> None:
        embedder = StubEmbedder(available=False)
        entries = [make_entry("e1", "some content")]
        result = check_duplicate("new content", entries, embedder)
        assert result == DedupResult("store", None, 0.0)

    def test_no_entries_returns_store(self) -> None:
        embedder = StubEmbedder(available=True)
        result = check_duplicate("new content", [], embedder)
        assert result == DedupResult("store", None, 0.0)

    def test_skip_when_similarity_at_threshold(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("existing content ", vec)
        embedder.set_vector("new content ", vec)

        entries = [make_entry("e1", "existing content")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        assert result.action == "skip"
        assert result.existing_id == "e1"
        assert result.similarity == pytest.approx(1.0)

    def test_merge_when_similarity_between_thresholds(self) -> None:
        embedder = StubEmbedder(available=True)

        sq = math.sqrt(1.0 - 0.81)
        vec_existing = [1.0, 0.0, 0.0]
        vec_new = [0.9, sq, 0.0]

        embedder.set_vector("existing entry ", vec_existing)
        embedder.set_vector("new entry ", vec_new)

        entries = [make_entry("e1", "existing entry")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new entry", entries, embedder, config=config)
        assert result.action == "merge"
        assert result.existing_id == "e1"
        assert result.similarity == pytest.approx(0.9, abs=1e-5)

    def test_store_when_similarity_below_merge(self) -> None:
        embedder = StubEmbedder(available=True)
        embedder.set_vector("completely different content ", [1.0, 0.0, 0.0])
        embedder.set_vector("existing entry with no overlap ", [0.0, 1.0, 0.0])

        entries = [make_entry("e1", "existing entry with no overlap")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate(
            "completely different content",
            entries,
            embedder,
            config=config,
        )
        assert result.action == "store"
        assert result.existing_id is None

    def test_skips_non_active_entries(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("archived content ", vec)
        embedder.set_vector("new content ", vec)

        entries = [make_entry("e1", "archived content", status=MemoryStatus.ARCHIVED)]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        assert result.action == "store"
        assert result.existing_id is None

    def test_picks_best_matching_entry(self) -> None:
        embedder = StubEmbedder(available=True)

        vec_new = [1.0, 0.0, 0.0]
        vec_e1 = [0.9, math.sqrt(0.19), 0.0]
        vec_e2 = [1.0, 0.0, 0.0]

        embedder.set_vector("new content ", vec_new)
        embedder.set_vector("entry one ", vec_e1)
        embedder.set_vector("entry two ", vec_e2)

        entries = [
            make_entry("e1", "entry one"),
            make_entry("e2", "entry two"),
        ]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("new content", entries, embedder, config=config)
        assert result.action == "skip"
        assert result.existing_id == "e2"

    def test_threshold_validation_merge_gte_skip_uses_defaults(self) -> None:
        embedder = StubEmbedder(available=True)
        config = MemoryConfig(
            dedup_skip_threshold=0.80,
            dedup_merge_threshold=0.90,
        )
        embedder.set_vector("content a ", [1.0, 0.0, 0.0])
        embedder.set_vector("content b ", [0.0, 1.0, 0.0])
        entries = [make_entry("e1", "content b")]
        result = check_duplicate("content a", entries, embedder, config=config)
        assert result.action == "store"

    def test_uses_content_plus_detail_for_embedding(self) -> None:
        embedder = StubEmbedder(available=True)
        vec = [1.0, 0.0, 0.0]
        embedder.set_vector("my content my detail", vec)
        embedder.set_vector("existing content existing detail", vec)

        entries = [make_entry("e1", "existing content", detail="existing detail")]
        config = MemoryConfig(
            dedup_skip_threshold=0.95,
            dedup_merge_threshold=0.85,
        )
        result = check_duplicate("my content", entries, embedder, config=config, detail="my detail")
        assert result.action == "skip"
        assert result.existing_id == "e1"
