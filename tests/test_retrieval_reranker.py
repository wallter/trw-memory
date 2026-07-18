"""Tests for trw_memory.retrieval.reranker — cross-encoder re-ranking."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval import reranker
from trw_memory.retrieval.reranker import _CROSS_ENCODER_AVAILABLE, cross_encode_rerank


def _entry(id: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id=id,
        content=content,
        tags=[],
        valid_from=datetime.now(timezone.utc),
    )


class TestCrossEncodeRerank:
    def test_empty_input_returns_empty(self) -> None:
        result = cross_encode_rerank("query", [])
        assert result == []

    def test_returns_same_count_by_default(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(5)]
        result = cross_encode_rerank("test query", entries)
        assert len(result) == 5

    def test_top_k_truncates(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(10)]
        result = cross_encode_rerank("test query", entries, top_k=3)
        assert len(result) == 3

    def test_all_input_entries_present_in_output(self) -> None:
        entries = [_entry(f"e{i}", f"content about topic {i}") for i in range(5)]
        result = cross_encode_rerank("topic query", entries)
        assert {e.id for e in result} == {e.id for e in entries}

    @pytest.mark.skipif(not _CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
    def test_relevant_entry_ranks_higher(self) -> None:
        entries = [
            _entry("relevant", "SQLite WAL checkpoint corruption fix for concurrent writes"),
            _entry("irrelevant", "Pydantic model field validation with use_enum_values"),
            _entry("noise", "Frontend React component state management hook"),
        ]
        result = cross_encode_rerank("SQLite database corruption fix", entries)
        # The relevant entry should rank first or second after cross-encoding
        top_ids = [e.id for e in result[:2]]
        assert "relevant" in top_ids

    def test_missing_model_falls_back_gracefully(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(3)]
        # Using a nonexistent model name should fall back to input order
        result = cross_encode_rerank(
            "test query",
            entries,
            model_name="nonexistent/model-that-does-not-exist",
        )
        assert len(result) == 3
        # Should return entries (possibly in original order due to fallback)
        assert {e.id for e in result} == {e.id for e in entries}

    def test_single_entry_returns_single_entry(self) -> None:
        entries = [_entry("e1", "only entry")]
        result = cross_encode_rerank("query", entries)
        assert len(result) == 1
        assert result[0].id == "e1"


class TestLazyCrossEncoderImport:
    """The sentence_transformers import is deferred to first use (behavior-preserving)."""

    def test_import_cross_encoder_matches_availability_flag(self) -> None:
        # The public lazy-resolved flag must agree with the probe helper.
        assert reranker._import_cross_encoder() is _CROSS_ENCODER_AVAILABLE

    def test_import_cross_encoder_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First real probe populates the cache; a subsequent call must NOT
        # re-enter the import machinery. We assert idempotence of the flag.
        first = reranker._import_cross_encoder()
        # Corrupt the class cache but leave the availability flag: a cached
        # probe must short-circuit and return the same answer without retrying.
        monkeypatch.setattr(reranker, "_cross_encoder_cls", object())
        assert reranker._import_cross_encoder() is first

    def test_degrades_gracefully_when_dependency_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate sentence_transformers being unavailable: cross_encode_rerank
        # must preserve input order exactly (the fusion ranking is untouched).
        monkeypatch.setattr(reranker, "_cross_encoder_available", False)
        monkeypatch.setattr(reranker, "_cross_encoder_cls", None)
        entries = [_entry(f"e{i}", f"content {i}") for i in range(4)]
        result = cross_encode_rerank("test query", entries)
        assert [e.id for e in result] == [e.id for e in entries]

    def test_degraded_path_honors_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reranker, "_cross_encoder_available", False)
        monkeypatch.setattr(reranker, "_cross_encoder_cls", None)
        entries = [_entry(f"e{i}", f"content {i}") for i in range(6)]
        result = cross_encode_rerank("test query", entries, top_k=2)
        assert [e.id for e in result] == ["e0", "e1"]
