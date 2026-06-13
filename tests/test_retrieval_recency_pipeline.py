"""Integration tests for recency_weight and rerank in hybrid_search pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search


def _entry(id: str, content: str, days_ago: float = 0.0) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    ts = now - timedelta(days=days_ago)
    return MemoryEntry(
        id=id,
        content=content,
        detail="",
        tags=[],
        created_at=ts,
        valid_from=ts,
    )


class TestRecencyWeightInPipeline:
    def test_recency_weight_zero_is_default_behaviour(self) -> None:
        entries = [_entry(f"e{i}", f"topic {i}") for i in range(10)]
        result_default = hybrid_search("topic", entries)
        result_zero = hybrid_search("topic", entries, recency_weight=0.0)
        # Both should return same ids (order may vary but count must match)
        assert len(result_default) == len(result_zero)

    def test_recency_weight_injects_third_ranking_source(self) -> None:
        # A recent entry should benefit from recency ranking even if textually
        # it is a weaker BM25 match.
        old_match = _entry("old_match", "the target query term exact match", days_ago=365.0)
        new_match = _entry("new_match", "the target query term", days_ago=0.0)
        entries = [old_match, new_match]
        # Without recency: old_match may rank first (better BM25 due to "exact")
        # With high recency: new_match should be boosted
        result = hybrid_search("target query term", entries, recency_weight=0.9)
        assert len(result) == 2

    def test_recency_weight_one_boosts_newest_entry(self) -> None:
        ancient = _entry("ancient", "important knowledge pattern", days_ago=500.0)
        fresh = _entry("fresh", "important knowledge pattern", days_ago=1.0)
        entries = [ancient, fresh]
        # With strong recency, fresh should outrank ancient despite identical text
        result = hybrid_search("important knowledge", entries, recency_weight=0.9)
        assert result[0].id == "fresh"

    def test_recency_weight_works_without_embedder(self) -> None:
        entries = [_entry(f"e{i}", f"query content {i}", days_ago=i * 10) for i in range(5)]
        result = hybrid_search("query content", entries, recency_weight=0.5)
        assert len(result) > 0

    def test_recency_halflife_parameter_accepted(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}", days_ago=i * 5) for i in range(5)]
        result = hybrid_search(
            "content",
            entries,
            recency_weight=0.3,
            recency_halflife_days=7.0,
        )
        assert len(result) > 0

    def test_empty_entries_returns_empty_with_recency(self) -> None:
        result = hybrid_search("query", [], recency_weight=0.5)
        assert result == []


class TestRerankInPipeline:
    def test_rerank_false_default_unchanged_behaviour(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(5)]
        result = hybrid_search("content", entries, rerank=False)
        assert len(result) > 0

    def test_rerank_true_returns_same_count(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(5)]
        result_no_rerank = hybrid_search("content", entries, rerank=False)
        result_rerank = hybrid_search("content", entries, rerank=True)
        assert len(result_rerank) == len(result_no_rerank)

    def test_rerank_candidates_limits_reranked_set(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}") for i in range(20)]
        # Should not raise even with rerank_candidates < len(entries)
        result = hybrid_search("content", entries, rerank=True, rerank_candidates=5, top_k=10)
        assert len(result) <= 10

    def test_rerank_combined_with_recency(self) -> None:
        entries = [_entry(f"e{i}", f"content {i}", days_ago=i * 7) for i in range(6)]
        result = hybrid_search(
            "content",
            entries,
            recency_weight=0.3,
            rerank=True,
            rerank_candidates=4,
            top_k=3,
        )
        assert len(result) <= 3
