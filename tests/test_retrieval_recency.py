"""Tests for trw_memory.retrieval.recency — recency-based ranking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recency import (
    DEFAULT_HALFLIFE_DAYS,
    _MIN_RECENCY_SCORE,
    recency_rank,
    recency_score,
)


def _entry(id: str, days_ago: float, now: datetime) -> MemoryEntry:
    ts = now - timedelta(days=days_ago)
    return MemoryEntry(
        id=id,
        content=f"entry {id}",
        tags=[],
        created_at=ts,
        valid_from=ts,
    )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


class TestRecencyScore:
    def test_brand_new_entry_scores_near_one(self, now: datetime) -> None:
        e = _entry("e1", 0.0, now)
        s = recency_score(e, now, 30.0)
        assert s > 0.99

    def test_halflife_gives_half_score(self, now: datetime) -> None:
        e = _entry("e1", 30.0, now)
        s = recency_score(e, now, 30.0)
        assert abs(s - 0.5) < 0.01

    def test_ancient_entry_hits_floor(self, now: datetime) -> None:
        e = _entry("e1", 10_000.0, now)
        s = recency_score(e, now, 30.0)
        assert s == _MIN_RECENCY_SCORE

    def test_score_monotone_decreasing(self, now: datetime) -> None:
        ages = [0, 7, 30, 90, 365]
        scores = [recency_score(_entry("e", d, now), now, 30.0) for d in ages]
        for a, b in zip(scores, scores[1:]):
            assert a >= b

    def test_longer_halflife_slower_decay(self, now: datetime) -> None:
        e = _entry("e", 30.0, now)
        s30 = recency_score(e, now, 30.0)
        s90 = recency_score(e, now, 90.0)
        assert s90 > s30

    def test_uses_valid_from_not_created_at(self, now: datetime) -> None:
        # Entry created today but valid_from is 60 days ago
        e = MemoryEntry(
            id="e1",
            content="test",
            tags=[],
            created_at=now,
            valid_from=now - timedelta(days=60),
        )
        s = recency_score(e, now, 30.0)
        # 60 days / halflife 30 = 2 half-lives → score ≈ 0.25
        assert abs(s - 0.25) < 0.02


class TestRecencyRank:
    def test_empty_returns_empty(self, now: datetime) -> None:
        assert recency_rank([], now=now) == []

    def test_returns_most_recent_first(self, now: datetime) -> None:
        entries = [_entry(f"e{i}", i * 10, now) for i in range(5)]
        results = recency_rank(entries, now=now)
        ids = [eid for eid, _ in results]
        assert ids == ["e0", "e1", "e2", "e3", "e4"]

    def test_scores_strictly_decreasing(self, now: datetime) -> None:
        entries = [_entry(f"e{i}", i * 10, now) for i in range(5)]
        results = recency_rank(entries, now=now)
        scores = [s for _, s in results]
        for a, b in zip(scores, scores[1:]):
            assert a > b

    def test_top_k_truncates(self, now: datetime) -> None:
        entries = [_entry(f"e{i}", i, now) for i in range(10)]
        results = recency_rank(entries, top_k=3, now=now)
        assert len(results) == 3

    def test_all_valid_entries_appear_in_output(self, now: datetime) -> None:
        entries = [_entry(f"e{i}", float(i), now) for i in range(5)]
        results = recency_rank(entries, now=now)
        assert len(results) == len(entries)

    def test_halflife_affects_score_magnitude(self, now: datetime) -> None:
        entries = [_entry("e1", 30.0, now)]
        r30 = recency_rank(entries, halflife_days=30.0, now=now)
        r90 = recency_rank(entries, halflife_days=90.0, now=now)
        # Longer halflife → higher score at same age
        assert r90[0][1] > r30[0][1]

    def test_default_halflife_is_30_days(self, now: datetime) -> None:
        e = _entry("e1", DEFAULT_HALFLIFE_DAYS, now)
        results = recency_rank([e], now=now)
        # At halflife age, score should be ≈ 0.5
        assert abs(results[0][1] - 0.5) < 0.01

    def test_all_entries_receive_scores(self, now: datetime) -> None:
        entries = [_entry(f"e{i}", i * 5, now) for i in range(20)]
        results = recency_rank(entries, now=now)
        assert len(results) == 20
        assert all(s > 0 for _, s in results)
