"""Tests for trw_memory.retrieval.temporal_query — temporal query classifier."""

from __future__ import annotations

import pytest

from trw_memory.retrieval.temporal_query import (
    TemporalClassification,
    classify_temporal,
)


class TestClassifyTemporal:
    def test_plain_query_is_not_temporal(self) -> None:
        result = classify_temporal("what is the auth middleware pattern")
        assert result.is_temporal is False
        assert result.recency_weight == 0.0

    def test_latest_keyword_is_temporal(self) -> None:
        result = classify_temporal("show me the latest changes to the auth module")
        assert result.is_temporal is True
        assert result.recency_weight > 0.0

    def test_newest_keyword_is_temporal(self) -> None:
        result = classify_temporal("newest memory entries for the sprint")
        assert result.is_temporal is True

    def test_most_recent_is_temporal(self) -> None:
        result = classify_temporal("most recent session notes")
        assert result.is_temporal is True

    def test_recently_is_temporal(self) -> None:
        result = classify_temporal("what was recently added to trw-memory")
        assert result.is_temporal is True

    def test_last_week_is_temporal(self) -> None:
        result = classify_temporal("entries added last week")
        assert result.is_temporal is True

    def test_last_month_is_temporal(self) -> None:
        result = classify_temporal("bugs fixed last month")
        assert result.is_temporal is True

    def test_last_n_days_is_temporal(self) -> None:
        result = classify_temporal("changes in the last 7 days")
        assert result.is_temporal is True

    def test_today_is_temporal(self) -> None:
        result = classify_temporal("today's session notes")
        assert result.is_temporal is True

    def test_this_week_is_temporal(self) -> None:
        result = classify_temporal("this week's progress")
        assert result.is_temporal is True

    def test_was_updated_is_temporal(self) -> None:
        result = classify_temporal("which module was updated recently")
        assert result.is_temporal is True

    def test_recency_weight_increases_with_confidence(self) -> None:
        low_conf = classify_temporal("today")
        high_conf = classify_temporal("the latest session notes from this week")
        # higher confidence query → higher recency weight
        assert high_conf.recency_weight >= low_conf.recency_weight

    def test_recency_weight_bounded_max(self) -> None:
        result = classify_temporal(
            "the latest most recent newest entries from last week today"
        )
        assert result.recency_weight <= 0.6

    def test_non_temporal_query_has_zero_weight(self) -> None:
        result = classify_temporal("SQLite WAL checkpoint corruption")
        assert result.recency_weight == 0.0
        assert result.is_temporal is False

    def test_matched_patterns_populated(self) -> None:
        result = classify_temporal("latest session notes from last week")
        assert len(result.matched_patterns) >= 1
        assert "superlative" in result.matched_patterns

    def test_empty_query_is_not_temporal(self) -> None:
        result = classify_temporal("")
        assert result.is_temporal is False

    def test_case_insensitive(self) -> None:
        lower = classify_temporal("latest changes")
        upper = classify_temporal("LATEST changes")
        assert lower.is_temporal == upper.is_temporal

    def test_returns_temporal_classification(self) -> None:
        result = classify_temporal("recent memory entries")
        assert isinstance(result, TemporalClassification)

    def test_current_is_temporal(self) -> None:
        result = classify_temporal("current state of the retrieval pipeline")
        assert result.is_temporal is True

    def test_future_anchor_does_not_reach_threshold_alone(self) -> None:
        # A future anchor alone (confidence=0.4) stays below the 0.5 threshold.
        # Futures don't need freshness boosting — we want recency for past queries.
        result = classify_temporal("upcoming sprint tasks")
        assert result.is_temporal is False
        assert result.recency_weight == 0.0

    def test_future_anchor_combined_with_recency_is_temporal(self) -> None:
        result = classify_temporal("latest upcoming sprint tasks this week")
        assert result.is_temporal is True


class TestStripTemporalPrefix:
    """Tests for strip_temporal_prefix — removes boilerplate from temporal queries."""

    def setup_method(self) -> None:
        from trw_memory.retrieval.temporal_query import strip_temporal_prefix
        self._fn = strip_temporal_prefix

    def test_latest_guidance_on_stripped(self) -> None:
        assert self._fn("latest guidance on auth middleware") == "auth middleware"

    def test_current_guidance_on_stripped(self) -> None:
        assert self._fn("current guidance on session tokens") == "session tokens"

    def test_most_recent_guidance_on_stripped(self) -> None:
        assert self._fn("most recent guidance on SQLite WAL") == "SQLite WAL"

    def test_latest_guidance_for_stripped(self) -> None:
        assert self._fn("latest guidance for deployment") == "deployment"

    def test_latest_guidance_about_stripped(self) -> None:
        assert self._fn("latest guidance about config schema") == "config schema"

    def test_latest_information_on_stripped(self) -> None:
        assert self._fn("latest information on retries") == "retries"

    def test_the_latest_guidance_on_stripped(self) -> None:
        assert self._fn("the latest guidance on backoffs") == "backoffs"

    def test_whats_the_latest_guidance_on_stripped(self) -> None:
        assert self._fn("what's the latest guidance on CRDTs") == "CRDTs"

    def test_plain_query_unchanged(self) -> None:
        assert self._fn("auth middleware pattern") == "auth middleware pattern"

    def test_non_temporal_unchanged(self) -> None:
        assert self._fn("how does BM25 work") == "how does BM25 work"

    def test_empty_remainder_returns_original(self) -> None:
        # "latest guidance on" with no topic — should not strip to empty string
        result = self._fn("latest guidance on")
        assert result  # not empty

    def test_case_insensitive(self) -> None:
        assert self._fn("Latest Guidance On deployment") == "deployment"
        assert self._fn("LATEST GUIDANCE ON auth") == "auth"

    def test_what_is_current_state_stripped(self) -> None:
        result = self._fn("what is the current guidance on hooks")
        assert result == "hooks"

    def test_latest_without_guidance_stripped(self) -> None:
        result = self._fn("latest on deployment")
        assert result == "deployment"
