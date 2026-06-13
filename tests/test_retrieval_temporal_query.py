"""Tests for trw_memory.retrieval.temporal_query — temporal query classifier."""

from __future__ import annotations

from trw_memory.retrieval.temporal_query import (
    TemporalClassification,
    classify_temporal,
    prepare_temporal_query,
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

    def test_temporal_arithmetic_n_days_ago(self) -> None:
        result = classify_temporal("What kitchen appliance did I buy 10 days ago?")
        assert result.is_temporal is True
        assert "temporal_arithmetic" in result.matched_patterns

    def test_temporal_arithmetic_n_weeks_ago(self) -> None:
        result = classify_temporal("What did I do two weeks ago?")
        assert result.is_temporal is True

    def test_temporal_arithmetic_last_weekday(self) -> None:
        result = classify_temporal("Who did I meet with last Tuesday?")
        assert result.is_temporal is True
        assert "temporal_arithmetic" in result.matched_patterns

    def test_temporal_arithmetic_months_ago(self) -> None:
        result = classify_temporal("What did I do on the Wednesday two months ago?")
        assert result.is_temporal is True

    def test_prior_context_previous_conversation(self) -> None:
        result = classify_temporal("In our previous conversation, you mentioned something")
        assert result.is_temporal is True
        assert "prior_context" in result.matched_patterns

    def test_prior_context_previous_chat(self) -> None:
        result = classify_temporal("I was looking back at our previous chat and wanted to confirm")
        assert result.is_temporal is True

    def test_prior_context_low_recency_weight(self) -> None:
        # Prior context and temporal arithmetic shouldn't get huge recency boosts;
        # they're not "give me recent stuff" queries. Confidence=0.6 which maps to
        # a moderate recency_weight — still less than superlative recency (0.9).
        result = classify_temporal("In our previous chat, you said something")
        result2 = classify_temporal("latest auth middleware guidance")
        assert result.recency_weight <= result2.recency_weight


class TestPrepareTemporalQuery:
    def test_strips_prefix_and_auto_fills_zero_recency_weight(self) -> None:
        result = prepare_temporal_query(
            "latest guidance on auth middleware",
            current_recency_weight=0.0,
            auto_temporal=True,
            strip_prefix=True,
        )

        assert result.retrieval_query == "auth middleware"
        assert result.recency_weight > 0.0
        assert result.prefix_stripped is True
        assert result.classification is not None
        assert result.classification.is_temporal is True

    def test_preserves_explicit_recency_weight(self) -> None:
        result = prepare_temporal_query(
            "latest guidance on auth middleware",
            current_recency_weight=0.25,
            auto_temporal=True,
            strip_prefix=True,
        )

        assert result.retrieval_query == "auth middleware"
        assert result.recency_weight == 0.25

    def test_disabled_auto_temporal_is_passthrough(self) -> None:
        result = prepare_temporal_query(
            "latest guidance on auth middleware",
            current_recency_weight=0.0,
            auto_temporal=False,
            strip_prefix=True,
        )

        assert result.retrieval_query == "latest guidance on auth middleware"
        assert result.recency_weight == 0.0
        assert result.classification is None


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

    def test_previous_conversation_about_stripped(self) -> None:
        result = self._fn(
            "I was looking back at our previous conversation about Native American powwows and I was wondering"
        )
        assert "powwows" in result
        assert "looking back" not in result.lower()

    def test_going_back_to_previous_conversation_stripped(self) -> None:
        result = self._fn(
            "I'm going back to our previous conversation about the children's book on dinosaurs. Can you remind me"
        )
        assert "dinosaurs" in result

    def test_wanted_to_follow_up_stripped(self) -> None:
        result = self._fn(
            "I wanted to follow up on our previous conversation about binaural beats"
        )
        assert "binaural beats" in result
        assert "follow up" not in result.lower()

    def test_remember_you_told_stripped(self) -> None:
        result = self._fn(
            "I remember you told me about the refining processes at CITGO"
        )
        assert "refining processes" in result or "CITGO" in result

    def test_we_discussed_stripped(self) -> None:
        result = self._fn("I think we discussed work from home jobs for seniors earlier")
        assert "work from home" in result or "seniors" in result

    def test_previous_conversation_empty_remainder_is_safe(self) -> None:
        result = self._fn("I was looking back at our previous conversation about")
        assert result  # must not be empty

    def test_previous_conversation_strip_case_insensitive(self) -> None:
        result = self._fn(
            "I WAS LOOKING BACK AT OUR PREVIOUS CONVERSATION ABOUT deployment practices"
        )
        assert "deployment practices" in result

    def test_previous_chat_and_wanted_to_confirm_stripped(self) -> None:
        result = self._fn(
            "I was looking back at our previous chat and I wanted to confirm, "
            "how many times did the Chiefs play the Jaguars at Arrowhead Stadium?"
        )
        assert "Chiefs" in result
        assert "looking back" not in result.lower()

    def test_previous_chat_and_wanted_to_verify_stripped(self) -> None:
        result = self._fn(
            "I was looking back at our previous conversation and I wanted to verify, "
            "what was the deployment command?"
        )
        assert "deployment command" in result
