"""Tests for lifecycle/_utils.py — shared days_since_access helper.

Covers:
- datetime object input
- date (non-datetime) object input
- ISO string with time component (T separator)
- Date-only string
- None value (uses fallback)
- "None" string sentinel
- "null" string sentinel
- Empty string sentinel
- Field resolution order (last_accessed_at takes priority over created_at)
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from trw_memory.lifecycle._utils import days_since_access

# ---------------------------------------------------------------------------
# Basic type handling
# ---------------------------------------------------------------------------


class TestDaysSinceAccessTypes:
    def test_datetime_object(self) -> None:
        today = date(2026, 3, 3)
        ts = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)  # 10 days before
        entry: dict[str, object] = {"last_accessed_at": ts}
        result = days_since_access(entry, today)
        assert result == 10

    def test_date_object(self) -> None:
        today = date(2026, 3, 3)
        d = date(2026, 2, 26)  # 5 days before
        entry: dict[str, object] = {"last_accessed_at": d}
        result = days_since_access(entry, today)
        assert result == 5

    def test_iso_string_with_time(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": "2026-02-24T12:00:00+00:00"}
        result = days_since_access(entry, today)
        assert result == 7

    def test_date_only_string(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": "2026-02-28"}
        result = days_since_access(entry, today)
        assert result == 3

    def test_iso_string_with_space_separator(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": "2026-03-01 10:00:00+00:00"}
        result = days_since_access(entry, today)
        assert result == 2


# ---------------------------------------------------------------------------
# Sentinel / None handling
# ---------------------------------------------------------------------------


class TestDaysSinceAccessSentinels:
    def test_none_value_uses_fallback(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": None}
        result = days_since_access(entry, today, fallback_days=42)
        assert result == 42

    def test_none_string_sentinel(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": "None"}
        result = days_since_access(entry, today, fallback_days=42)
        assert result == 42

    def test_null_string_sentinel(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": "null"}
        result = days_since_access(entry, today, fallback_days=42)
        assert result == 42

    def test_empty_string_sentinel(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"last_accessed_at": ""}
        result = days_since_access(entry, today, fallback_days=42)
        assert result == 42

    def test_no_fields_at_all_uses_fallback(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {"content": "no date fields"}
        result = days_since_access(entry, today, fallback_days=30)
        assert result == 30


# ---------------------------------------------------------------------------
# Field resolution order
# ---------------------------------------------------------------------------


class TestDaysSinceAccessFieldOrder:
    def test_last_accessed_at_takes_priority(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {
            "last_accessed_at": "2026-02-26T12:00:00+00:00",  # 5 days
            "created_at": "2025-11-23T12:00:00+00:00",  # 100 days
            "created": "2025-08-15T12:00:00+00:00",  # 200 days
        }
        result = days_since_access(entry, today)
        assert result == 5

    def test_falls_back_to_created_at(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {
            "last_accessed_at": None,
            "created_at": "2026-02-11T12:00:00+00:00",  # 20 days
        }
        result = days_since_access(entry, today)
        assert result == 20

    def test_falls_back_to_created(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {
            "last_accessed_at": None,
            "created_at": None,
            "created": "2026-01-12T12:00:00+00:00",  # 50 days
        }
        result = days_since_access(entry, today)
        assert result == 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDaysSinceAccessEdgeCases:
    def test_future_date_returns_zero(self) -> None:
        today = date(2026, 3, 3)
        future = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
        entry: dict[str, object] = {"last_accessed_at": future}
        result = days_since_access(entry, today)
        assert result == 0

    def test_invalid_string_skipped(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {
            "last_accessed_at": "not-a-date",
            "created_at": "2026-02-16T12:00:00+00:00",  # 15 days
        }
        result = days_since_access(entry, today)
        assert result == 15

    def test_default_fallback_is_30(self) -> None:
        today = date(2026, 3, 3)
        entry: dict[str, object] = {}
        result = days_since_access(entry, today)
        assert result == 30
