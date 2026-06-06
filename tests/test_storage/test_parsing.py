"""Unit tests for the shared storage parsing seam (``storage/_parsing.py``).

Covers the fail-open contract that keeps a single malformed persisted field
from crashing the whole entry load in :func:`row_to_entry` / the YAML backend.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.storage._parsing import (
    parse_dt,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
)


@pytest.mark.unit
class TestParseJsonDictInt:
    def test_json_string_valid(self) -> None:
        assert parse_json_dict_int('{"node1": 3, "node2": 5}') == {"node1": 3, "node2": 5}

    def test_already_parsed_dict_valid(self) -> None:
        assert parse_json_dict_int({"node1": 10}) == {"node1": 10}

    def test_none_and_empty(self) -> None:
        assert parse_json_dict_int(None) == {}
        assert parse_json_dict_int("") == {}
        assert parse_json_dict_int({}) == {}

    def test_json_string_with_noninteger_value_degrades(self) -> None:
        # JSON-string branch was already guarded; assert it stays fail-open.
        assert parse_json_dict_int('{"node1": "x"}') == {}

    def test_already_parsed_dict_with_noninteger_value_degrades(self) -> None:
        # Regression: a YAML secondary store can hand back an already-parsed
        # dict whose value is not int-coercible. Previously ``int(v)`` ran
        # outside the try/except and raised ValueError, crashing entry load.
        assert parse_json_dict_int({"node1": "not-an-int"}) == {}

    def test_already_parsed_dict_with_none_value_degrades(self) -> None:
        # int(None) raises TypeError — must be caught, not propagated.
        assert parse_json_dict_int({"node1": None}) == {}

    def test_float_values_truncate_like_json_branch(self) -> None:
        # Behaviour-preserving: int(1.9) == 1 for both branches.
        assert parse_json_dict_int({"node1": 1.9}) == {"node1": 1}
        assert parse_json_dict_int('{"node1": 1.9}') == {"node1": 1}

    def test_non_object_json_returns_empty(self) -> None:
        assert parse_json_dict_int("[1, 2, 3]") == {}
        assert parse_json_dict_int("42") == {}

    def test_malformed_json_returns_empty(self) -> None:
        assert parse_json_dict_int("{not json") == {}


@pytest.mark.unit
class TestParseJsonDictStr:
    def test_json_string_valid(self) -> None:
        assert parse_json_dict_str('{"k": "v"}') == {"k": "v"}

    def test_already_parsed_dict_coerces_values(self) -> None:
        assert parse_json_dict_str({"k": 3}) == {"k": "3"}

    def test_malformed_returns_empty(self) -> None:
        assert parse_json_dict_str("{not json") == {}


@pytest.mark.unit
class TestParseJsonList:
    def test_json_string_valid(self) -> None:
        assert parse_json_list('["a", "b"]') == ["a", "b"]

    def test_already_parsed_list(self) -> None:
        assert parse_json_list(["a", 1]) == ["a", "1"]

    def test_none_uses_fallback(self) -> None:
        assert parse_json_list(None) == []
        assert parse_json_list(None, fallback=["x"]) == ["x"]

    def test_malformed_uses_fallback(self) -> None:
        assert parse_json_list("{not json", fallback=["x"]) == ["x"]


@pytest.mark.unit
class TestParseDt:
    def test_naive_string_assumed_utc(self) -> None:
        assert parse_dt("2024-01-15T10:30:00") == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_aware_datetime_normalised_to_utc(self) -> None:
        aware = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert parse_dt(aware) == aware
