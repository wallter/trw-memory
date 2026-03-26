"""Tests for Assertion and AssertionType models.

PRD-CORE-086: Executable assertions data model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import Assertion, AssertionType


class TestAssertionTypeEnum:
    """Test the AssertionType enum has all 4 values."""

    def test_assertion_type_enum(self) -> None:
        assert AssertionType.GREP_PRESENT == "grep_present"
        assert AssertionType.GREP_ABSENT == "grep_absent"
        assert AssertionType.GLOB_EXISTS == "glob_exists"
        assert AssertionType.GLOB_ABSENT == "glob_absent"
        assert len(AssertionType) == 4


class TestAssertionModel:
    """Test the Assertion Pydantic model."""

    def test_assertion_model_grep_present(self) -> None:
        a = Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py")
        assert a.type == "grep_present"
        assert a.pattern == "hello"
        assert a.target == "*.py"
        assert a.last_result is None
        assert a.last_verified_at is None
        assert a.last_evidence == ""
        assert a.first_failed_at is None

    def test_assertion_model_grep_absent(self) -> None:
        a = Assertion(type=AssertionType.GREP_ABSENT, pattern="deprecated_func", target="src/**/*.py")
        assert a.type == "grep_absent"
        assert a.pattern == "deprecated_func"

    def test_assertion_model_glob_exists(self) -> None:
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/main.py")
        assert a.type == "glob_exists"
        assert a.pattern == ""

    def test_assertion_model_glob_absent(self) -> None:
        a = Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.bak")
        assert a.type == "glob_absent"

    def test_empty_pattern_rejected_for_grep_present(self) -> None:
        with pytest.raises(ValidationError, match="grep assertion types require a non-empty pattern"):
            Assertion(type=AssertionType.GREP_PRESENT, pattern="", target="*.py")

    def test_empty_pattern_rejected_for_grep_absent(self) -> None:
        with pytest.raises(ValidationError, match="grep assertion types require a non-empty pattern"):
            Assertion(type=AssertionType.GREP_ABSENT, pattern="", target="*.py")

    def test_whitespace_only_pattern_rejected_for_grep(self) -> None:
        with pytest.raises(ValidationError, match="grep assertion types require a non-empty pattern"):
            Assertion(type=AssertionType.GREP_PRESENT, pattern="   ", target="*.py")

    def test_empty_pattern_allowed_for_glob_exists(self) -> None:
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="*.py")
        assert a.pattern == ""

    def test_empty_pattern_allowed_for_glob_absent(self) -> None:
        a = Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.py")
        assert a.pattern == ""

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute paths not allowed"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="/etc/passwd")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="../secret.txt")

    def test_path_traversal_middle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/../../secret.txt")

    def test_pattern_length_limit(self) -> None:
        long_pattern = "a" * 501
        with pytest.raises(ValidationError, match="pattern exceeds 500 character limit"):
            Assertion(type=AssertionType.GREP_PRESENT, pattern=long_pattern, target="*.py")

    def test_pattern_at_limit_accepted(self) -> None:
        pattern_500 = "a" * 500
        a = Assertion(type=AssertionType.GREP_PRESENT, pattern=pattern_500, target="*.py")
        assert len(a.pattern) == 500

    def test_empty_target_rejected(self) -> None:
        """Target has min_length=1, so empty string should fail."""
        with pytest.raises(ValidationError):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="")

    def test_first_failed_at_transitions(self) -> None:
        """first_failed_at can be set/cleared and round-trips correctly."""
        now = datetime.now(timezone.utc)
        a = Assertion(
            type=AssertionType.GREP_PRESENT,
            pattern="hello",
            target="*.py",
            first_failed_at=now,
        )
        assert a.first_failed_at == now

        # Round-trip via model_dump/model_validate
        data = a.model_dump()
        a2 = Assertion.model_validate(data, strict=False)
        assert a2.first_failed_at == now

        # Can be None
        a3 = Assertion(
            type=AssertionType.GREP_PRESENT,
            pattern="hello",
            target="*.py",
            first_failed_at=None,
        )
        assert a3.first_failed_at is None

    def test_json_round_trip(self) -> None:
        """model_dump() -> model_validate() preserves all fields."""
        now = datetime.now(timezone.utc)
        a = Assertion(
            type=AssertionType.GREP_PRESENT,
            pattern="def hello",
            target="src/**/*.py",
            last_result=True,
            last_verified_at=now,
            last_evidence="found in 3 files",
            first_failed_at=None,
        )
        data = a.model_dump()
        restored = Assertion.model_validate(data, strict=False)

        assert restored.type == a.type
        assert restored.pattern == a.pattern
        assert restored.target == a.target
        assert restored.last_result == a.last_result
        assert restored.last_verified_at == a.last_verified_at
        assert restored.last_evidence == a.last_evidence
        assert restored.first_failed_at == a.first_failed_at


class TestPublicImports:
    """Test that models are importable from the public package."""

    def test_public_imports(self) -> None:
        from trw_memory.models import Assertion, AssertionResult, AssertionType

        assert AssertionType.GREP_PRESENT is not None
        assert Assertion is not None
        assert AssertionResult is not None
