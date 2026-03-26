"""Tests for AssertionResult model.

PRD-CORE-086: Executable assertion results.
"""

from __future__ import annotations

from trw_memory.models.memory import AssertionResult, AssertionType


class TestAssertionResult:
    """Test AssertionResult model construction and states."""

    def test_result_passed_true(self) -> None:
        r = AssertionResult(
            type=AssertionType.GREP_PRESENT,
            pattern="hello",
            target="*.py",
            passed=True,
            evidence="pattern found in 2 file(s)",
        )
        assert r.passed is True
        assert r.evidence == "pattern found in 2 file(s)"

    def test_result_passed_false(self) -> None:
        r = AssertionResult(
            type=AssertionType.GREP_PRESENT,
            pattern="hello",
            target="*.py",
            passed=False,
            evidence="pattern not found",
        )
        assert r.passed is False

    def test_result_passed_none(self) -> None:
        r = AssertionResult(
            type=AssertionType.GREP_PRESENT,
            pattern="hello",
            target="*.py",
            passed=None,
            evidence="project_root unavailable",
        )
        assert r.passed is None

    def test_result_with_evidence(self) -> None:
        evidence_text = "pattern found in 5 file(s): a.py, b.py, c.py, d.py, e.py"
        r = AssertionResult(
            type=AssertionType.GLOB_EXISTS,
            target="src/*.py",
            passed=True,
            evidence=evidence_text,
        )
        assert r.evidence == evidence_text

    def test_result_defaults(self) -> None:
        r = AssertionResult(type=AssertionType.GLOB_ABSENT)
        assert r.pattern == ""
        assert r.target == ""
        assert r.passed is None
        assert r.evidence == ""

    def test_result_json_round_trip(self) -> None:
        r = AssertionResult(
            type=AssertionType.GREP_ABSENT,
            pattern="old_api",
            target="src/**/*.py",
            passed=True,
            evidence="correctly absent from 10 files",
        )
        data = r.model_dump()
        restored = AssertionResult.model_validate(data, strict=False)

        assert restored.type == r.type
        assert restored.pattern == r.pattern
        assert restored.target == r.target
        assert restored.passed == r.passed
        assert restored.evidence == r.evidence

    def test_result_all_types(self) -> None:
        """All four assertion types can be used in results."""
        for atype in AssertionType:
            r = AssertionResult(type=atype, passed=True)
            assert r.type == atype.value
