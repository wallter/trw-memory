"""Tests for grep_absent assertion verification.

PRD-CORE-086 FR04: verify_assertions() for grep_absent type.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGrepAbsent:
    """Test grep_absent assertion type."""

    def test_grep_absent_no_match(self, tmp_path: Path) -> None:
        """Pattern not found in any file -> passed=True."""
        (tmp_path / "safe.py").write_text("def safe_function(): pass")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="eval\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is True
        assert "correctly absent" in results[0].evidence

    def test_grep_absent_found(self, tmp_path: Path) -> None:
        """Pattern found in a file -> passed=False."""
        (tmp_path / "bad.py").write_text("result = eval(user_input)")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="eval\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is False
        assert "unexpectedly found" in results[0].evidence

    def test_grep_absent_empty_dir(self, tmp_path: Path) -> None:
        """No files matching target -> passed=True (nothing to find)."""
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
        assert "correctly absent" in results[0].evidence

    def test_grep_absent_multiple_files_all_clean(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("safe code")
        (tmp_path / "b.py").write_text("also safe")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="exec\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True

    def test_grep_absent_one_file_has_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("safe code")
        (tmp_path / "b.py").write_text("exec('danger')")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="exec\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False
