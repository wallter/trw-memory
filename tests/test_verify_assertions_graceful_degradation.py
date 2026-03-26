"""Tests for graceful degradation of assertion verification.

PRD-CORE-086 FR09: Graceful degradation when project_root unavailable.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGracefulDegradation:
    """Test that verification degrades gracefully with missing/invalid inputs."""

    def test_project_root_none(self) -> None:
        """project_root=None returns all passed=None."""
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py"),
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
        ]
        results = verify_assertions(assertions, None)
        assert len(results) == 2
        assert all(r.passed is None for r in results)
        assert all("project_root unavailable" in r.evidence for r in results)

    def test_project_root_missing(self, tmp_path: Path) -> None:
        """Nonexistent path returns all passed=None."""
        nonexistent = tmp_path / "does_not_exist"
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py"),
        ]
        results = verify_assertions(assertions, nonexistent)
        assert len(results) == 1
        assert results[0].passed is None
        assert "project_root unavailable" in results[0].evidence

    def test_invalid_regex_returns_none(self, tmp_path: Path) -> None:
        """Bad regex pattern returns passed=None with error evidence."""
        (tmp_path / "test.py").write_text("content")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="[invalid(regex", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is None
        assert "invalid regex" in results[0].evidence

    def test_empty_assertions_returns_empty(self) -> None:
        """No assertions returns empty list."""
        results = verify_assertions([], Path("/tmp"))
        assert results == []

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        """File that can't be read is silently skipped."""
        f = tmp_path / "unreadable.py"
        f.write_text("hello_world")
        # Make file unreadable (only works if not root)
        import os
        if os.getuid() != 0:
            f.chmod(0o000)
            try:
                assertions = [
                    Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.py"),
                ]
                results = verify_assertions(assertions, tmp_path)
                # File is either skipped (binary check fails on read) or not readable
                # Either way, no crash
                assert len(results) == 1
            finally:
                f.chmod(0o644)
