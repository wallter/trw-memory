"""Tests for glob_absent assertion verification.

PRD-CORE-086 FR04: verify_assertions() for glob_absent type.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGlobAbsent:
    """Test glob_absent assertion type."""

    def test_glob_absent_no_files(self, tmp_path: Path) -> None:
        """No files matching pattern -> passed=True."""
        assertions = [
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.bak"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is True
        assert "correctly no files" in results[0].evidence

    def test_glob_absent_files_found(self, tmp_path: Path) -> None:
        """Files matching pattern -> passed=False."""
        (tmp_path / "backup.bak").write_text("old data")
        assertions = [
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.bak"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is False
        assert "unexpectedly found" in results[0].evidence

    def test_glob_absent_excluded_dirs_dont_count(self, tmp_path: Path) -> None:
        """Files in excluded directories should not cause failure."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text("{}")
        assertions = [
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="**/package.json"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True  # node_modules is excluded

    def test_glob_absent_multiple_matches(self, tmp_path: Path) -> None:
        (tmp_path / "a.tmp").write_text("tmp")
        (tmp_path / "b.tmp").write_text("tmp")
        assertions = [
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.tmp"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False
        assert "2 file(s) unexpectedly found" in results[0].evidence
