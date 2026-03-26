"""Tests for glob_exists assertion verification.

PRD-CORE-086 FR04: verify_assertions() for glob_exists type.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGlobExists:
    """Test glob_exists assertion type."""

    def test_glob_exists_found(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Hello")
        assertions = [
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is True
        assert "1 file(s) found" in results[0].evidence

    def test_glob_exists_not_found(self, tmp_path: Path) -> None:
        assertions = [
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is False
        assert "no files found" in results[0].evidence

    def test_glob_exists_wildcard(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.py").write_text("pass")
        (tmp_path / "c.txt").write_text("text")
        assertions = [
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
        assert "2 file(s) found" in results[0].evidence

    def test_glob_exists_with_excludes(self, tmp_path: Path) -> None:
        """Files in excluded directories should be filtered out."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("pass")
        # No non-excluded py files
        assertions = [
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="**/*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False  # Excluded dir file doesn't count

    def test_glob_exists_nested(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("pass")
        assertions = [
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/main.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
