"""Tests for grep_present assertion verification.

PRD-CORE-086 FR04: verify_assertions() for grep_present type.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGrepPresentFound:
    """Test grep_present when pattern IS found."""

    def test_grep_present_found(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("def hello_world(): pass")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is True
        assert "hello.py" in results[0].evidence

    def test_grep_present_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("def greet(): pass")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is False
        assert "not found" in results[0].evidence

    def test_grep_present_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def hello(): pass")
        (tmp_path / "b.py").write_text("def hello(): return True")
        (tmp_path / "c.py").write_text("def world(): pass")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
        assert "2 file(s)" in results[0].evidence

    def test_regex_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("x = re.compile(r'\\d{3}-\\d{4}')")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern=r"re\.compile", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True


class TestGrepPresentEdgeCases:
    """Test edge cases for grep_present."""

    def test_binary_file_skipped(self, tmp_path: Path) -> None:
        # Create a binary file with null bytes
        (tmp_path / "binary.py").write_bytes(b"hello_world\x00binary_data")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False  # Binary file skipped, so pattern not found

    def test_oversized_file_skipped(self, tmp_path: Path) -> None:
        # Create a file larger than 1MB
        large_content = "hello_world\n" * 200_000  # ~2.4MB
        (tmp_path / "large.py").write_text(large_content)
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False  # Oversized file skipped
        assert "file exceeds 1MB limit" in results[0].evidence

    def test_default_excludes_applied(self, tmp_path: Path) -> None:
        # Create file in __pycache__ which is in DEFAULT_EXCLUDES
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.py").write_text("hello_world")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="**/*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False  # File in excluded dir

    def test_empty_assertions_returns_empty(self) -> None:
        results = verify_assertions([], Path("/tmp"))
        assert results == []
