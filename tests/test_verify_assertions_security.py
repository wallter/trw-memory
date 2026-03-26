"""Security tests for assertion verification engine.

PRD-CORE-086: Ensure no shell execution, path traversal, or ReDoS vectors.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import Assertion, AssertionType


class TestNoShellExecution:
    """Verify the verification engine has no shell execution paths."""

    def test_no_shell_execution_in_verification(self) -> None:
        """Grep verification.py for dangerous execution functions."""
        from pathlib import Path

        verification_path = Path(__file__).parent.parent / "src" / "trw_memory" / "lifecycle" / "verification.py"
        assert verification_path.exists(), f"verification.py not found at {verification_path}"

        content = verification_path.read_text()
        dangerous_patterns = [
            "subprocess",
            "os.system",
            "os.popen",
            "eval(",
            "exec(",
            "os.exec",
            "__import__",
        ]
        for pattern in dangerous_patterns:
            assert pattern not in content, (
                f"SECURITY: verification.py contains '{pattern}' — "
                f"shell execution is forbidden in assertion verification"
            )


class TestPathTraversal:
    """Verify that path traversal is rejected at the model level."""

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute paths not allowed"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="/etc/passwd")

    def test_path_traversal_rejected_leading(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="../secret.txt")

    def test_path_traversal_rejected_middle(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GREP_PRESENT, pattern="x", target="src/../../etc/passwd")

    def test_safe_path_with_dots_allowed(self) -> None:
        """Paths with dots that aren't traversal should be OK."""
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/.hidden/file.py")
        assert a.target == "src/.hidden/file.py"

    def test_dotdot_in_filename_allowed(self) -> None:
        """A filename containing '..' but not as a path component should be OK."""
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="file..name.py")
        assert a.target == "file..name.py"


class TestPatternLengthLimit:
    """Verify pattern length limit for ReDoS mitigation."""

    def test_pattern_length_limit_501(self) -> None:
        with pytest.raises(ValidationError, match="pattern exceeds 500 character limit"):
            Assertion(
                type=AssertionType.GREP_PRESENT,
                pattern="a" * 501,
                target="*.py",
            )

    def test_pattern_length_limit_500_ok(self) -> None:
        a = Assertion(
            type=AssertionType.GREP_PRESENT,
            pattern="a" * 500,
            target="*.py",
        )
        assert len(a.pattern) == 500

    def test_pattern_length_limit_1000(self) -> None:
        with pytest.raises(ValidationError, match="pattern exceeds 500 character limit"):
            Assertion(
                type=AssertionType.GREP_PRESENT,
                pattern="x" * 1000,
                target="*.py",
            )
