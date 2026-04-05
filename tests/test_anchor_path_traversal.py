"""Tests for Anchor.file path traversal rejection (PRD-CORE-111 fix).

Anchor._validate_file must reject '..' path components, matching
Assertion._validate_target() behavior for security parity.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import Anchor


class TestAnchorPathTraversal:
    """Anchor.file must reject path traversal (..) components."""

    def test_dotdot_raises(self) -> None:
        """Anchor.file containing '..' raises ValidationError."""
        with pytest.raises(ValidationError, match="path traversal"):
            Anchor(file="../etc/passwd", symbol_name="foo")

    def test_dotdot_in_middle_raises(self) -> None:
        """Anchor.file with '..' in middle raises ValidationError."""
        with pytest.raises(ValidationError, match="path traversal"):
            Anchor(file="src/../../../etc/shadow", symbol_name="bar")

    def test_single_dotdot_raises(self) -> None:
        """Anchor.file of just '..' raises ValidationError."""
        with pytest.raises(ValidationError, match="path traversal"):
            Anchor(file="..", symbol_name="baz")

    def test_dotdot_suffix_allowed(self) -> None:
        """Filenames that contain '..' but not as a path component are allowed."""
        anchor = Anchor(file="src/foo..bar.py", symbol_name="fn")
        assert anchor.file == "src/foo..bar.py"

    def test_absolute_still_rejected(self) -> None:
        """Absolute paths are still rejected independently of traversal check."""
        with pytest.raises(ValidationError, match="relative path"):
            Anchor(file="/usr/bin/python", symbol_name="main")

    def test_normal_relative_path_accepted(self) -> None:
        """Normal relative paths are accepted."""
        anchor = Anchor(file="src/trw_memory/models/memory.py", symbol_name="Anchor")
        assert anchor.file == "src/trw_memory/models/memory.py"
