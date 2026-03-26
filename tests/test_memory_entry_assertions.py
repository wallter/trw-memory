"""Tests for assertions field on MemoryEntry.

PRD-CORE-086: MemoryEntry.assertions integration.
"""

from __future__ import annotations

import json

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry


class TestMemoryEntryAssertions:
    """Test that MemoryEntry properly supports the assertions field."""

    def test_default_empty_assertions(self) -> None:
        entry = MemoryEntry(id="M-001", content="test content")
        assert entry.assertions == []
        assert isinstance(entry.assertions, list)

    def test_entry_with_assertions(self) -> None:
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="def hello", target="*.py"),
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
        ]
        entry = MemoryEntry(id="M-002", content="test content", assertions=assertions)
        assert len(entry.assertions) == 2
        assert entry.assertions[0].type == "grep_present"
        assert entry.assertions[1].type == "glob_exists"

    def test_entry_assertions_round_trip(self) -> None:
        """model_dump -> model_validate preserves assertions."""
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="import os", target="src/**/*.py"),
            Assertion(type=AssertionType.GREP_ABSENT, pattern="eval\\(", target="src/**/*.py"),
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="pyproject.toml"),
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.bak"),
        ]
        entry = MemoryEntry(id="M-003", content="test", assertions=assertions)

        data = entry.model_dump()
        assert len(data["assertions"]) == 4

        restored = MemoryEntry.model_validate(data, strict=False)
        assert len(restored.assertions) == 4
        assert restored.assertions[0].type == "grep_present"
        assert restored.assertions[0].pattern == "import os"
        assert restored.assertions[1].type == "grep_absent"
        assert restored.assertions[2].type == "glob_exists"
        assert restored.assertions[3].type == "glob_absent"

    def test_entry_assertions_json_serialization(self) -> None:
        """JSON serialization/deserialization works for assertions."""
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py"),
        ]
        entry = MemoryEntry(id="M-004", content="test", assertions=assertions)

        # Serialize to JSON
        data = entry.model_dump()
        json_str = json.dumps(data, default=str)
        parsed = json.loads(json_str)

        # Verify assertions survived JSON round-trip
        assert len(parsed["assertions"]) == 1
        assert parsed["assertions"][0]["type"] == "grep_present"
        assert parsed["assertions"][0]["pattern"] == "hello"
        assert parsed["assertions"][0]["target"] == "*.py"

    def test_entry_without_assertions_backward_compatible(self) -> None:
        """Entries without assertions field still work (default=[])."""
        data = {
            "id": "M-005",
            "content": "legacy entry",
        }
        entry = MemoryEntry.model_validate(data, strict=False)
        assert entry.assertions == []
