"""Tests for MemoryEntry empty-string coercion backward compatibility.

PRD-CORE-110 Fix A: MemoryEntry validators for type, confidence, and
protection_tier must accept empty strings and return the default enum value.
This handles pre-migration SQLite rows where DEFAULT may not backfill.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import (
    Confidence,
    MemoryEntry,
    MemoryType,
    ProtectionTier,
)


class TestMemoryEntryEmptyStringCoercion:
    """Empty strings in enum fields should coerce to defaults, not raise."""

    def test_empty_type_defaults_to_pattern(self) -> None:
        entry = MemoryEntry(id="test", content="test", type="")
        assert entry.type == MemoryType.PATTERN.value

    def test_empty_confidence_defaults_to_unverified(self) -> None:
        entry = MemoryEntry(id="test", content="test", confidence="")
        assert entry.confidence == Confidence.UNVERIFIED.value

    def test_empty_protection_tier_defaults_to_normal(self) -> None:
        entry = MemoryEntry(id="test", content="test", protection_tier="")
        assert entry.protection_tier == ProtectionTier.NORMAL.value

    def test_all_three_empty_together(self) -> None:
        """All three empty strings at once should not raise."""
        entry = MemoryEntry(
            id="test", content="test",
            type="", confidence="", protection_tier="",
        )
        assert entry.type == MemoryType.PATTERN.value
        assert entry.confidence == Confidence.UNVERIFIED.value
        assert entry.protection_tier == ProtectionTier.NORMAL.value

    def test_valid_values_still_work(self) -> None:
        """Non-empty valid values are accepted as before."""
        entry = MemoryEntry(
            id="test", content="test",
            type="incident", confidence="high", protection_tier="critical",
        )
        assert entry.type == MemoryType.INCIDENT.value
        assert entry.confidence == Confidence.HIGH.value
        assert entry.protection_tier == ProtectionTier.CRITICAL.value

    def test_invalid_non_empty_still_raises(self) -> None:
        """Invalid non-empty strings still raise ValueError."""
        with pytest.raises(ValidationError, match="type must be one of"):
            MemoryEntry(id="test", content="test", type="bogus")
        with pytest.raises(ValidationError, match="confidence must be one of"):
            MemoryEntry(id="test", content="test", confidence="excellent")
        with pytest.raises(ValidationError, match="protection_tier must be one of"):
            MemoryEntry(id="test", content="test", protection_tier="top-secret")
