"""Tests for MemoryEntry typed learning classification fields (PRD-CORE-110).

Covers:
- Default values for the 10 new classification fields
- Field validation for type, confidence, protection_tier enums
- nudge_line truncation at word boundary
- domain list max-20 constraint
- phase_affinity list max-6 constraint
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import MemoryEntry

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_new_fields_defaults() -> None:
    """Default MemoryEntry has correct defaults for all 10 new fields."""
    entry = MemoryEntry(id="L-test", content="test content")
    assert entry.type == "pattern"
    assert entry.nudge_line == ""
    assert entry.expires == ""
    assert entry.confidence == "unverified"
    assert entry.task_type == ""
    assert entry.domain == []
    assert entry.phase_origin == ""
    assert entry.phase_affinity == []
    assert entry.team_origin == ""
    assert entry.protection_tier == "normal"


# ---------------------------------------------------------------------------
# type enum validation
# ---------------------------------------------------------------------------


def test_type_enum_valid_incident() -> None:
    """type='incident' is accepted."""
    entry = MemoryEntry(id="L-x", content="c", type="incident")
    assert entry.type == "incident"


def test_type_enum_valid_all() -> None:
    """All valid type values are accepted."""
    valid_types = ["incident", "pattern", "convention", "hypothesis", "workaround"]
    for t in valid_types:
        entry = MemoryEntry(id="L-x", content="c", type=t)
        assert entry.type == t


def test_type_enum_validation() -> None:
    """Invalid type raises ValidationError."""
    with pytest.raises(ValidationError, match="type must be one of"):
        MemoryEntry(id="L-x", content="c", type="bogus")


def test_type_enum_validation_empty() -> None:
    """Empty string type coerces to PATTERN (backward compat)."""
    entry = MemoryEntry(id="L-x", content="c", type="")
    assert entry.type == "pattern"


# ---------------------------------------------------------------------------
# confidence enum validation
# ---------------------------------------------------------------------------


def test_confidence_enum_valid_verified() -> None:
    """confidence='verified' is accepted."""
    entry = MemoryEntry(id="L-x", content="c", confidence="verified")
    assert entry.confidence == "verified"


def test_confidence_enum_valid_all() -> None:
    """All valid confidence values are accepted."""
    valid = ["unverified", "low", "medium", "high", "verified"]
    for v in valid:
        entry = MemoryEntry(id="L-x", content="c", confidence=v)
        assert entry.confidence == v


def test_confidence_enum_validation() -> None:
    """Invalid confidence raises ValidationError."""
    with pytest.raises(ValidationError, match="confidence must be one of"):
        MemoryEntry(id="L-x", content="c", confidence="maybe")


# ---------------------------------------------------------------------------
# protection_tier enum validation
# ---------------------------------------------------------------------------


def test_protection_tier_enum_valid_permanent() -> None:
    """protection_tier='permanent' is accepted."""
    entry = MemoryEntry(id="L-x", content="c", protection_tier="permanent")
    assert entry.protection_tier == "permanent"


def test_protection_tier_enum_valid_all() -> None:
    """All valid protection_tier values are accepted."""
    for tier in ["normal", "protected", "permanent"]:
        entry = MemoryEntry(id="L-x", content="c", protection_tier=tier)
        assert entry.protection_tier == tier


def test_protection_tier_enum_validation() -> None:
    """Invalid protection_tier raises ValidationError."""
    with pytest.raises(ValidationError, match="protection_tier must be one of"):
        MemoryEntry(id="L-x", content="c", protection_tier="ultra")


# ---------------------------------------------------------------------------
# nudge_line truncation
# ---------------------------------------------------------------------------


def test_nudge_line_within_80_unchanged() -> None:
    """nudge_line <= 80 chars passes through unchanged."""
    line_80 = "x" * 80
    entry = MemoryEntry(id="L-x", content="c", nudge_line=line_80)
    assert entry.nudge_line == line_80


def test_nudge_line_max_80_truncated() -> None:
    """nudge_line >80 chars with spaces gets truncated with ellipsis."""
    # 100-char string with a space at position 70
    prefix = "word " * 14  # 70 chars (5 * 14)
    suffix = "x" * 30  # extra 30 chars to push over 80
    long_line = prefix + suffix
    assert len(long_line) > 80
    entry = MemoryEntry(id="L-x", content="c", nudge_line=long_line)
    assert len(entry.nudge_line) <= 81  # up to 80 chars + ellipsis
    assert entry.nudge_line.endswith("\u2026")


def test_nudge_line_boundary_79() -> None:
    """79-char nudge_line passes unchanged (< 80 threshold)."""
    line = "a" * 79
    entry = MemoryEntry(id="L-x", content="c", nudge_line=line)
    assert entry.nudge_line == line
    assert len(entry.nudge_line) == 79


def test_nudge_line_boundary_80() -> None:
    """80-char nudge_line passes unchanged (exactly at threshold)."""
    line = "a" * 80
    entry = MemoryEntry(id="L-x", content="c", nudge_line=line)
    assert entry.nudge_line == line
    assert len(entry.nudge_line) == 80


def test_nudge_line_boundary_81_truncated() -> None:
    """81-char nudge_line without space gets hard-cut at 80."""
    line = "a" * 81
    entry = MemoryEntry(id="L-x", content="c", nudge_line=line)
    # No spaces in range [60,80), so hard-cut at 80
    assert entry.nudge_line == "a" * 80
    assert len(entry.nudge_line) == 80


def test_nudge_line_word_boundary_in_range() -> None:
    """Truncation at word boundary within [60, 80) produces ellipsis."""
    # Construct string: 65 chars of word, space at 65, then more chars to exceed 80
    line = "a" * 65 + " " + "b" * 20  # len = 86
    entry = MemoryEntry(id="L-x", content="c", nudge_line=line)
    # Should truncate at the space at position 65
    assert entry.nudge_line == "a" * 65 + "\u2026"


# ---------------------------------------------------------------------------
# domain list constraints
# ---------------------------------------------------------------------------


def test_domain_list_max_20_ok() -> None:
    """20 domain entries is accepted."""
    entry = MemoryEntry(id="L-x", content="c", domain=[f"d{i}" for i in range(20)])
    assert len(entry.domain) == 20


def test_domain_list_max_20_raises() -> None:
    """21 domain entries raises ValidationError."""
    with pytest.raises(ValidationError, match="domain may have at most 20 entries"):
        MemoryEntry(id="L-x", content="c", domain=[f"d{i}" for i in range(21)])


def test_domain_list_boundary_19() -> None:
    """19 domain entries is accepted."""
    entry = MemoryEntry(id="L-x", content="c", domain=[f"d{i}" for i in range(19)])
    assert len(entry.domain) == 19


def test_domain_list_boundary_20() -> None:
    """20 domain entries is accepted (boundary)."""
    entry = MemoryEntry(id="L-x", content="c", domain=[f"d{i}" for i in range(20)])
    assert len(entry.domain) == 20


def test_domain_list_boundary_21_raises() -> None:
    """21 domain entries raises ValidationError."""
    with pytest.raises(ValidationError):
        MemoryEntry(id="L-x", content="c", domain=[f"d{i}" for i in range(21)])


# ---------------------------------------------------------------------------
# phase_affinity list constraints
# ---------------------------------------------------------------------------


def test_phase_affinity_max_6_ok() -> None:
    """6 phase_affinity entries is accepted."""
    phases = ["RESEARCH", "PLAN", "IMPLEMENT", "VALIDATE", "REVIEW", "DELIVER"]
    entry = MemoryEntry(id="L-x", content="c", phase_affinity=phases)
    assert len(entry.phase_affinity) == 6


def test_phase_affinity_max_6_raises() -> None:
    """7 phase_affinity entries raises ValidationError."""
    phases = ["RESEARCH", "PLAN", "IMPLEMENT", "VALIDATE", "REVIEW", "DELIVER", "EXTRA"]
    with pytest.raises(ValidationError, match="phase_affinity may have at most 6 entries"):
        MemoryEntry(id="L-x", content="c", phase_affinity=phases)


def test_phase_affinity_boundary_5() -> None:
    """5 phase_affinity entries is accepted."""
    entry = MemoryEntry(id="L-x", content="c", phase_affinity=["A", "B", "C", "D", "E"])
    assert len(entry.phase_affinity) == 5


# ---------------------------------------------------------------------------
# to_dict() includes new fields
# ---------------------------------------------------------------------------


def test_to_dict_includes_new_fields() -> None:
    """to_dict() serializes all 10 new fields."""
    entry = MemoryEntry(
        id="L-x",
        content="test",
        type="incident",
        nudge_line="short line",
        expires="2026-12-31",
        confidence="high",
        task_type="debugging",
        domain=["auth", "api"],
        phase_origin="IMPLEMENT",
        phase_affinity=["IMPLEMENT", "VALIDATE"],
        team_origin="platform",
        protection_tier="protected",
    )
    d = entry.to_dict()
    assert d["type"] == "incident"
    assert d["nudge_line"] == "short line"
    assert d["expires"] == "2026-12-31"
    assert d["confidence"] == "high"
    assert d["task_type"] == "debugging"
    assert d["domain"] == ["auth", "api"]
    assert d["phase_origin"] == "IMPLEMENT"
    assert d["phase_affinity"] == ["IMPLEMENT", "VALIDATE"]
    assert d["team_origin"] == "platform"
    assert d["protection_tier"] == "protected"
