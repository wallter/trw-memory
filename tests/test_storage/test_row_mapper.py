"""Tests for row mapper with 10 new typed learning fields (PRD-CORE-110).

Covers:
- Round-trip for entry with type='incident'
- Round-trip for entry with domain list
- expires field maps to/from expires_at column position
- Old hex-format IDs (8-char) still load correctly
"""

from __future__ import annotations

from trw_memory.models.memory import Confidence, MemoryEntry, MemoryType, ProtectionTier
from trw_memory.storage._row_mapper import entry_to_row, row_to_entry
from trw_memory.storage._shared import ENTRY_COLUMNS

# ---------------------------------------------------------------------------
# Helper: build minimal valid row tuple from entry
# ---------------------------------------------------------------------------


def _entry_to_full_row(entry: MemoryEntry) -> tuple[object, ...]:
    """Convert entry to row and verify length matches ENTRY_COLUMNS."""
    row = entry_to_row(entry)
    assert len(row) == len(ENTRY_COLUMNS), f"Row length {len(row)} != ENTRY_COLUMNS length {len(ENTRY_COLUMNS)}"
    return row


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_round_trip_typed_entry() -> None:
    """Entry with type='incident' survives row round-trip."""
    entry = MemoryEntry(
        id="L-rnd1",
        content="incident test",
        type=MemoryType.INCIDENT,
        confidence=Confidence.VERIFIED,
        protection_tier=ProtectionTier.NORMAL,
    )
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)

    assert restored.id == "L-rnd1"
    assert restored.type == MemoryType.INCIDENT
    assert restored.confidence == Confidence.VERIFIED
    assert restored.protection_tier == ProtectionTier.NORMAL


def test_round_trip_with_domain_list() -> None:
    """domain list survives JSON round-trip through row mapper."""
    entry = MemoryEntry(
        id="L-rnd2",
        content="domain test",
        domain=["auth", "api"],
        phase_affinity=["IMPLEMENT", "VALIDATE"],
    )
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)

    assert restored.domain == ["auth", "api"]
    assert restored.phase_affinity == ["IMPLEMENT", "VALIDATE"]


def test_expires_at_column_mapping() -> None:
    """entry.expires='2026-12-31' maps to expires_at column, survives round-trip."""
    entry = MemoryEntry(id="L-rnd3", content="expires test", expires="2026-12-31")
    row = _entry_to_full_row(entry)

    # Find expires column position in ENTRY_COLUMNS
    expires_idx = ENTRY_COLUMNS.index("expires")
    assert row[expires_idx] == "2026-12-31", f"Expected '2026-12-31' at index {expires_idx}, got {row[expires_idx]!r}"

    restored = row_to_entry(row)
    assert restored.expires == "2026-12-31"


def test_old_hex_ids_still_load() -> None:
    """Entry with old-style 8-char hex ID loads correctly."""
    entry = MemoryEntry(id="L-a1b2c3d4", content="old id test")
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)

    assert restored.id == "L-a1b2c3d4"
    assert restored.type == MemoryType.PATTERN  # default
    assert restored.confidence == Confidence.UNVERIFIED  # default
    assert restored.protection_tier == ProtectionTier.NORMAL  # default


def test_row_length_matches_entry_columns() -> None:
    """entry_to_row always produces a tuple with len == len(ENTRY_COLUMNS)."""
    entry = MemoryEntry(id="L-len1", content="length check")
    row = entry_to_row(entry)
    assert len(row) == len(ENTRY_COLUMNS)


def test_round_trip_all_defaults() -> None:
    """Default entry round-trips cleanly through row mapper."""
    entry = MemoryEntry(id="L-def1", content="defaults test")
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)

    assert restored.type == MemoryType.PATTERN
    assert restored.nudge_line == ""
    assert restored.expires == ""
    assert restored.confidence == Confidence.UNVERIFIED
    assert restored.task_type == ""
    assert restored.domain == []
    assert restored.phase_origin == ""
    assert restored.phase_affinity == []
    assert restored.team_origin == ""
    assert restored.protection_tier == ProtectionTier.NORMAL


def test_round_trip_nudge_line() -> None:
    """nudge_line value survives row round-trip."""
    entry = MemoryEntry(id="L-nud1", content="nudge test", nudge_line="Short nudge")
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)
    assert restored.nudge_line == "Short nudge"


def test_round_trip_team_origin() -> None:
    """team_origin and task_type survive round-trip."""
    entry = MemoryEntry(
        id="L-team1",
        content="team test",
        team_origin="backend-team",
        task_type="debugging",
        phase_origin="IMPLEMENT",
    )
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)
    assert restored.team_origin == "backend-team"
    assert restored.task_type == "debugging"
    assert restored.phase_origin == "IMPLEMENT"


# ---------------------------------------------------------------------------
# Anchor fields round-trip tests (PRD-CORE-111)
# ---------------------------------------------------------------------------


def test_anchor_empty_list_default() -> None:
    """Entry with no anchors has anchors=[] and anchor_validity=1.0."""
    entry = MemoryEntry(id="L-anc-def", content="anchor defaults")
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)
    assert restored.anchors == []
    assert restored.anchor_validity == 1.0


def test_anchor_json_round_trip() -> None:
    """Entry with anchors survives row mapper round-trip."""
    from trw_memory.models.memory import Anchor

    anchors = [
        Anchor(file="src/mod.py", symbol_name="my_func", symbol_type="function"),
        Anchor(file="src/cls.py", symbol_name="MyClass", symbol_type="class", signature="class MyClass:"),
    ]
    entry = MemoryEntry(
        id="L-anc-rt",
        content="anchor round-trip",
        anchors=anchors,
        anchor_validity=0.5,
    )
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)

    assert len(restored.anchors) == 2
    assert restored.anchors[0].file == "src/mod.py"
    assert restored.anchors[0].symbol_name == "my_func"
    assert restored.anchors[0].symbol_type == "function"
    assert restored.anchors[1].file == "src/cls.py"
    assert restored.anchors[1].symbol_name == "MyClass"
    assert restored.anchors[1].signature == "class MyClass:"
    assert restored.anchor_validity == 0.5
