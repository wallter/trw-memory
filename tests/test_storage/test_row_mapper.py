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

    # Find expires_at column position in ENTRY_COLUMNS
    expires_idx = ENTRY_COLUMNS.index("expires_at")
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


def test_round_trip_session_count() -> None:
    """Dedicated session_count survives row round-trip independently of access_count."""
    entry = MemoryEntry(id="L-sess1", content="session count", access_count=9, session_count=4)
    row = _entry_to_full_row(entry)
    restored = row_to_entry(row)
    assert restored.access_count == 9
    assert restored.session_count == 4


def test_row_to_entry_defaults_missing_session_count_to_zero() -> None:
    """Older rows without session_count data load with a safe zero default."""
    entry = MemoryEntry(id="L-sess0", content="missing session count")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("session_count")] = None

    restored = row_to_entry(tuple(row))

    assert restored.session_count == 0


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


def test_anchor_validity_zero_survives_round_trip() -> None:
    """anchor_validity=0.0 (all anchors stale) must NOT resurrect to 1.0.

    Regression: the old ``float(str(v)) if v else 1.0`` falsy-check treated a
    legitimate persisted 0.0 as missing and read it back as 1.0 (fresh),
    inverting the staleness signal the lifecycle relies on.
    """
    entry = MemoryEntry(id="L-anc-zero", content="all anchors stale")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("anchor_validity")] = 0.0

    restored = row_to_entry(tuple(row))

    assert restored.anchor_validity == 0.0


# ---------------------------------------------------------------------------
# Corrupt-column resilience: a single bad JSON column must not crash row
# mapping for an entire query (fail-open, mirroring assertions handling).
# ---------------------------------------------------------------------------


def test_corrupt_anchor_validity_degrades_to_default() -> None:
    """A non-numeric anchor_validity column degrades to 1.0 instead of raising."""
    entry = MemoryEntry(id="L-anc-val-bad", content="corrupt validity")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("anchor_validity")] = "not-a-number"

    restored = row_to_entry(tuple(row))

    assert restored.anchor_validity == 1.0
    assert restored.id == "L-anc-val-bad"


def test_null_anchor_validity_uses_default() -> None:
    """A NULL anchor_validity column uses the 1.0 model default."""
    entry = MemoryEntry(id="L-anc-val-null", content="null validity")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("anchor_validity")] = None

    restored = row_to_entry(tuple(row))

    assert restored.anchor_validity == 1.0


def test_corrupt_anchors_json_degrades_to_empty() -> None:
    """A malformed anchors_json column maps to [] instead of raising."""
    entry = MemoryEntry(id="L-anc-bad", content="corrupt anchors")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("anchors")] = "{not valid json"

    restored = row_to_entry(tuple(row))

    assert restored.anchors == []
    assert restored.id == "L-anc-bad"


def test_anchors_json_non_list_degrades_to_empty() -> None:
    """A JSON object (non-list) in anchors_json degrades to [] rather than raising."""
    entry = MemoryEntry(id="L-anc-obj", content="anchors object")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("anchors")] = '{"file": "x.py"}'

    restored = row_to_entry(tuple(row))

    assert restored.anchors == []


def test_anchors_json_invalid_item_degrades_to_empty() -> None:
    """An anchor item failing model validation degrades the list to []."""
    entry = MemoryEntry(id="L-anc-inv", content="invalid anchor item")
    row = list(_entry_to_full_row(entry))
    # Missing required symbol_name -> Anchor validation fails.
    row[ENTRY_COLUMNS.index("anchors")] = '[{"file": "src/mod.py"}]'

    restored = row_to_entry(tuple(row))

    assert restored.anchors == []


def test_corrupt_assertions_json_degrades_to_empty() -> None:
    """A malformed assertions_json column maps to [] instead of raising."""
    entry = MemoryEntry(id="L-asr-bad", content="corrupt assertions")
    row = list(_entry_to_full_row(entry))
    row[ENTRY_COLUMNS.index("assertions")] = "not json at all"

    restored = row_to_entry(tuple(row))

    assert restored.assertions == []
