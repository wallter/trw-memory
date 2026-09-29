"""PRD-CORE-312 FR01 — an evidence-level field, defaulted to Unknown.

Failing on the pre-FR01 commit: ``MemoryEntry`` had no ``evidence_level``
field at all (``grep -rn evidence_level`` across ``trw-memory/src`` returned
nothing), so every assertion below raised ``AttributeError`` or
``ValidationError`` on an unknown keyword.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import EvidenceLevel, MemoryEntry


def test_entry_with_no_evidence_level_defaults_to_unknown() -> None:
    entry = MemoryEntry(id="M-1", content="a claim")
    assert entry.evidence_level == EvidenceLevel.UNKNOWN
    assert entry.evidence_level != EvidenceLevel.VERIFIED


@pytest.mark.parametrize("level", ["observed", "verified", "inferred", "unknown"])
def test_entry_accepts_every_valid_evidence_level_string(level: str) -> None:
    entry = MemoryEntry(id="M-2", content="a claim", evidence_level=level)
    assert entry.evidence_level == EvidenceLevel(level)


def test_entry_rejects_an_invalid_evidence_level_string() -> None:
    with pytest.raises(ValidationError, match="evidence_level must be one of"):
        MemoryEntry(id="M-3", content="a claim", evidence_level="guessed")


def test_entry_accepts_an_evidence_level_enum_instance() -> None:
    entry = MemoryEntry(id="M-4", content="a claim", evidence_level=EvidenceLevel.OBSERVED)
    assert entry.evidence_level == EvidenceLevel.OBSERVED


def test_pre_migration_row_with_no_evidence_level_key_loads_unchanged() -> None:
    """A 4.0-era serialized row with no ``evidence_level`` key loads as unknown, not an error."""
    legacy_payload = {
        "id": "M-legacy",
        "content": "a pre-FR01 row",
        "tags": ["gotcha"],
        "importance": 0.6,
        "confidence": "high",
    }
    assert "evidence_level" not in legacy_payload
    entry = MemoryEntry.model_validate(legacy_payload)
    assert entry.evidence_level == EvidenceLevel.UNKNOWN


def test_evidence_level_is_a_distinct_axis_from_confidence() -> None:
    """Promoting confidence does not silently promote evidence_level (and vice versa)."""
    entry = MemoryEntry(id="M-5", content="a claim", confidence="high")
    assert entry.evidence_level == EvidenceLevel.UNKNOWN
