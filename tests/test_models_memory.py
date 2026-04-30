"""Tests for trw_memory memory entry and index models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus


def test_memory_status_values() -> None:
    assert MemoryStatus.ACTIVE.value == "active"
    assert MemoryStatus.RESOLVED.value == "resolved"
    assert MemoryStatus.OBSOLETE.value == "obsolete"
    assert MemoryStatus.ARCHIVED.value == "archived"


def test_memory_entry_minimal_construction() -> None:
    entry = MemoryEntry(id="M-001", content="Use absolute paths in WSL2")
    assert entry.id == "M-001"
    assert entry.content == "Use absolute paths in WSL2"
    assert entry.detail == ""
    assert entry.tags == []
    assert entry.evidence == []
    assert entry.importance == 0.5
    assert entry.status == MemoryStatus.ACTIVE
    assert entry.recurrence == 1
    assert entry.namespace == "default"
    assert entry.access_count == 0
    assert entry.q_value == 0.5
    assert entry.q_observations == 0
    assert entry.source == "agent"
    assert entry.source_identity == ""
    assert entry.merged_from == []
    assert entry.consolidated_from == []
    assert entry.consolidated_into is None
    assert entry.metadata == {}
    assert entry.last_accessed_at is None
    assert isinstance(entry.created_at, datetime)
    assert isinstance(entry.updated_at, datetime)
    assert entry.created_at.tzinfo is not None


def test_memory_entry_full_construction() -> None:
    now = datetime.now(timezone.utc)
    entry = MemoryEntry(
        id="M-002",
        content="Pydantic v2 use_enum_values required for YAML round-trip",
        detail="Without use_enum_values the enum object is stored, not the string",
        tags=["pydantic", "yaml"],
        evidence=["test_models.py::test_round_trip"],
        importance=0.9,
        status=MemoryStatus.RESOLVED,
        recurrence=5,
        namespace="project",
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
        access_count=3,
        q_value=0.8,
        q_observations=10,
        source="human",
        source_identity="tyler",
        merged_from=["M-000"],
        consolidated_from=["M-010", "M-011"],
        consolidated_into="M-050",
        metadata={"sprint": "31"},
    )
    assert entry.importance == 0.9
    assert entry.status == MemoryStatus.RESOLVED
    assert entry.tags == ["pydantic", "yaml"]
    assert entry.evidence == ["test_models.py::test_round_trip"]
    assert entry.recurrence == 5
    assert entry.namespace == "project"
    assert entry.access_count == 3
    assert entry.q_value == 0.8
    assert entry.q_observations == 10
    assert entry.source == "human"
    assert entry.source_identity == "tyler"
    assert entry.merged_from == ["M-000"]
    assert entry.consolidated_from == ["M-010", "M-011"]
    assert entry.consolidated_into == "M-050"
    assert entry.metadata == {"sprint": "31"}


def test_memory_entry_importance_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", importance=1.1)


def test_memory_entry_importance_below_min_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", importance=-0.1)


def test_memory_entry_importance_boundary_values_valid() -> None:
    low = MemoryEntry(id="M-low", content="x", importance=0.0)
    high = MemoryEntry(id="M-high", content="x", importance=1.0)
    assert low.importance == 0.0
    assert high.importance == 1.0


def test_memory_entry_negative_access_count_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", access_count=-1)


def test_memory_entry_negative_recurrence_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", recurrence=-1)


def test_memory_entry_coerces_string_importance() -> None:
    entry = MemoryEntry.model_validate({"id": "M-coerce", "content": "test", "importance": "0.5"})
    assert entry.importance == 0.5


def test_memory_entry_json_round_trip() -> None:
    original = MemoryEntry(
        id="M-rt",
        content="JSON round-trip check",
        tags=["roundtrip"],
        importance=0.75,
        status=MemoryStatus.ARCHIVED,
    )
    json_str = original.model_dump_json()
    parsed_dict = json.loads(json_str)
    assert parsed_dict["id"] == "M-rt"
    assert parsed_dict["importance"] == 0.75

    restored = MemoryEntry.model_validate_json(json_str)
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.importance == original.importance
    assert restored.status == original.status
    assert restored.tags == original.tags


def test_memory_entry_status_serialized_as_string() -> None:
    entry = MemoryEntry(id="M-s", content="x", status=MemoryStatus.OBSOLETE)
    data = entry.model_dump()
    assert data["status"] == "obsolete"


def test_memory_index_empty_defaults() -> None:
    idx = MemoryIndex()
    assert idx.entries == []
    assert idx.total_count == 0


def test_memory_index_with_entries() -> None:
    e1 = MemoryEntry(id="M-i1", content="first")
    e2 = MemoryEntry(id="M-i2", content="second")
    idx = MemoryIndex(entries=[e1, e2], total_count=2)
    assert len(idx.entries) == 2
    assert idx.total_count == 2
    assert idx.entries[0].id == "M-i1"
    assert idx.entries[1].id == "M-i2"
