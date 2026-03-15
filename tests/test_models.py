"""Tests for trw_memory models.

Covers:
- MemoryEntry: default construction, full construction, field validation,
  JSON round-trip, strict mode enforcement
- MemoryStatus: enum values
- MemoryIndex: construction with entries
- MemoryConfig: defaults, env var override, storage_backend literal validation
- MemoryEvent: construction
- MemoryEventType: enum values
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trw_memory.models.config import MemoryConfig
from trw_memory.models.events import MemoryEvent, MemoryEventType
from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus

# ---------------------------------------------------------------------------
# MemoryStatus
# ---------------------------------------------------------------------------


def test_memory_status_values() -> None:
    assert MemoryStatus.ACTIVE == "active"
    assert MemoryStatus.RESOLVED == "resolved"
    assert MemoryStatus.OBSOLETE == "obsolete"
    assert MemoryStatus.ARCHIVED == "archived"


# ---------------------------------------------------------------------------
# MemoryEntry — construction
# ---------------------------------------------------------------------------


def test_memory_entry_minimal_construction() -> None:
    """Minimal required fields: id + content; all others use defaults."""
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
    # Timestamps are set automatically
    assert isinstance(entry.created_at, datetime)
    assert isinstance(entry.updated_at, datetime)
    assert entry.created_at.tzinfo is not None


def test_memory_entry_full_construction() -> None:
    """All fields can be set explicitly."""
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


# ---------------------------------------------------------------------------
# MemoryEntry — field validation
# ---------------------------------------------------------------------------


def test_memory_entry_importance_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", importance=1.1)


def test_memory_entry_importance_below_min_raises() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-bad", content="test", importance=-0.1)


def test_memory_entry_importance_boundary_values_valid() -> None:
    """0.0 and 1.0 are both valid boundaries."""
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


# ---------------------------------------------------------------------------
# MemoryEntry — strict mode
# ---------------------------------------------------------------------------


def test_memory_entry_strict_mode_string_importance_raises() -> None:
    """strict=True means passing "0.5" (str) for a float field must fail."""
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-strict", content="test", importance="0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MemoryEntry — JSON round-trip
# ---------------------------------------------------------------------------


def test_memory_entry_json_round_trip() -> None:
    """model_dump_json() -> model_validate_json() must produce an equal entry."""
    original = MemoryEntry(
        id="M-rt",
        content="JSON round-trip check",
        tags=["roundtrip"],
        importance=0.75,
        status=MemoryStatus.ARCHIVED,
    )
    json_str = original.model_dump_json()
    # Sanity: the JSON is a valid JSON string
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
    """use_enum_values=True means status is stored as its string value."""
    entry = MemoryEntry(id="M-s", content="x", status=MemoryStatus.OBSOLETE)
    data = entry.model_dump()
    # With use_enum_values=True the dumped value is the raw string
    assert data["status"] == "obsolete"


# ---------------------------------------------------------------------------
# MemoryIndex
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MemoryConfig — defaults
# ---------------------------------------------------------------------------


def test_memory_config_defaults() -> None:
    cfg = MemoryConfig()
    assert cfg.storage_backend == "sqlite"
    assert cfg.storage_path == ".memory"
    assert cfg.sqlite_db_name == "memory.db"
    assert cfg.embedding_dim == 384
    assert cfg.bm25_candidates == 50
    assert cfg.vector_candidates == 50
    assert cfg.rrf_k == 60
    assert cfg.dedup_enabled is True
    assert cfg.hot_max_entries == 50
    assert cfg.decay_half_life_days == 14.0
    assert cfg.q_learning_rate == 0.15
    assert cfg.consolidation_enabled is True


def test_memory_config_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEMORY_STORAGE_BACKEND env var should override the default."""
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    cfg = MemoryConfig()
    assert cfg.storage_backend == "yaml"


def test_memory_config_storage_backend_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """storage_backend only accepts 'sqlite' or 'yaml'."""
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "postgres")
    with pytest.raises(ValidationError):
        MemoryConfig()


def test_memory_config_env_var_numeric_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numeric env vars are coerced from string by pydantic-settings."""
    monkeypatch.setenv("MEMORY_BM25_CANDIDATES", "100")
    cfg = MemoryConfig()
    assert cfg.bm25_candidates == 100


# ---------------------------------------------------------------------------
# MemoryEventType
# ---------------------------------------------------------------------------


def test_memory_event_type_values() -> None:
    assert MemoryEventType.STORE == "store"
    assert MemoryEventType.RECALL == "recall"
    assert MemoryEventType.UPDATE == "update"
    assert MemoryEventType.DELETE == "delete"
    assert MemoryEventType.CONSOLIDATE == "consolidate"
    assert MemoryEventType.MIGRATE == "migrate"
    assert MemoryEventType.TIER_PROMOTE == "tier_promote"
    assert MemoryEventType.TIER_DEMOTE == "tier_demote"
    assert MemoryEventType.TIER_PURGE == "tier_purge"


# ---------------------------------------------------------------------------
# MemoryEvent — construction
# ---------------------------------------------------------------------------


def test_memory_event_minimal_construction() -> None:
    """event_type is required; all other fields have defaults."""
    evt = MemoryEvent(event_type=MemoryEventType.STORE)
    assert evt.event_type == MemoryEventType.STORE
    assert evt.memory_id == ""
    assert evt.namespace == "default"
    assert evt.actor == ""
    assert evt.detail == {}
    assert isinstance(evt.timestamp, datetime)
    assert evt.timestamp.tzinfo is not None


def test_memory_event_full_construction() -> None:
    now = datetime.now(timezone.utc)
    evt = MemoryEvent(
        timestamp=now,
        event_type=MemoryEventType.MIGRATE,
        memory_id="M-001",
        namespace="project",
        actor="trw-tester",
        detail={"source": "trw-mcp", "count": "5"},
    )
    assert evt.memory_id == "M-001"
    assert evt.namespace == "project"
    assert evt.actor == "trw-tester"
    assert evt.detail == {"source": "trw-mcp", "count": "5"}
    assert evt.timestamp == now


def test_memory_event_type_serialized_as_string() -> None:
    """use_enum_values=True means event_type is stored as its string value."""
    evt = MemoryEvent(event_type=MemoryEventType.RECALL)
    data = evt.model_dump()
    assert data["event_type"] == "recall"


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for trw_memory.exceptions hierarchy."""

    def test_memory_error_message_and_path(self) -> None:
        from trw_memory.exceptions import MemoryError as MemErr

        err = MemErr("test msg", path="/some/path")
        assert str(err) == "test msg"
        assert err.path == "/some/path"

    def test_memory_error_default_path_empty(self) -> None:
        from trw_memory.exceptions import MemoryError as MemErr

        err = MemErr("test")
        assert err.path == ""

    def test_storage_error_inherits_memory_error(self) -> None:
        from trw_memory.exceptions import MemoryError as MemErr
        from trw_memory.exceptions import StorageError

        err = StorageError("storage fail", path="/db")
        assert isinstance(err, MemErr)
        assert err.path == "/db"

    def test_config_error_inherits_memory_error(self) -> None:
        from trw_memory.exceptions import ConfigError
        from trw_memory.exceptions import MemoryError as MemErr

        err = ConfigError("config fail")
        assert isinstance(err, MemErr)
