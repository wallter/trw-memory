"""Tests for trw_memory audit event models."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.models.events import MemoryEvent, MemoryEventType


def test_memory_event_type_values() -> None:
    assert MemoryEventType.STORE.value == "store"
    assert MemoryEventType.RECALL.value == "recall"
    assert MemoryEventType.UPDATE.value == "update"
    assert MemoryEventType.DELETE.value == "delete"
    assert MemoryEventType.CONSOLIDATE.value == "consolidate"
    assert MemoryEventType.MIGRATE.value == "migrate"
    assert MemoryEventType.TIER_PROMOTE.value == "tier_promote"
    assert MemoryEventType.TIER_DEMOTE.value == "tier_demote"
    assert MemoryEventType.TIER_PURGE.value == "tier_purge"


def test_memory_event_minimal_construction() -> None:
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
    evt = MemoryEvent(event_type=MemoryEventType.RECALL)
    data = evt.model_dump()
    assert data["event_type"] == "recall"
