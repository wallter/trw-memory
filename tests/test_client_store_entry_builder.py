"""Shared single/bulk store entry construction semantics."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory._client_store import _build_store_entry
from trw_memory.models.memory import Assertion, MemoryEntry


def test_build_store_entry_populates_new_entry_fields() -> None:
    now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assertion = Assertion(type="grep_present", pattern="needle", target="file.py")
    entry = _build_store_entry(
        memory_id="M-new",
        existing=None,
        content="  content  ",
        detail="detail",
        tags=["tag"],
        evidence=["proof"],
        importance=0.8,
        namespace="project:test",
        metadata={"key": "value"},
        expires="2099-01-01",
        assertions=[assertion],
        source="agent",
        source_identity="worker",
        now=now,
        installation_id="install-1",
        local_node_id="node-1",
    )

    assert entry.content == "content"
    assert entry.detail == "detail"
    assert entry.tags == ["tag"]
    assert entry.importance == 0.8
    assert entry.namespace == "project:test"
    assert entry.metadata == {"key": "value", "installation_id": "install-1"}
    assert entry.expires == "2099-01-01"
    assert entry.evidence == ["proof"]
    assert entry.assertions == [assertion]
    assert entry.source == "agent"
    assert entry.source_identity == "worker"
    assert entry.vector_clock == {"node-1": 1}
    assert entry.created_at == now == entry.updated_at


def test_build_store_entry_preserves_omitted_update_fields_and_advances_clock() -> None:
    existing = MemoryEntry(
        id="M-existing",
        content="old",
        evidence=["old-proof"],
        assertions=[Assertion(type="grep_present", pattern="old", target="old.py")],
        metadata={"old": "value", "installation_id": "original-install"},
        expires="2090-01-01",
        source_identity="original-worker",
        vector_clock={"node-1": 2},
    )
    updated = _build_store_entry(
        memory_id=existing.id,
        existing=existing,
        content="new",
        detail="new detail",
        tags=None,
        evidence=None,
        importance=0.6,
        namespace=existing.namespace,
        metadata={"new": "value"},
        expires="",
        assertions=None,
        source="agent",
        source_identity="",
        now=datetime.now(timezone.utc),
        installation_id="ignored-install",
        local_node_id="node-1",
    )

    assert updated.evidence == existing.evidence
    assert updated.assertions == existing.assertions
    assert updated.expires == existing.expires
    assert updated.source_identity == existing.source_identity
    assert updated.metadata == {"old": "value", "new": "value", "installation_id": "original-install"}
    assert updated.vector_clock == {"node-1": 3}


def test_build_store_entry_applies_explicit_update_overrides() -> None:
    existing = MemoryEntry(
        id="M-existing",
        content="old",
        evidence=["old-proof"],
        assertions=[Assertion(type="grep_present", pattern="old", target="old.py")],
        expires="2090-01-01",
        source_identity="original-worker",
        vector_clock={"node-1": 1},
    )
    updated = _build_store_entry(
        memory_id=existing.id,
        existing=existing,
        content="new",
        detail="",
        tags=[],
        evidence=[],
        importance=0.5,
        namespace=existing.namespace,
        metadata=None,
        expires="2099-01-01",
        assertions=[],
        source="tool",
        source_identity="replacement-worker",
        now=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        installation_id="install-1",
        local_node_id="node-1",
    )

    assert updated.evidence == []
    assert updated.assertions == []
    assert updated.expires == "2099-01-01"
    assert updated.source == "tool"
    assert updated.source_identity == "replacement-worker"
