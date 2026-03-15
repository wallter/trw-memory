"""Tests for team namespace promotion in tools/consolidate.py.

Tests the _promote_team_memories function: copying high-impact entries
from team namespaces to the project namespace.
"""

from __future__ import annotations

from datetime import datetime

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.consolidate import _promote_team_memories, memory_consolidate_impl

# ---------------------------------------------------------------------------
# In-memory backend for team promotion tests
# ---------------------------------------------------------------------------


class _InMemoryBackend(StorageBackend):
    """Simple in-memory StorageBackend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}

    def store(self, entry: MemoryEntry) -> None:
        self._data[entry.id] = entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._data.get(entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        existing = self._data.get(entry_id)
        if existing is None:
            return None
        data = existing.model_dump()
        for k, v in fields.items():
            if k == "status":
                if isinstance(v, MemoryStatus):
                    data[k] = v
                else:
                    data[k] = MemoryStatus(str(v))
            elif isinstance(v, datetime):
                data[k] = v
            else:
                data[k] = v
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MemoryStatus(data["status"])
        self._data[entry_id] = MemoryEntry(**data)
        return self._data[entry_id]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            return True
        return False

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                e for e in results if (e.status.value if isinstance(e.status, MemoryStatus) else str(e.status)) == sv
            ]
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        if namespace is not None:
            return sum(1 for e in self._data.values() if e.namespace == namespace)
        return len(self._data)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                e for e in results if (e.status.value if isinstance(e.status, MemoryStatus) else str(e.status)) == sv
            ]
        if namespace is not None:
            results = [e for e in results if e.namespace == namespace]
        return results[:limit]

    def close(self) -> None:
        pass


def _make_entry(
    entry_id: str,
    importance: float = 0.5,
    namespace: str = "team:sprint-37",
    tags: list[str] | None = None,
    outcome_history: list[str] | None = None,
    source_identity: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        importance=importance,
        namespace=namespace,
        tags=tags or [],
        outcome_history=outcome_history or [],
        source_identity=source_identity,
    )


# ===========================================================================
# Team Promotion Tests
# ===========================================================================


class TestPromoteTeamMemories:
    def test_copies_high_impact_entries_to_project_namespace(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))
        backend.store(_make_entry("e2", importance=0.9))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 2
        # Promoted entries should exist in project:default
        promoted_e1 = backend.get("promoted-e1")
        promoted_e2 = backend.get("promoted-e2")
        assert promoted_e1 is not None
        assert promoted_e2 is not None
        assert promoted_e1.namespace == "project:default"
        assert promoted_e2.namespace == "project:default"

    def test_skips_low_impact_entries(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.3))
        backend.store(_make_entry("e2", importance=0.5))
        backend.store(_make_entry("e3", importance=0.8))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 1
        assert result["discarded_count"] == 2
        # Only e3 should be promoted
        assert backend.get("promoted-e3") is not None
        assert backend.get("promoted-e1") is None
        assert backend.get("promoted-e2") is None

    def test_preserves_source_identity(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))

        _promote_team_memories("team:sprint-37", backend)

        promoted = backend.get("promoted-e1")
        assert promoted is not None
        assert promoted.source_identity == "team:sprint-37"

    def test_records_promoted_from_in_outcome_history(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8, outcome_history=["previous_event"]))

        _promote_team_memories("team:sprint-37", backend)

        promoted = backend.get("promoted-e1")
        assert promoted is not None
        assert len(promoted.outcome_history) == 2
        assert promoted.outcome_history[0] == "previous_event"
        assert "promoted_from:team:sprint-37" in promoted.outcome_history[1]

    def test_returns_correct_counts(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.9))
        backend.store(_make_entry("e2", importance=0.3))
        backend.store(_make_entry("e3", importance=0.8))
        backend.store(_make_entry("e4", importance=0.1))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 2  # e1 (0.9) + e3 (0.8)
        assert result["discarded_count"] == 2  # e2 (0.3) + e4 (0.1)
        assert result["namespace_id"] == "team:sprint-37"
        assert "completed_at" in result


class TestConsolidateImplTeamDispatch:
    def test_team_namespace_dispatches_to_promotion(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))

        result = memory_consolidate_impl("team:sprint-37", backend=backend)

        assert "promoted_count" in result
        assert result["promoted_count"] == 1

    def test_non_team_namespace_uses_normal_consolidation(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8, namespace="project:default"))

        result = memory_consolidate_impl("project:default", backend=backend)

        # Normal consolidation path (no embedder = no clusters)
        assert "clusters_found" in result or "entries_consolidated" in result

    def test_custom_promotion_threshold(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.5))
        backend.store(_make_entry("e2", importance=0.9))

        result = _promote_team_memories("team:sprint-37", backend, promotion_threshold=0.6)

        assert result["promoted_count"] == 1  # only e2 at 0.9
        assert result["discarded_count"] == 1  # e1 at 0.5
