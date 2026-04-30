"""Shared helpers for the ``test_team_memory_*`` test family."""

from __future__ import annotations

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend


class _InMemoryBackend(StorageBackend):
    """Simple in-memory ``StorageBackend`` for team-memory tests."""

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
        for key, value in fields.items():
            if key == "status" and not isinstance(value, MemoryStatus):
                data[key] = MemoryStatus(str(value))
                continue
            data[key] = value
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
        del query, tags, min_importance, namespace
        results = list(self._data.values())
        if status is not None:
            status_value = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                entry
                for entry in results
                if (entry.status.value if isinstance(entry.status, MemoryStatus) else str(entry.status))
                == status_value
            ]
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return len(self._data)
        return sum(1 for entry in self._data.values() if entry.namespace == namespace)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            status_value = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                entry
                for entry in results
                if (entry.status.value if isinstance(entry.status, MemoryStatus) else str(entry.status))
                == status_value
            ]
        if namespace is not None:
            results = [entry for entry in results if entry.namespace == namespace]
        return results[:limit]

    def close(self) -> None:
        return None


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
