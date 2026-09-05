"""Reusable in-memory storage backend for memory test families."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from datetime import datetime

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend


class _InMemoryBackend(StorageBackend):
    """In-memory backend with transactions, vectors, namespaces, and failure hooks."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, list[float]] = {}
        self.update_override: Callable[[str, dict[str, object]], MemoryEntry | None] | None = None
        self.delete_override: Callable[[str], bool] | None = None
        self.upsert_vector_override: Callable[[str, list[float]], None] | None = None

    @contextlib.contextmanager
    def transaction(self) -> Iterator[StorageBackend]:
        data_snapshot = dict(self._data)
        vectors_snapshot = dict(self._vectors)
        try:
            yield self
        except Exception:
            self._data = data_snapshot
            self._vectors = vectors_snapshot
            raise

    def store(self, entry: MemoryEntry) -> None:
        self._data[entry.id] = entry

    def get(self, entry_id: str, *, namespace: str = "default") -> MemoryEntry | None:
        return self._data.get(entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        if self.update_override is not None:
            return self.update_override(entry_id, fields)
        existing = self._data.get(entry_id)
        if existing is None:
            return None
        data = existing.model_dump()
        for key, value in fields.items():
            if key == "status" and not isinstance(value, MemoryStatus):
                data[key] = MemoryStatus(str(value))
            elif isinstance(value, datetime):
                data[key] = value
            else:
                data[key] = value
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MemoryStatus(data["status"])
        self._data[entry_id] = MemoryEntry(**data)
        return self._data[entry_id]

    def delete(self, entry_id: str, *, namespace: str = "default") -> bool:
        if self.delete_override is not None:
            return self.delete_override(entry_id)
        if entry_id not in self._data:
            return False
        del self._data[entry_id]
        self._vectors.pop(entry_id, None)
        return True

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
        del query, tags, min_importance
        results = list(self._data.values())
        if status is not None:
            status_value = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                entry
                for entry in results
                if (entry.status.value if isinstance(entry.status, MemoryStatus) else str(entry.status)) == status_value
            ]
        if namespace is not None:
            results = [entry for entry in results if entry.namespace == namespace]
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        return len(self.list_entries(namespace=namespace, limit=len(self._data)))

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            status_value = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                entry
                for entry in results
                if (entry.status.value if isinstance(entry.status, MemoryStatus) else str(entry.status)) == status_value
            ]
        if namespace is not None:
            results = [entry for entry in results if entry.namespace == namespace]
        if min_importance > 0.0:
            results = [entry for entry in results if entry.importance >= min_importance]
        return results[:limit]

    def close(self) -> None:
        return None

    def supports_vectors(self) -> bool:
        return True

    def upsert_vector(self, entry_id: str, embedding: list[float], *, namespace: str = "default") -> None:
        if self.upsert_vector_override is not None:
            self.upsert_vector_override(entry_id, embedding)
            return
        self._vectors[entry_id] = embedding

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        return {entry_id: self._vectors[entry_id] for entry_id in entry_ids if entry_id in self._vectors}
