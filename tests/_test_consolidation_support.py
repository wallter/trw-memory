"""Shared helpers for the ``test_consolidation*`` test family."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Literal
from unittest.mock import MagicMock

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend


def _make_entry(
    entry_id: str,
    content: str = "content",
    detail: str = "detail",
    importance: float = 0.5,
    tags: list[str] | None = None,
    evidence: list[str] | None = None,
    recurrence: int = 1,
    q_value: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    source: Literal["human", "agent", "tool", "consolidated"] = "agent",
    consolidated_into: str | None = None,
    namespace: str = "default",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        importance=importance,
        tags=tags or [],
        evidence=evidence or [],
        recurrence=recurrence,
        q_value=q_value,
        status=status,
        source=source,
        consolidated_into=consolidated_into,
        namespace=namespace,
    )


class _InMemoryBackend(StorageBackend):
    """Simple in-memory ``StorageBackend`` for testing."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, list[float]] = {}
        self.update_override: Callable[[str, dict[str, object]], MemoryEntry | None] | None = None
        self.delete_override: Callable[[str], bool] | None = None
        self.upsert_vector_override: Callable[[str, list[float]], None] | None = None

    @contextlib.contextmanager
    def transaction(self) -> Iterator[StorageBackend]:
        """Atomic batch with rollback, modelling SQLite's ``transaction()``.

        Snapshots the in-memory maps on entry and restores them if the block
        raises, so tests assert the real production invariant (row + vector land
        atomically; a vector failure leaves neither) rather than a
        compensating-delete artifact.
        """
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

    def get(self, entry_id: str) -> MemoryEntry | None:
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

    def delete(self, entry_id: str) -> bool:
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
            results = [entry for entry in results if str(entry.status) == status_value]
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
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            status_value = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [entry for entry in results if str(entry.status) == status_value]
        if namespace is not None:
            results = [entry for entry in results if entry.namespace == namespace]
        return results[:limit]

    def close(self) -> None:
        return None

    def supports_vectors(self) -> bool:
        # This double persists and serves vectors via ``self._vectors``, so it
        # must honestly report vector capability — store paths now consult this
        # before paying for an embedding (see ``embedding_has_consumer``).
        return True

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        if self.upsert_vector_override is not None:
            self.upsert_vector_override(entry_id, embedding)
            return
        self._vectors[entry_id] = embedding

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        return {entry_id: self._vectors[entry_id] for entry_id in entry_ids if entry_id in self._vectors}


def _make_embedder(
    dim: int = 4,
    available: bool = True,
    vectors: list[list[float] | None] | None = None,
) -> MagicMock:
    """Create a mock embedding provider."""
    embedder = MagicMock()
    embedder.available.return_value = available
    embedder.dim.return_value = dim
    if vectors is None:
        embedder.embed_batch.return_value = []
        embedder.embed.return_value = None
        return embedder
    embedder.embed_batch.return_value = vectors
    embedder.embed.return_value = vectors[0] if vectors else None
    return embedder


_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.99, 0.1, 0.0, 0.0]
_V3 = [0.98, 0.15, 0.0, 0.0]
_W1 = [0.0, 1.0, 0.0, 0.0]
_W2 = [0.1, 0.99, 0.0, 0.0]
_W3 = [0.15, 0.98, 0.0, 0.0]
_V_OUTLIER = [0.0, 0.0, 0.0, 1.0]
