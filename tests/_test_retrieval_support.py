"""Shared helpers for split retrieval tests."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry, MemoryStatus


def make_entry(
    entry_id: str,
    content: str,
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        status=status,
        created_at=now,
        updated_at=now,
    )


class StubEmbedder:
    """Minimal EmbeddingProvider stub that uses the first 3 chars as a vector."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        return [float(ord(c)) / 128.0 for c in text[:3].ljust(3)]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return self._available

    def dim(self) -> int:
        return 3


def stored_embeddings_for(
    entry_ids: list[str],
    embedder: EmbeddingProvider,
) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    for entry_id in entry_ids:
        vector = embedder.embed(entry_id)
        if vector is not None:
            embeddings[entry_id] = vector
    return embeddings
