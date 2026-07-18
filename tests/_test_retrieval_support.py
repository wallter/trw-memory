"""Shared helpers for split retrieval tests."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry, MemoryStatus

from ._test_embedding_support import StubEmbedder as StubEmbedder


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
