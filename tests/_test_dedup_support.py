"""Shared helpers for split dedup tests."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.models.memory import MemoryEntry, MemoryStatus


def make_entry(
    entry_id: str,
    content: str,
    detail: str = "",
    tags: list[str] | None = None,
    evidence: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    recurrence: int = 1,
    merged_from: list[str] | None = None,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        evidence=evidence or [],
        importance=importance,
        status=status,
        recurrence=recurrence,
        merged_from=merged_from or [],
        created_at=now,
        updated_at=now,
    )


class StubEmbedder:
    """Minimal embedding provider stub with deterministic embeddings."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self._vectors: dict[str, list[float]] = {}

    def set_vector(self, text: str, vector: list[float]) -> None:
        self._vectors[text] = vector

    def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        if text in self._vectors:
            return self._vectors[text]
        return [float(ord(c)) / 128.0 for c in text[:3].ljust(3)]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return self._available

    def dim(self) -> int:
        return 3
