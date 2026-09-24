"""Shared helpers for split dedup tests."""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.models.memory import Confidence, MemoryEntry, MemoryStatus, MemoryType

from ._test_embedding_support import StubEmbedder as StubEmbedder


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
    confidence: Confidence = Confidence.UNVERIFIED,
    type: MemoryType = MemoryType.PATTERN,
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
        confidence=confidence,
        type=type,
        created_at=now,
        updated_at=now,
    )
