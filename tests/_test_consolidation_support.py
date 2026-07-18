"""Shared helpers for the ``test_consolidation*`` test family."""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

from trw_memory.models.memory import MemoryEntry, MemoryStatus

from ._test_memory_backend_support import _InMemoryBackend as _InMemoryBackend


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
