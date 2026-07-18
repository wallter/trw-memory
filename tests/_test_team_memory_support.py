"""Shared helpers for the ``test_team_memory_*`` test family."""

from __future__ import annotations

from trw_memory.models.memory import MemoryEntry

from ._test_memory_backend_support import _InMemoryBackend as _InMemoryBackend


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
