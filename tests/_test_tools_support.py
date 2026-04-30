from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory.models.memory import MemoryEntry, MemoryStatus


def _make_entry(
    entry_id: str = "M-001",
    content: str = "test memory",
    namespace: str = "project:default",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    tags: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        namespace=namespace,
        status=status,
        tags=tags or [],
    )


def _mock_backend(entries: list[MemoryEntry] | None = None) -> MagicMock:
    """Create a mock StorageBackend with sensible defaults."""
    backend = MagicMock()
    entries = entries or []
    backend.list_entries.return_value = entries
    backend.get_stored_embeddings.return_value = {}
    backend.search.return_value = entries
    backend.get.return_value = entries[0] if entries else None
    backend.delete.return_value = True
    backend.count.return_value = len(entries)
    return backend
