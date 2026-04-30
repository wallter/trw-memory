"""Shared test helpers for poisoning-related test shards."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from trw_memory.models.memory import MemoryEntry


def make_entry(
    entry_id: str = "M-001",
    content: str = "Normal content",
    detail: str = "",
    created_at: datetime | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        created_at=created_at or datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def make_entries_spread(
    count: int,
    interval_minutes: int = 120,
    content: str = "Normal content",
) -> list[MemoryEntry]:
    """Create *count* entries spread evenly across time."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        make_entry(
            entry_id=f"M-{index:03d}",
            content=f"{content} #{index}",
            created_at=base + timedelta(minutes=index * interval_minutes),
        )
        for index in range(count)
    ]


def serialized_size(entry: MemoryEntry) -> int:
    return len(
        json.dumps(entry.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
