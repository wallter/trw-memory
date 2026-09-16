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
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        created_at=created_at or datetime.now(timezone.utc),
        metadata=metadata or {},
        tags=tags or [],
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


def rejections_for(
    entry: MemoryEntry, *, max_chars: int = 10_240, min_evidence_items_for_verified: int = 1
) -> list[str]:
    """Return the validator's rejection reasons for ``entry`` — empty when accepted.

    ``validate_entry_payload`` returns ``None`` and signals rejection by raising,
    so "accepted" has no positive value to compare against, and the acceptance
    tests were all written as a bare call asserting nothing. That shape survives a
    total inversion of the validator: if every input were suddenly accepted, only
    the rejection tests would notice — and they are the half that an over-eager
    NARROWING does not break, which is the direction this file's history actually
    moved in (the 2026-07-27 window narrowing).

    Turning the raise into a returned list lets each test state the claim as
    ``assert rejections_for(...) == []``, so the assertion lives where a reader
    looks for it and the failure message names the reason instead of a traceback.
    """
    from trw_memory.exceptions import PoisoningError, SchemaValidationError
    from trw_memory.security.poisoning import validate_entry_payload

    try:
        validate_entry_payload(
            entry, max_chars=max_chars, min_evidence_items_for_verified=min_evidence_items_for_verified
        )
    except (PoisoningError, SchemaValidationError) as exc:
        return [f"{type(exc).__name__}(reason={getattr(exc, 'reason', None)!r})"]
    return []
