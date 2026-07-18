"""Partition timestamp parsing for cold-tier archives."""

import contextlib
from datetime import datetime, timezone


def entry_partition_timestamp(entry_data: dict[str, object]) -> datetime | None:
    """Return the UTC creation timestamp used for archive partitioning."""
    raw = entry_data.get("created_at", entry_data.get("created"))
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc)
    if isinstance(raw, str) and raw:
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    return None
