"""Cold archive-record representation: partition timestamps and private vector payloads."""

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


_WARM_EMBEDDING_KEY = "_warm_embedding"


def extract_archived_embedding(entry_data: dict[str, object]) -> list[float] | None:
    raw_embedding = entry_data.pop(_WARM_EMBEDDING_KEY, None)
    if not isinstance(raw_embedding, list):
        return None
    values: list[float] = []
    for value in raw_embedding:
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values


def sanitize_archived_entry(entry_data: dict[str, object]) -> dict[str, object]:
    sanitized = dict(entry_data)
    sanitized.pop(_WARM_EMBEDDING_KEY, None)
    return sanitized


def archive_payload(
    entry_data: dict[str, object],
    embedding: list[float] | None,
) -> tuple[dict[str, object], list[float] | None]:
    """Copy an archive record and attach its private restoration vector."""
    archive_data = dict(entry_data)
    if embedding is not None:
        archive_data[_WARM_EMBEDDING_KEY] = embedding
    return archive_data, embedding
