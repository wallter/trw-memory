"""Shared lifecycle utilities — extracted from tiers.py and scoring.py."""

from __future__ import annotations

from datetime import date, datetime


def days_since_access(
    entry: dict[str, object],
    today: date,
    fallback_days: int = 30,
) -> int:
    """Compute days since last access from an entry dict.

    Resolution order: last_accessed_at -> created_at -> created -> fallback_days.

    Handles datetime objects (from model_dump), date objects, and ISO strings
    (from YAML/JSON). Sentinels ``"None"``, ``"null"``, and ``""`` are treated
    as missing values.

    Args:
        entry: Entry data dict (from YAML, JSON, or model_dump).
        today: Reference date for computing delta.
        fallback_days: Days to return when no date is parseable.

    Returns:
        Number of days since the entry was last accessed (>= 0).
    """
    for field in ("last_accessed_at", "created_at", "created"):
        val = entry.get(field)
        if val is None:
            continue
        # datetime object (from model_dump) — check before date since datetime is a date subclass
        if isinstance(val, datetime):
            return max(0, (today - val.date()).days)
        if isinstance(val, date):
            return max(0, (today - val).days)
        # String (from YAML/JSON)
        raw = str(val)
        if not raw or raw in ("None", "null"):
            continue
        try:
            if "T" in raw or " " in raw:
                dt = datetime.fromisoformat(raw)
                return max(0, (today - dt.date()).days)
            return max(0, (today - date.fromisoformat(raw)).days)
        except (ValueError, TypeError):
            continue
    return fallback_days
