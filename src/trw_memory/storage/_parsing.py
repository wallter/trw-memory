"""Shared parsing utilities for storage backends.

Centralises datetime and JSON-field parsing so that
:mod:`sqlite_backend` and :mod:`yaml_backend` use a single implementation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def parse_dt(val: object) -> datetime:
    """Parse an ISO-8601 value to a timezone-aware UTC datetime.

    Accepts ``datetime`` instances and strings.  Naïve datetimes (no tzinfo)
    are assumed UTC; tz-aware ones are normalised to UTC.

    >>> parse_dt("2024-01-15T10:30:00")
    datetime.datetime(2024, 1, 15, 10, 30, tzinfo=datetime.timezone.utc)
    """
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    dt = datetime.fromisoformat(str(val))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dt_safe(val: object, *, default: datetime | None) -> datetime | None:
    """Fail-open variant of :func:`parse_dt` for read paths.

    Returns *default* (rather than raising ``ValueError``) when *val* is a
    malformed ISO string. Used by row/dict mappers so a single corrupt
    timestamp degrades one field instead of crashing the whole listing.

    The 2026-06-10 corruption class motivated this: a SQLite 3.51.1 WAL-reset
    byte-shift produced ``'026-04-13T00:00:00+00:002'`` (leading ``2`` lost,
    stray trailing ``2``), and the unguarded ``parse_dt`` in the mappers
    raised ``Invalid isoformat string``, taking down ``list_entries`` for the
    whole store. Degrading the bad field is consistent with the fail-open
    contract the same mappers already apply to status / anchors / ints /
    floats / JSON.

    >>> parse_dt_safe("2024-01-15T10:30:00", default=None) is not None
    True
    >>> parse_dt_safe("026-04-13T00:00:00+00:002", default=None) is None
    True
    """
    try:
        return parse_dt(val)
    except (ValueError, TypeError):
        return default


def parse_optional_float(raw: object) -> float | None:
    """Coerce a persisted value to ``float``, or ``None`` when absent/unparseable.

    For columns where SQL ``NULL`` carries meaning of its own — ``anchor_validity``
    reads ``None`` as "never assessed" (PRD-CORE-244-FR01), which is a different statement from any
    score including ``0.0``. A legitimately falsy ``0.0`` survives.

    >>> parse_optional_float(0.0)
    0.0
    >>> parse_optional_float(None) is None
    True
    >>> parse_optional_float("nope") is None
    True
    """
    if raw is None:
        return None
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return None


def parse_json_list(raw: object, *, fallback: list[str] | None = None) -> list[str]:
    """Deserialise a JSON-encoded list, or return *fallback* on failure.

    Handles ``None``, empty strings, and already-parsed lists.

    >>> parse_json_list('["a","b"]')
    ['a', 'b']
    >>> parse_json_list(None)
    []
    """
    if fallback is None:
        fallback = []
    if not raw:
        return fallback
    if isinstance(raw, list):
        return [str(v) for v in raw]
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return fallback


def parse_json_dict_str(raw: object) -> dict[str, str]:
    """Deserialise a JSON-encoded ``{str: str}`` dict, or return ``{}``.

    >>> parse_json_dict_str('{"k": "v"}')
    {'k': 'v'}
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def parse_json_dict_int(raw: object) -> dict[str, int]:
    """Deserialise a JSON-encoded ``{str: int}`` dict, or return ``{}``.

    Accepts a JSON string (SQLite TEXT column) or an already-parsed dict (YAML
    secondary store). Both branches route the ``int(v)`` coercion through one
    guarded conversion so a malformed value (``{"node1": "x"}``, ``null``) on a
    pre-parsed dict degrades to ``{}`` instead of raising ``ValueError`` /
    ``TypeError`` and crashing the whole entry load — the same fail-open
    contract the JSON-string branch already had.

    >>> parse_json_dict_int('{"node1": 3}')
    {'node1': 3}
    >>> parse_json_dict_int({"node1": "x"})
    {}
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        candidate: object = raw
    else:
        try:
            candidate = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(candidate, dict):
        return {}
    try:
        return {str(k): int(v) for k, v in candidate.items()}
    except (TypeError, ValueError):
        return {}


def parse_validity_fields(
    created: object, valid_from: object, invalid_from: object, *, reference_time: datetime
) -> tuple[datetime, datetime, datetime | None]:
    """Shared legacy-safe timestamp mapping for rows and early selection."""
    created_at = parse_dt_safe(created, default=reference_time) or reference_time
    opened = (parse_dt_safe(valid_from, default=created_at) or created_at) if valid_from else created_at
    closed = parse_dt_safe(invalid_from, default=None) if invalid_from else None
    return created_at, opened, closed
