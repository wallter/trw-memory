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

    >>> parse_json_dict_int('{"node1": 3}')
    {'node1': 3}
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {}
