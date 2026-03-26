"""Shared constants and helpers for storage backends.

Centralises field definitions, update-field validation, and value
serialisation logic used by both :mod:`sqlite_backend` and
:mod:`yaml_backend`.
"""

from __future__ import annotations

from datetime import datetime

from trw_memory.models.memory import MemoryStatus

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

#: All column/field names on MemoryEntry, in canonical order.
#: Used by SQLiteBackend for SELECT/INSERT column lists.
ENTRY_COLUMNS: tuple[str, ...] = (
    "id",
    "content",
    "detail",
    "tags",
    "evidence",
    "importance",
    "status",
    "recurrence",
    "namespace",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "access_count",
    "q_value",
    "q_observations",
    "source",
    "source_identity",
    "merged_from",
    "consolidated_from",
    "consolidated_into",
    "metadata",
    "vector_clock",
    "remote_id",
    "published_to_platform",
    "pending_delete",
    "cross_validated",
    "outcome_history",
    "assertions",
)

#: Fields that must never be changed via ``update()``.
IMMUTABLE_FIELDS: frozenset[str] = frozenset({"id", "created_at"})

#: Fields whose values are JSON-encoded lists.
LIST_FIELDS: frozenset[str] = frozenset(
    {
        "tags",
        "evidence",
        "merged_from",
        "consolidated_from",
        "outcome_history",
        "assertions",
    }
)

#: Fields whose values are JSON-encoded dicts.
DICT_FIELDS: frozenset[str] = frozenset({"metadata", "vector_clock"})


# ---------------------------------------------------------------------------
# Update helpers
# ---------------------------------------------------------------------------


def validate_update_fields(
    fields: dict[str, object],
    valid_columns: frozenset[str],
) -> None:
    """Raise ``ValueError`` if any key is not in *valid_columns*.

    Args:
        fields: Mapping of field names to values.
        valid_columns: Allowed field names for update.

    Raises:
        ValueError: If an invalid field name is found.
    """
    for key in fields:
        if key not in valid_columns:
            raise ValueError(key)


def serialize_update_value(key: str, val: object) -> list[str] | dict[str, str] | str | object:
    """Normalise a single update value for storage.

    Handles:
    - ``list`` fields in :data:`LIST_FIELDS` -> kept as list (caller wraps for format)
    - ``dict`` fields in :data:`DICT_FIELDS` -> kept as dict
    - ``datetime`` -> ISO-8601 string
    - ``MemoryStatus`` -> its ``.value`` string

    Returns the normalised value. The caller is responsible for
    format-specific encoding (e.g. ``json.dumps`` for SQLite, plain
    list/dict for YAML).
    """
    if key in LIST_FIELDS and isinstance(val, list):
        return [str(v) for v in val]
    if key in DICT_FIELDS and isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, MemoryStatus):
        return val.value
    return val
