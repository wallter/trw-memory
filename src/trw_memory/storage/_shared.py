"""Shared constants and helpers for storage backends.

Centralises field definitions, update-field validation, and value
serialisation logic used by both :mod:`sqlite_backend` and
:mod:`yaml_backend`.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from trw_memory.models.memory import Anchor, Assertion, Confidence, MemoryStatus, MemoryType, ProtectionTier

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
    "session_count",
    "q_value",
    "q_observations",
    "source",
    "source_identity",
    "client_profile",
    "model_id",
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
    "anchors",
    "anchor_validity",
    "type",
    "nudge_line",
    "expires_at",
    "confidence",
    "task_type",
    "domain",
    "phase_origin",
    "phase_affinity",
    "team_origin",
    "protection_tier",
    "sessions_surfaced",
    "avg_rework_delta",
    "outcome_correlation",
    "sync_hash",
    "sync_seq",
    "last_synced_at",
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
        "anchors",
        "domain",
        "phase_affinity",
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


def serialize_update_value(key: str, val: object) -> list[object] | dict[str, str] | str:
    """Normalise a single update value for storage.

    Handles:
    - ``assertions`` list -> each item serialised via ``model_dump()``
    - ``anchors`` list -> each item serialised via ``model_dump()``
    - ``list`` fields in :data:`LIST_FIELDS` -> kept as list of strings
    - ``dict`` fields in :data:`DICT_FIELDS` -> kept as dict
    - ``datetime`` -> ISO-8601 string
    - ``MemoryStatus`` -> its ``.value`` string
    - ``MemoryType``, ``Confidence``, ``ProtectionTier`` -> their ``.value`` string

    Returns the normalised value. The caller is responsible for
    format-specific encoding (e.g. ``json.dumps`` for SQLite, plain
    list/dict for YAML).
    """
    # Assertions need model_dump(), not str() — Pydantic models have complex structure
    if key == "assertions" and isinstance(val, list):
        return [a.model_dump() if isinstance(a, Assertion) else a for a in val]
    # Anchors are list[Anchor] and need JSON serialization with model_dump()
    if key == "anchors" and isinstance(val, list):
        return [a.model_dump() if isinstance(a, Anchor) else a for a in val]
    if key in LIST_FIELDS and isinstance(val, list):
        return [str(v) for v in val]
    if key in DICT_FIELDS and isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, MemoryStatus):
        return val.value
    if isinstance(val, (MemoryType, Confidence, ProtectionTier)):
        return cast("str", val.value)
    return cast("list[object] | dict[str, str] | str", val)
