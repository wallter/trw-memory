"""Write-time UTF-8 validation for SQLite text columns.

Prevents lone surrogates and other non-encodable Python str values from
reaching the database, where they would cause deterministic read failures
(sqlite3.OperationalError: Could not decode to UTF-8 column ...).

Usage::

    from trw_memory.storage._utf8_validator import validate_utf8_fields
    validate_utf8_fields(row_dict)  # raises Utf8ValidationError on bad fields
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.exceptions import Utf8ValidationError

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry

# Bare TEXT columns and their MemoryEntry attributes. JSON-serialised fields
# are already safe because json.dumps() escapes surrogates.
_ENTRY_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "id"),
    ("content", "content"),
    ("detail", "detail"),
    ("nudge_line", "nudge_line"),
    ("type", "type"),
    ("namespace", "namespace"),
    ("source", "source"),
    ("source_identity", "source_identity"),
    ("client_profile", "client_profile"),
    ("model_id", "model_id"),
    ("consolidated_into", "consolidated_into"),
    ("remote_id", "remote_id"),
    ("expires_at", "expires"),
    ("task_type", "task_type"),
    ("phase_origin", "phase_origin"),
    ("team_origin", "team_origin"),
    ("outcome_correlation", "outcome_correlation"),
    ("sync_hash", "sync_hash"),
    ("invalidated_by", "invalidated_by"),
)
_TEXT_FIELD_ORDER: tuple[str, ...] = tuple(column for column, _attribute in _ENTRY_TEXT_FIELDS)
_TEXT_FIELDS: frozenset[str] = frozenset(_TEXT_FIELD_ORDER)


def _is_valid_utf8(value: str) -> bool:
    """Return True iff *value* encodes cleanly as strict UTF-8.

    Lone surrogates (\\uD800–\\uDFFF) and any char that Python's codec
    rejects with errors='strict' return False.
    """
    try:
        value.encode("utf-8", errors="strict")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def validate_utf8_fields(row_dict: dict[str, object]) -> None:
    """Validate all TEXT-column fields in *row_dict* for UTF-8 safety.

    Args:
        row_dict: Mapping of column name → value (as returned by
            :func:`trw_memory.storage._row_mapper.entry_to_row` converted to
            a dict, or any partial update dict).

    Raises:
        Utf8ValidationError: If one or more string fields contain bytes that
            cannot be encoded as strict UTF-8.  ``failed_fields`` on the
            exception lists every offending field name.
    """
    failed: list[str] = []
    for field in _TEXT_FIELD_ORDER:
        raw = row_dict.get(field)
        if not isinstance(raw, str):
            continue
        if not _is_valid_utf8(raw):
            failed.append(field)
    if failed:
        raise Utf8ValidationError(
            f"Write rejected: {len(failed)} field(s) contain invalid UTF-8: {failed!r}",
            failed_fields=failed,
        )


def validate_entry_utf8(entry: MemoryEntry) -> None:
    """Validate the bare TEXT fields persisted from a memory entry."""
    validate_utf8_fields({column: getattr(entry, attribute) for column, attribute in _ENTRY_TEXT_FIELDS})
