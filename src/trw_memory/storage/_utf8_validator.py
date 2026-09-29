"""Write-time validation of a memory entry: UTF-8-safe text columns and a bounded id.

Prevents lone surrogates and other non-encodable Python str values from
reaching the database, where they would cause deterministic read failures
(sqlite3.OperationalError: Could not decode to UTF-8 column ...), and ids longer
than :data:`~trw_memory.models.memory.MAX_ENTRY_ID_CHARS`, which a resumable
sweep's cursor could not carry (B71-85).

Usage::

    from trw_memory.storage._utf8_validator import validate_utf8_fields
    validate_utf8_fields(row_dict)  # raises Utf8ValidationError on bad fields
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.exceptions import SchemaValidationError, Utf8ValidationError
from trw_memory.models.memory import MAX_ENTRY_ID_CHARS, MAX_TEXT_FIELD_CHARS

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


def validate_entry_for_write(entry: MemoryEntry) -> None:
    """Refuse an entry whose id is past :data:`MAX_ENTRY_ID_CHARS` or whose TEXT fields are not UTF-8-safe."""
    if len(entry.id) > MAX_ENTRY_ID_CHARS:
        raise SchemaValidationError(
            f"Write rejected: entry id is {len(entry.id)} characters, over {MAX_ENTRY_ID_CHARS}",
            failed_fields=["id"],
            reason="entry_id_too_long",
        )
    validate_utf8_fields({column: getattr(entry, attribute) for column, attribute in _ENTRY_TEXT_FIELDS})


def overlong_text_field(value: str) -> bool:
    """True when *value* is past :data:`MAX_TEXT_FIELD_CHARS`, the daemon's per-field TEXT bound.

    PRD-CORE-331 FR07 / B71-94: the local store and update paths refuse a ``content``/``detail`` this
    long instead of truncating it, matching the daemon's ``_arg_bounds.py::TEXT`` limit exactly.
    """
    return len(value) > MAX_TEXT_FIELD_CHARS


def refuse_overlong_text_fields(**fields: str | None) -> None:
    """Raise :class:`SchemaValidationError` naming every keyword field past :data:`MAX_TEXT_FIELD_CHARS`.

    A ``None`` value means the caller did not name that field (an update's unpatched fields); it is
    skipped rather than refused.
    """
    failed = [name for name, value in fields.items() if isinstance(value, str) and overlong_text_field(value)]
    if failed:
        raise SchemaValidationError(
            f"field(s) exceed {MAX_TEXT_FIELD_CHARS} characters: {failed!r}",
            failed_fields=failed,
            reason="text_field_too_long",
        )
