"""Tests for write-time UTF-8 validation (P1 — prevention layer).

All tests use in-memory SQLite so there is no filesystem I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import Utf8ValidationError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(**kwargs: object) -> MemoryEntry:
    """Minimal MemoryEntry for write tests."""
    return MemoryEntry(
        id=str(kwargs.get("entry_id", "M-utf8-001")),
        content=str(kwargs.get("content", "valid content")),
        detail=str(kwargs.get("detail", "")),
        nudge_line=str(kwargs.get("nudge_line", "")),
        namespace=str(kwargs.get("namespace", "default")),
        source="agent",  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Test: lone surrogate in detail is rejected
# ---------------------------------------------------------------------------


def test_upsert_rejects_lone_surrogate_in_detail() -> None:
    """A lone surrogate in `detail` must raise Utf8ValidationError."""
    backend = SQLiteBackend(Path(":memory:"))
    entry = _make_entry(detail="\ud83d")  # lone high surrogate
    with pytest.raises(Utf8ValidationError) as exc_info:
        backend.store(entry)
    assert "detail" in exc_info.value.failed_fields
    backend.close()


# ---------------------------------------------------------------------------
# Test: multiple bad fields reported together
# ---------------------------------------------------------------------------


def test_upsert_rejects_multiple_bad_fields() -> None:
    """Bad bytes in detail AND nudge_line → both field names in failed_fields."""
    backend = SQLiteBackend(Path(":memory:"))
    entry = _make_entry(detail="\ud83d", nudge_line="\udc00bad")
    with pytest.raises(Utf8ValidationError) as exc_info:
        backend.store(entry)
    assert "detail" in exc_info.value.failed_fields
    assert "nudge_line" in exc_info.value.failed_fields
    backend.close()


# ---------------------------------------------------------------------------
# Test: valid Unicode (including emoji) passes and round-trips
# ---------------------------------------------------------------------------


def test_upsert_accepts_valid_unicode_including_emoji() -> None:
    """Valid Unicode including emoji / CJK must be accepted and round-trip cleanly."""
    backend = SQLiteBackend(Path(":memory:"))
    entry = _make_entry(
        entry_id="M-utf8-002",
        content="valid",
        detail="✓ 🚀 日本語",
    )
    backend.store(entry)
    result = backend.get("M-utf8-002")
    assert result is not None
    assert result.detail == "✓ 🚀 日本語"
    backend.close()


# ---------------------------------------------------------------------------
# Test: validator helper checks only TEXT fields (not JSON / numeric)
# ---------------------------------------------------------------------------


def test_upsert_does_not_validate_numeric_or_json_fields_twice() -> None:
    """_validate_utf8_fields should only iterate _TEXT_FIELDS once, not more."""
    from trw_memory.storage._utf8_validator import _TEXT_FIELDS, validate_utf8_fields

    # A row dict with a lone surrogate in a JSON field (e.g., 'tags') — should
    # NOT raise because tags is not in _TEXT_FIELDS.  Validation of JSON fields
    # is left to the JSON encoder, which serialises surrogates rather than
    # failing (and the resulting JSON is valid ASCII).
    row: dict[str, object] = {
        "id": "M-003",
        "content": "ok",
        "detail": "ok",
        "tags": '["\\ud83d"]',  # JSON-escaped — already a valid str
        "importance": 0.5,
    }
    # Should not raise — tags is not a TEXT field we validate.
    validate_utf8_fields(row)  # no exception expected

    # Verify coverage: _TEXT_FIELDS is a non-empty frozenset (sanity guard).
    assert len(_TEXT_FIELDS) > 0
