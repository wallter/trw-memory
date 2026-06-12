"""PRD-CORE-194 FR01 — bi-temporal validity fields on MemoryEntry.

Schema additions: ``valid_from`` (event time, defaults to ``created_at``),
``invalid_from`` (nullable window-close instant), ``invalidated_by`` (id of the
superseding record). Closed window must name its closer; a window may not close
before it opens; absent fields = open validity (back-compat).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import MemoryEntry

from tests.conftest import make_entry_dict


def _fixed_created() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_absent_fields_default_to_open_validity() -> None:
    """A pre-migration row (no validity keys) loads open: valid_from==created_at."""
    created = _fixed_created()
    # make_entry_dict produces the legacy serialised shape WITHOUT validity keys.
    row = make_entry_dict(entry_id="M-bc", content="x", created_at=created)
    assert "valid_from" not in row
    assert "invalid_from" not in row
    assert "invalidated_by" not in row

    entry = MemoryEntry(**row)

    assert entry.valid_from == created
    assert entry.invalid_from is None
    assert entry.invalidated_by is None


def test_valid_from_defaults_to_created_at_on_construction() -> None:
    """Constructing with only content+created_at sets valid_from = created_at."""
    created = _fixed_created()
    entry = MemoryEntry(id="M-1", content="c", created_at=created)
    assert entry.valid_from == created
    assert entry.invalid_from is None
    assert entry.invalidated_by is None


def test_explicit_valid_from_distinct_from_created_at() -> None:
    """A fact learned today about a past event records the past event time."""
    created = _fixed_created()
    past_event = created - timedelta(days=30)
    entry = MemoryEntry(id="M-2", content="c", created_at=created, valid_from=past_event)
    assert entry.valid_from == past_event
    assert entry.created_at == created


def test_invalid_from_requires_invalidated_by() -> None:
    """A closed window must name its closer (invalid_from without invalidated_by)."""
    created = _fixed_created()
    with pytest.raises(ValidationError):
        MemoryEntry(
            id="M-3",
            content="c",
            created_at=created,
            invalid_from=created + timedelta(hours=1),
        )


def test_invalidated_by_requires_invalid_from() -> None:
    """invalidated_by without invalid_from is rejected (closed iff named closer)."""
    created = _fixed_created()
    with pytest.raises(ValidationError):
        MemoryEntry(
            id="M-4",
            content="c",
            created_at=created,
            invalidated_by="M-other",
        )


def test_invalid_from_before_valid_from_rejected() -> None:
    """A window cannot close before it opens (back-dated supersession)."""
    created = _fixed_created()
    with pytest.raises(ValidationError):
        MemoryEntry(
            id="M-5",
            content="c",
            created_at=created,
            valid_from=created,
            invalid_from=created - timedelta(hours=1),
            invalidated_by="M-newer",
        )


def test_same_instant_supersession_is_gap_free() -> None:
    """invalid_from == valid_from is the intended gap-free boundary case."""
    created = _fixed_created()
    entry = MemoryEntry(
        id="M-6",
        content="c",
        created_at=created,
        valid_from=created,
        invalid_from=created,
        invalidated_by="M-newer",
    )
    assert entry.invalid_from == entry.valid_from
    assert entry.invalidated_by == "M-newer"


def test_closed_window_round_trips_through_to_dict() -> None:
    """A closed window serializes the three validity fields and reloads identical."""
    created = _fixed_created()
    close = created + timedelta(days=2)
    entry = MemoryEntry(
        id="M-7",
        content="c",
        created_at=created,
        valid_from=created,
        invalid_from=close,
        invalidated_by="M-b",
    )
    d = entry.to_dict()
    assert d["valid_from"] == created.isoformat()
    assert d["invalid_from"] == close.isoformat()
    assert d["invalidated_by"] == "M-b"

    reloaded = MemoryEntry(**d)
    assert reloaded.valid_from == created
    assert reloaded.invalid_from == close
    assert reloaded.invalidated_by == "M-b"


def test_open_entry_serializes_none_invalid_fields() -> None:
    """OQ5: open entry round-trips invalid_from=None / invalidated_by=None cleanly."""
    created = _fixed_created()
    entry = MemoryEntry(id="M-8", content="c", created_at=created)
    d = entry.to_dict()
    assert d["valid_from"] == created.isoformat()
    assert d["invalid_from"] is None
    assert d["invalidated_by"] is None

    reloaded = MemoryEntry(**d)
    assert reloaded.invalid_from is None
    assert reloaded.invalidated_by is None
