"""``list_entries(after=...)`` keyset paging — disjoint AND complete.

An OFFSET window is wrong for any caller that mutates the rows it pages over
(``namespaces.curate`` deletes what it moves and leaves what it skips), so the
listing grew a keyset cursor. The property that makes the cursor worth having
is exactly what these tests assert: consecutive pages never repeat a row and
never omit one — including when many rows share an ``updated_at``, which is the
case an ``updated_at``-only ORDER BY cannot order deterministically.

Both shipped backends are exercised: a merge across two YAML namespace
directories runs the same loop as a merge inside one SQLite file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import make_entry
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import EntryCursor, StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend

NAMESPACE = "project:keyset-00000000"

#: 12 rows over 4 distinct timestamps — every timestamp is a 3-way tie, so the
#: ``id`` tiebreak is load-bearing for all 12, not for a lucky pair.
_ROW_COUNT = 12
_TIE_WIDTH = 3


def _seed_with_ties(backend: StorageBackend) -> list[str]:
    """Store ``_ROW_COUNT`` rows, ``_TIE_WIDTH`` of them per ``updated_at``."""
    base = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    ids: list[str] = []
    for index in range(_ROW_COUNT):
        entry_id = f"M-{index:02d}"
        stamp = base - timedelta(minutes=index // _TIE_WIDTH)
        backend.store(make_entry(entry_id=entry_id, namespace=NAMESPACE).model_copy(update={"updated_at": stamp}))
        ids.append(entry_id)
    return ids


def _page_through(backend: StorageBackend, *, limit: int) -> list[list[MemoryEntry]]:
    """Drain the namespace with ``after=`` and return the pages in order."""
    pages: list[list[MemoryEntry]] = []
    cursor: EntryCursor | None = None
    # Bound the loop independently of the cursor: a non-advancing cursor is a
    # hang, and a test that hangs reports nothing.
    for _ in range(_ROW_COUNT + 2):
        page = backend.list_entries(namespace=NAMESPACE, limit=limit, after=cursor)
        if not page:
            return pages
        pages.append(page)
        cursor = EntryCursor.from_entry(page[-1])
    raise AssertionError("keyset paging did not terminate")


@pytest.fixture
def sqlite_store(tmp_path: Path) -> StorageBackend:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


@pytest.fixture
def yaml_store(tmp_path: Path) -> StorageBackend:
    store = YAMLBackend(tmp_path / "entries")
    yield store
    store.close()


@pytest.fixture(params=["sqlite", "yaml"])
def store(request: pytest.FixtureRequest) -> StorageBackend:
    """Both shipped backends — the curate verbs run against either."""
    backend: StorageBackend = request.getfixturevalue(f"{request.param}_store")
    return backend


def test_keyset_pages_are_disjoint_and_complete_across_ties(store: StorageBackend) -> None:
    """Every row appears exactly once across the pages, ties included."""
    seeded = _seed_with_ties(store)

    pages = _page_through(store, limit=5)
    paged_ids = [entry.id for page in pages for entry in page]

    assert len(paged_ids) == len(set(paged_ids)), "a row was served by two pages"
    assert sorted(paged_ids) == sorted(seeded), "keyset paging dropped or invented rows"
    assert all(len(page) <= 5 for page in pages)
    assert len(pages) == 3, "12 rows at 5 per page is 3 pages, the last one short"


def test_keyset_pages_reproduce_the_unpaged_order(store: StorageBackend) -> None:
    """Paging is a partition of the single-shot listing, in the same order."""
    _seed_with_ties(store)

    unpaged = [entry.id for entry in store.list_entries(namespace=NAMESPACE, limit=_ROW_COUNT * 2)]
    paged = [entry.id for page in _page_through(store, limit=5) for entry in page]

    assert paged == unpaged


def test_a_page_size_of_one_still_terminates_and_covers_every_row(store: StorageBackend) -> None:
    """The tightest window: a tie group of 3 spans 3 single-row pages."""
    seeded = _seed_with_ties(store)

    pages = _page_through(store, limit=1)

    assert [entry.id for page in pages for entry in page] == [
        entry.id for entry in store.list_entries(namespace=NAMESPACE, limit=_ROW_COUNT * 2)
    ]
    assert len(pages) == len(seeded)


def test_a_cursor_past_the_last_row_returns_nothing(store: StorageBackend) -> None:
    """The drain condition: an exhausted cursor is an empty page, not a repeat."""
    _seed_with_ties(store)
    listing = store.list_entries(namespace=NAMESPACE, limit=_ROW_COUNT * 2)

    assert store.list_entries(namespace=NAMESPACE, limit=5, after=EntryCursor.from_entry(listing[-1])) == []


def test_a_cursor_filters_only_below_itself_not_the_whole_tie_group(store: StorageBackend) -> None:
    """A tie must be split BY ID, not skipped wholesale or served twice."""
    _seed_with_ties(store)
    listing = store.list_entries(namespace=NAMESPACE, limit=_ROW_COUNT * 2)
    # The first tie group is listing[0:3]; resume from its middle row.
    resumed = store.list_entries(namespace=NAMESPACE, limit=_ROW_COUNT, after=EntryCursor.from_entry(listing[1]))

    assert [entry.id for entry in resumed] == [entry.id for entry in listing[2:]]
