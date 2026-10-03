"""Recall bookkeeping is local telemetry: it never marks a row dirty for push.

A recall bumped ``sync_seq`` and cleared ``last_synced_at`` on every row it returned. The push payload never carried the
counters, so the next push re-uploaded unchanged content for every recalled row; for a row PULLED from a teammate it
went up under the local id and the platform stored a new learning (``team-sync-team-sync-...``), and the next pull
merge mistook the clean teammate row for an unpushed local edit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.delta import DeltaTracker

NS = "project:bk-11111111"


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    store.store(MemoryEntry(id="A-1", namespace=NS, content="a synced row"))
    DeltaTracker.mark_synced(["A-1"], store, namespace=NS)
    yield store
    store.close()


def _state(store: SQLiteBackend) -> tuple[int, object]:
    row = store.get("A-1", namespace=NS)
    assert row is not None
    return row.sync_seq, row.last_synced_at


def test_a_recall_does_not_make_a_synced_row_dirty(backend: SQLiteBackend) -> None:
    before = _state(backend)
    assert before[1] is not None

    backend.increment_recall_access(["A-1"], namespace=NS)

    assert _state(backend) == before
    assert DeltaTracker.get_dirty_entries(backend, namespace=NS) == []
    row = backend.get("A-1", namespace=NS)
    assert row is not None and row.recall_count == 1  # the counter itself still counts


def test_a_session_count_does_not_make_a_synced_row_dirty(backend: SQLiteBackend) -> None:
    before = _state(backend)

    backend.increment_session_counts(["A-1"], namespace=NS)

    assert _state(backend) == before
    assert DeltaTracker.get_dirty_entries(backend, namespace=NS) == []
    row = backend.get("A-1", namespace=NS)
    assert row is not None and row.session_count == 1


def test_a_content_edit_still_makes_it_dirty(backend: SQLiteBackend) -> None:
    backend.update("A-1", namespace=NS, content="edited")

    assert [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=NS)] == ["A-1"]
