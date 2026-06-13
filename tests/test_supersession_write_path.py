"""PRD-CORE-194 FR04 — supersession write path (writer sets invalidated_by).

Consolidation ``_archive_originals`` closes the prior records' validity windows
(invalid_from + invalidated_by = consolidated id) IN ADDITION to the existing
``consolidated_into`` + ``status=archived`` flags, retaining the rows (never a
delete). The consolidated entry's ``valid_from`` is the same consolidation
instant (gap-free per FR01). The sync hash changes on a close-window write.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trw_memory.lifecycle.consolidation import _archive_originals, _create_consolidated_entry
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.delta import DeltaTracker


def _store_cluster(backend: SQLiteBackend) -> list[MemoryEntry]:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cluster = [
        MemoryEntry(id="M-a", content="alpha fact", created_at=created),
        MemoryEntry(id="M-b", content="beta fact", created_at=created),
    ]
    for e in cluster:
        backend.store(e)
    return cluster


def test_consolidation_closes_window_not_delete(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "m.db")
    cluster = _store_cluster(backend)
    row_count_before = len(backend.list_entries(limit=100))

    consolidated = _create_consolidated_entry(cluster, "merged content", "merged detail", backend, namespace="default")
    _archive_originals(cluster, consolidated.id, backend, invalid_from=consolidated.valid_from)

    a = backend.get("M-a")
    b = backend.get("M-b")
    assert a is not None and b is not None

    # Window closed: invalid_from set + invalidated_by == consolidated id.
    assert a.invalid_from is not None
    assert a.invalidated_by == consolidated.id
    assert b.invalid_from is not None
    assert b.invalidated_by == consolidated.id
    # Existing flags preserved (complementary, not replaced).
    assert a.consolidated_into == consolidated.id
    assert a.status == MemoryStatus.ARCHIVED
    assert a.validity_state() == "superseded"

    # Retained, not deleted: row count is non-decreasing across the operation
    # (2 originals + 1 consolidated = 3).
    row_count_after = len(backend.list_entries(limit=100))
    assert row_count_after >= row_count_before
    assert row_count_after == row_count_before + 1


def test_consolidated_valid_from_equals_close_instant_gap_free(tmp_path: Path) -> None:
    """OQ3: the consolidated entry's valid_from == the originals' invalid_from."""
    backend = SQLiteBackend(tmp_path / "m.db")
    cluster = _store_cluster(backend)

    consolidated = _create_consolidated_entry(cluster, "merged content", "merged detail", backend, namespace="default")
    _archive_originals(cluster, consolidated.id, backend, invalid_from=consolidated.valid_from)

    a = backend.get("M-a")
    new = backend.get(consolidated.id)
    assert a is not None and new is not None
    # Gap-free: the superseding record opens exactly when the prior window closed.
    assert new.valid_from == a.invalid_from
    assert new.invalid_from is None  # consolidated entry is open


def test_supersession_no_delete(tmp_path: Path) -> None:
    """Row count never decreases across a supersession write."""
    backend = SQLiteBackend(tmp_path / "m.db")
    cluster = _store_cluster(backend)
    before = len(backend.list_entries(limit=100))
    consolidated = _create_consolidated_entry(cluster, "merged", "merged", backend, namespace="default")
    _archive_originals(cluster, consolidated.id, backend, invalid_from=consolidated.valid_from)
    after = len(backend.list_entries(limit=100))
    assert after >= before


def test_sync_hash_validity(tmp_path: Path) -> None:
    """FR04: closing a window changes the entry's sync hash (marks dirty)."""
    backend = SQLiteBackend(tmp_path / "m.db")
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    entry = MemoryEntry(id="M-x", content="c", created_at=created)
    backend.store(entry)
    before = backend.get("M-x")
    assert before is not None
    hash_before = DeltaTracker.compute_sync_hash(before)

    close = datetime(2026, 1, 2, tzinfo=timezone.utc)
    backend.update("M-x", invalid_from=close, invalidated_by="M-y")
    after = backend.get("M-x")
    assert after is not None
    hash_after = DeltaTracker.compute_sync_hash(after)

    assert hash_before != hash_after
    # The backend.update() also persists a recomputed sync_hash on the row.
    assert after.sync_hash == hash_after
    assert after.sync_hash != before.sync_hash
