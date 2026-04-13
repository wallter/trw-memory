"""Tests for sync delta tracking — PRD-INFRA-051.

Covers:
- FR01: MemoryEntry sync fields defaults
- FR05: DeltaTracker.compute_sync_hash determinism and sensitivity
- FR05: DeltaTracker.get_dirty_entries
- FR05: DeltaTracker.mark_synced
- FR06: Auto dirty-marking on store/update
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.delta import DeltaTracker

# ---------------------------------------------------------------------------
# FR01: MemoryEntry sync field defaults
# ---------------------------------------------------------------------------


def test_memory_entry_sync_fields_defaults() -> None:
    """FR01: New MemoryEntry has sync_hash='', sync_seq=0, last_synced_at=None."""
    entry = MemoryEntry(id="M-default", content="test sync defaults")
    assert entry.sync_hash == ""
    assert entry.sync_seq == 0
    assert entry.last_synced_at is None


def test_memory_entry_sync_fields_in_to_dict() -> None:
    """FR01: to_dict() includes sync fields."""
    entry = MemoryEntry(id="M-dict", content="dict test", sync_hash="abc123", sync_seq=5)
    d = entry.to_dict()
    assert d["sync_hash"] == "abc123"
    assert d["sync_seq"] == 5
    assert d["last_synced_at"] is None


def test_memory_entry_sync_fields_to_dict_with_last_synced() -> None:
    """FR01: to_dict() serializes last_synced_at as ISO string."""
    now = datetime.now(tz=timezone.utc)
    entry = MemoryEntry(id="M-ts", content="ts test", last_synced_at=now)
    d = entry.to_dict()
    assert d["last_synced_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# FR05: DeltaTracker.compute_sync_hash
# ---------------------------------------------------------------------------


def test_compute_sync_hash_deterministic() -> None:
    """FR05: Same entry produces same hash every time."""
    entry = MemoryEntry(id="M-hash", content="deterministic test", detail="some detail")
    h1 = DeltaTracker.compute_sync_hash(entry)
    h2 = DeltaTracker.compute_sync_hash(entry)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length


def test_compute_sync_hash_changes_on_content_change() -> None:
    """FR05: Modifying a content field changes the hash."""
    entry1 = MemoryEntry(id="M-h1", content="original")
    entry2 = MemoryEntry(id="M-h2", content="modified")
    h1 = DeltaTracker.compute_sync_hash(entry1)
    h2 = DeltaTracker.compute_sync_hash(entry2)
    assert h1 != h2


def test_compute_sync_hash_ignores_non_content_fields() -> None:
    """FR05: Non-content fields (id, sync_seq, etc.) do not affect hash."""
    entry1 = MemoryEntry(id="M-a", content="same content", sync_seq=0)
    entry2 = MemoryEntry(id="M-b", content="same content", sync_seq=99)
    h1 = DeltaTracker.compute_sync_hash(entry1)
    h2 = DeltaTracker.compute_sync_hash(entry2)
    assert h1 == h2


def test_compute_sync_hash_normalizes_float_precision() -> None:
    """FR05: Float fields are canonicalized to six decimal places."""
    entry1 = MemoryEntry(id="M-f1", content="same", importance=0.1234564)
    entry2 = MemoryEntry(id="M-f2", content="same", importance=0.12345649)
    assert DeltaTracker.compute_sync_hash(entry1) == DeltaTracker.compute_sync_hash(entry2)


# ---------------------------------------------------------------------------
# FR06: Auto dirty-marking on store
# ---------------------------------------------------------------------------


def test_store_increments_sync_seq(tmp_path: Path) -> None:
    """FR06: Storing an entry sets sync_seq >= 1 and computes sync_hash."""
    backend = SQLiteBackend(tmp_path / "test.db")
    entry = MemoryEntry(id="M-store", content="store test")
    assert entry.sync_seq == 0  # default before store
    backend.store(entry)
    result = backend.get("M-store")
    assert result is not None
    assert result.sync_seq >= 1
    assert result.sync_hash != ""
    assert result.last_synced_at is None  # not yet synced
    backend.close()


def test_store_yaml_increments_sync_seq(tmp_path: Path) -> None:
    """FR06: YAML backend also marks entries dirty on store."""
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml_entries")
    entry = MemoryEntry(id="M-yaml-store", content="yaml store test")
    backend.store(entry)
    result = backend.get("M-yaml-store")
    assert result is not None
    assert result.sync_seq >= 1
    assert result.sync_hash != ""
    assert result.last_synced_at is None
    backend.close()


# ---------------------------------------------------------------------------
# FR05: DeltaTracker.get_dirty_entries
# ---------------------------------------------------------------------------


def test_get_dirty_entries_returns_unsynced(tmp_path: Path) -> None:
    """FR05: get_dirty_entries returns entries with last_synced_at=None."""
    backend = SQLiteBackend(tmp_path / "test.db")
    # Store two entries (auto dirty-marked by FR06)
    e1 = MemoryEntry(id="M-dirty-1", content="dirty one")
    e2 = MemoryEntry(id="M-dirty-2", content="dirty two")
    backend.store(e1)
    backend.store(e2)

    dirty = DeltaTracker.get_dirty_entries(backend, since_seq=0)
    assert len(dirty) == 2
    ids = {e.id for e in dirty}
    assert "M-dirty-1" in ids
    assert "M-dirty-2" in ids
    backend.close()


def test_get_dirty_entries_respects_since_seq(tmp_path: Path) -> None:
    """FR05: get_dirty_entries only returns entries with sync_seq > since_seq."""
    backend = SQLiteBackend(tmp_path / "test.db")
    e1 = MemoryEntry(id="M-seq1", content="seq one")
    backend.store(e1)

    e2 = MemoryEntry(id="M-seq2", content="seq two")
    backend.store(e2)

    # Both entries have sync_seq=1 after store
    # since_seq=0 should return both
    dirty_all = DeltaTracker.get_dirty_entries(backend, since_seq=0)
    assert len(dirty_all) == 2

    # since_seq=1 should return none (both have sync_seq=1, query is >)
    dirty_none = DeltaTracker.get_dirty_entries(backend, since_seq=1)
    assert len(dirty_none) == 0

    # Update e2 to increment its sync_seq to 2
    backend.update("M-seq2", content="seq two updated")
    # Now since_seq=1 should return only M-seq2
    dirty_one = DeltaTracker.get_dirty_entries(backend, since_seq=1)
    ids = {e.id for e in dirty_one}
    assert "M-seq2" in ids
    assert "M-seq1" not in ids
    backend.close()


def test_sqlite_update_recomputes_sync_hash(tmp_path: Path) -> None:
    """FR06: SQLite updates recompute sync_hash and clear last_synced_at."""
    backend = SQLiteBackend(tmp_path / "test.db")
    entry = MemoryEntry(id="M-update", content="before")
    backend.store(entry)
    DeltaTracker.mark_synced(["M-update"], backend)
    before = backend.get("M-update")
    assert before is not None

    backend.update("M-update", content="after")

    after = backend.get("M-update")
    assert after is not None
    assert after.sync_seq == before.sync_seq + 1
    assert after.sync_hash != before.sync_hash
    assert after.last_synced_at is None
    backend.close()


def test_yaml_update_recomputes_sync_hash(tmp_path: Path) -> None:
    """FR06: YAML updates also recompute sync_hash and clear last_synced_at."""
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml_entries")
    entry = MemoryEntry(id="M-yaml-update", content="before")
    backend.store(entry)
    DeltaTracker.mark_synced(["M-yaml-update"], backend)
    before = backend.get("M-yaml-update")
    assert before is not None

    backend.update("M-yaml-update", content="after")

    after = backend.get("M-yaml-update")
    assert after is not None
    assert after.sync_seq == before.sync_seq + 1
    assert after.sync_hash != before.sync_hash
    assert after.last_synced_at is None
    backend.close()


# ---------------------------------------------------------------------------
# FR05: DeltaTracker.mark_synced
# ---------------------------------------------------------------------------


def test_mark_synced_sets_timestamp(tmp_path: Path) -> None:
    """FR05: mark_synced sets last_synced_at on entries."""
    backend = SQLiteBackend(tmp_path / "test.db")
    e1 = MemoryEntry(id="M-sync1", content="sync one")
    e2 = MemoryEntry(id="M-sync2", content="sync two")
    backend.store(e1)
    backend.store(e2)

    count = DeltaTracker.mark_synced(["M-sync1", "M-sync2"], backend)
    assert count == 2

    r1 = backend.get("M-sync1")
    r2 = backend.get("M-sync2")
    assert r1 is not None and r1.last_synced_at is not None
    assert r2 is not None and r2.last_synced_at is not None
    backend.close()


def test_mark_synced_nonexistent_entry(tmp_path: Path) -> None:
    """FR05: mark_synced silently skips non-existent entries."""
    backend = SQLiteBackend(tmp_path / "test.db")
    count = DeltaTracker.mark_synced(["M-nonexist"], backend)
    # Depending on backend behavior, this may be 0 (entry not found)
    # or raise no error
    assert count == 0
    backend.close()


def test_mark_synced_removes_from_dirty(tmp_path: Path) -> None:
    """FR05: After mark_synced, entries no longer appear in get_dirty_entries."""
    backend = SQLiteBackend(tmp_path / "test.db")
    e1 = MemoryEntry(id="M-clean", content="will be synced")
    backend.store(e1)

    # Should be dirty initially
    dirty_before = DeltaTracker.get_dirty_entries(backend, since_seq=0)
    assert any(e.id == "M-clean" for e in dirty_before)

    # Mark synced
    DeltaTracker.mark_synced(["M-clean"], backend)

    # Should no longer be dirty
    dirty_after = DeltaTracker.get_dirty_entries(backend, since_seq=0)
    assert not any(e.id == "M-clean" for e in dirty_after)
    backend.close()


# ---------------------------------------------------------------------------
# FR02/FR03: Schema migration and round-trip
# ---------------------------------------------------------------------------


def test_sqlite_roundtrip_sync_fields(tmp_path: Path) -> None:
    """FR02/FR03: Sync fields survive SQLite store/get round-trip."""
    backend = SQLiteBackend(tmp_path / "test.db")
    now = datetime.now(tz=timezone.utc)
    entry = MemoryEntry(
        id="M-rt",
        content="roundtrip test",
        sync_hash="deadbeef",
        sync_seq=42,
        last_synced_at=now,
    )
    # Store with pre-set values (FR06 will overwrite sync_seq/hash)
    backend.store(entry)
    result = backend.get("M-rt")
    assert result is not None
    # FR06 auto-marks dirty, so sync_seq will be incremented and hash recomputed
    assert result.sync_seq >= 1
    assert result.sync_hash != ""
    # last_synced_at is cleared by FR06 dirty marking
    assert result.last_synced_at is None
    backend.close()


# ---------------------------------------------------------------------------
# FR04: YAML backend round-trip
# ---------------------------------------------------------------------------


def test_yaml_roundtrip_sync_fields(tmp_path: Path) -> None:
    """FR04: Sync fields survive YAML store/get round-trip."""
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml_entries")
    entry = MemoryEntry(
        id="M-yaml-rt",
        content="yaml roundtrip",
    )
    backend.store(entry)
    result = backend.get("M-yaml-rt")
    assert result is not None
    # FR06: auto dirty-marked
    assert result.sync_seq >= 1
    assert result.sync_hash != ""
    assert result.last_synced_at is None
    backend.close()
