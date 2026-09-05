"""Tests for sync delta tracking — PRD-INFRA-051.

Covers:
- FR01: MemoryEntry sync fields defaults
- FR05: DeltaTracker.compute_sync_hash determinism and sensitivity
- FR05: DeltaTracker.get_dirty_entries
- FR05: DeltaTracker.mark_synced
- FR06: Auto dirty-marking on store/update
- P1 regression: get_dirty_entries acquires backend._lock for thread-safety
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

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
    result = backend.get("M-store", namespace="default")
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
    result = backend.get("M-yaml-store", namespace="default")
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
    backend.update("M-seq2", content="seq two updated", namespace="default")
    # Now since_seq=1 should return only M-seq2
    dirty_one = DeltaTracker.get_dirty_entries(backend, since_seq=1)
    ids = {e.id for e in dirty_one}
    assert "M-seq2" in ids
    assert "M-seq1" not in ids
    backend.close()


def test_fallback_dirty_scan_expands_past_synced_prefix() -> None:
    synced = MemoryEntry(
        id="M-synced",
        content="already synced",
        sync_seq=2,
        last_synced_at=datetime.now(tz=timezone.utc),
    )
    dirty = MemoryEntry(id="M-old-dirty", content="must not starve", sync_seq=3)

    class _GrowingFallbackBackend:
        def __init__(self) -> None:
            self.counts = iter((10_001, 10_003, 10_003))
            self.limits: list[int] = []

        def count(self) -> int:
            return next(self.counts)

        def list_entries(self, *, limit: int) -> list[MemoryEntry]:
            self.limits.append(limit)
            return [synced] if limit < 10_004 else [synced, dirty]

    backend = _GrowingFallbackBackend()

    result = DeltaTracker.get_dirty_entries(backend, since_seq=2)  # type: ignore[arg-type]

    assert [entry.id for entry in result] == ["M-old-dirty"]
    assert backend.limits == [10_002, 10_004]


def test_fallback_dirty_scan_is_bounded_when_backend_keeps_growing() -> None:
    dirty = MemoryEntry(id="M-dirty", content="latest bounded snapshot", sync_seq=1)

    class _GrowingFallbackBackend:
        def __init__(self) -> None:
            self.current_count = 100
            self.limits: list[int] = []

        def count(self) -> int:
            self.current_count += 2
            return self.current_count

        def list_entries(self, *, limit: int) -> list[MemoryEntry]:
            self.limits.append(limit)
            return [dirty]

    backend = _GrowingFallbackBackend()

    result = DeltaTracker.get_dirty_entries(backend)  # type: ignore[arg-type]

    assert [entry.id for entry in result] == ["M-dirty"]
    assert backend.limits == [103, 105, 107]


def test_sqlite_update_recomputes_sync_hash(tmp_path: Path) -> None:
    """FR06: SQLite updates recompute sync_hash and clear last_synced_at."""
    backend = SQLiteBackend(tmp_path / "test.db")
    entry = MemoryEntry(id="M-update", content="before")
    backend.store(entry)
    DeltaTracker.mark_synced(["M-update"], backend, namespace="default")
    before = backend.get("M-update", namespace="default")
    assert before is not None

    backend.update("M-update", content="after", namespace="default")

    after = backend.get("M-update", namespace="default")
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
    DeltaTracker.mark_synced(["M-yaml-update"], backend, namespace="default")
    before = backend.get("M-yaml-update", namespace="default")
    assert before is not None

    backend.update("M-yaml-update", content="after", namespace="default")

    after = backend.get("M-yaml-update", namespace="default")
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

    count = DeltaTracker.mark_synced(["M-sync1", "M-sync2"], backend, namespace="default")
    assert count == 2

    r1 = backend.get("M-sync1", namespace="default")
    r2 = backend.get("M-sync2", namespace="default")
    assert r1 is not None and r1.last_synced_at is not None
    assert r2 is not None and r2.last_synced_at is not None
    backend.close()


def test_mark_synced_nonexistent_entry(tmp_path: Path) -> None:
    """FR05: mark_synced silently skips non-existent entries."""
    backend = SQLiteBackend(tmp_path / "test.db")
    count = DeltaTracker.mark_synced(["M-nonexist"], backend, namespace="default")
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
    DeltaTracker.mark_synced(["M-clean"], backend, namespace="default")

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
    result = backend.get("M-rt", namespace="default")
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
    result = backend.get("M-yaml-rt", namespace="default")
    assert result is not None
    # FR06: auto dirty-marked
    assert result.sync_seq >= 1
    assert result.sync_hash != ""
    assert result.last_synced_at is None
    backend.close()


# ---------------------------------------------------------------------------
# P1 regression: get_dirty_entries acquires backend._lock
# ---------------------------------------------------------------------------


def test_get_dirty_entries_acquires_lock(tmp_path: Path) -> None:
    """P1 regression: get_dirty_entries must hold backend._lock during the
    SQL execute to prevent races with concurrent writes on the same connection.

    Verifies that _lock is acquired by replacing it with an instrumented
    threading.Lock subclass that records acquisition events.
    """
    backend = SQLiteBackend(tmp_path / "lock_test.db")
    entry = MemoryEntry(id="M-lock-1", content="lock test entry")
    backend.store(entry)

    lock_acquired_during_query = threading.Event()
    original_lock = backend._lock

    class InstrumentedLock:
        """Wraps the real lock and records when it is held during get_dirty_entries."""

        def __enter__(self) -> InstrumentedLock:
            original_lock.acquire()
            lock_acquired_during_query.set()
            return self

        def __exit__(self, *args: object) -> None:
            original_lock.release()

        def acquire(self, *args: object, **kwargs: object) -> bool:
            result = original_lock.acquire(*args, **kwargs)  # type: ignore[call-overload]
            lock_acquired_during_query.set()
            return result

        def release(self) -> None:
            original_lock.release()

    backend._lock = InstrumentedLock()  # type: ignore[assignment]
    try:
        dirty = DeltaTracker.get_dirty_entries(backend, since_seq=0)
        assert len(dirty) >= 1, "Should return the stored dirty entry"
        assert lock_acquired_during_query.is_set(), "get_dirty_entries must acquire backend._lock during SQL execute"
    finally:
        backend._lock = original_lock
        backend.close()


# ---------------------------------------------------------------------------
# Wave 11 gap-fill: uncovered lines in sync/delta.py
# ---------------------------------------------------------------------------


def test_normalize_hash_value_datetime() -> None:
    """_normalize_hash_value converts datetime to UTC ISO string (line 48)."""
    from datetime import timezone

    from trw_memory.sync.delta import _normalize_hash_value  # type: ignore[attr-defined]

    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = _normalize_hash_value(dt)
    assert isinstance(result, str)
    assert "2024-06-01" in result


def test_mark_dirty_existing_entry_increments_seq(tmp_path: Path) -> None:
    """mark_dirty on an existing entry increments sync_seq and recomputes hash."""
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.store(MemoryEntry(id="M-dirty-1", content="initial content"))
    entry_before = backend.get("M-dirty-1", namespace="default")
    assert entry_before is not None
    seq_before = entry_before.sync_seq

    DeltaTracker.mark_dirty("M-dirty-1", backend, namespace="default")

    entry_after = backend.get("M-dirty-1", namespace="default")
    assert entry_after is not None
    assert entry_after.sync_seq == seq_before + 1
    assert entry_after.sync_hash != ""
    backend.close()


def test_mark_dirty_nonexistent_entry_is_noop(tmp_path: Path) -> None:
    """mark_dirty on a missing entry returns early without error (line 76)."""
    backend = SQLiteBackend(tmp_path / "test.db")
    DeltaTracker.mark_dirty("DOES-NOT-EXIST", backend, namespace="default")
    backend.close()


def test_get_dirty_entries_without_lock(tmp_path: Path) -> None:
    """get_dirty_entries takes the no-lock branch when backend._lock is None (line 104)."""
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.store(MemoryEntry(id="M-nl-1", content="no-lock entry"))
    original_lock = backend._lock  # type: ignore[attr-defined]
    backend._lock = None  # type: ignore[assignment]
    try:
        dirty = DeltaTracker.get_dirty_entries(backend, since_seq=0)
        assert any(e.id == "M-nl-1" for e in dirty)
    finally:
        backend._lock = original_lock
        backend.close()


def test_get_dirty_entries_fallback_non_sqlite(tmp_path: Path) -> None:
    """get_dirty_entries falls back to list_entries when backend has no _conn (line 107-108)."""
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path)
    backend.store(MemoryEntry(id="M-yaml-dirty", content="yaml dirty entry"))
    dirty = DeltaTracker.get_dirty_entries(backend, since_seq=0)
    assert any(e.id == "M-yaml-dirty" for e in dirty)


def test_mark_synced_exception_logs_warning_and_continues(tmp_path: Path) -> None:
    """mark_synced logs warning and continues when backend.update raises (lines 120-121)."""
    import structlog.testing

    from trw_memory.sync.delta import DeltaTracker

    backend = SQLiteBackend(tmp_path / "test.db")
    backend.store(MemoryEntry(id="M-ok-1", content="ok entry"))
    backend.store(MemoryEntry(id="M-ok-2", content="ok entry 2"))

    original_update = backend.update

    def _raise_on_first(entry_id: str, **kwargs: object) -> object:
        if entry_id == "M-ok-1":
            raise RuntimeError("simulated update failure")
        return original_update(entry_id, **kwargs)

    backend.update = _raise_on_first  # type: ignore[method-assign]

    with structlog.testing.capture_logs() as logs:
        count = DeltaTracker.mark_synced(["M-ok-1", "M-ok-2"], backend, namespace="default")

    assert count == 1
    warning_events = [l["event"] for l in logs if l.get("log_level") == "warning"]
    assert "delta_mark_synced_failed" in warning_events
    backend.close()
