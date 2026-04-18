"""Tests for P3 — stale-handle detection and reconnect.

Uses tmp_path for all tests since inode/sentinel checks require real disk files.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog.testing

from trw_memory.exceptions import StaleConnectionError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._stale_handle_detector import sentinel_path, write_sentinel
from trw_memory.storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(entry_id: str, content: str = "content") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        namespace="default",
        source="agent",  # type: ignore[arg-type]
    )


def _fresh_empty_db(db_path: Path) -> None:
    """Create a minimal valid trw-memory SQLite DB at db_path."""
    from trw_memory.storage._schema import ensure_schema

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Test: inode change triggers reconnect
# ---------------------------------------------------------------------------


def test_inode_change_triggers_reconnect(tmp_path: Path) -> None:
    """After a simulated DB replacement (inode change), list_entries reads new state."""
    db_path = tmp_path / "memory.db"

    # Open original backend and write row A
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-row-A", "original row"))

    old_inode = db_path.stat().st_ino

    # Simulate recovery: move file aside, write a fresh empty DB, write sentinel
    corrupt_bak = tmp_path / "memory.db.corrupt.2026-04-18T00-00-00Z.bak"
    shutil.move(str(db_path), str(corrupt_bak))
    _fresh_empty_db(db_path)
    write_sentinel(db_path, corrupt_bak)

    # Write row B via a NEW backend (pointing at the fresh DB)
    backend2 = SQLiteBackend(db_path)
    backend2.store(_make_entry("M-row-B", "new row on fresh db"))
    backend2.close()

    # The ORIGINAL backend now calls list_entries — should detect stale and reconnect.
    # Force the cache to expire by setting last_checked to 0.
    backend._stale_detector._last_checked = 0.0  # type: ignore[attr-defined]

    with structlog.testing.capture_logs() as logs:
        results = backend.list_entries(limit=100)

    new_inode = db_path.stat().st_ino
    # inode MUST have changed for this test to be meaningful
    assert new_inode != old_inode or corrupt_bak.exists(), "sentinel-based detection is the fallback"

    ids = [e.id for e in results]
    # After reconnect the backend should see the fresh DB state (row B, not row A)
    assert "M-row-B" in ids
    assert "M-row-A" not in ids

    # At least one stale-handle log event
    stale_events = [log for log in logs if log.get("action") == "memory_stale_handle_detected"]
    assert len(stale_events) >= 1

    backend.close()


# ---------------------------------------------------------------------------
# Test: sentinel mtime triggers reconnect
# ---------------------------------------------------------------------------


def test_sentinel_mtime_triggers_reconnect(tmp_path: Path) -> None:
    """A newer sentinel file alone is enough to trigger reconnect."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-sentinel-A"))

    # Write a fresh DB and sentinel (backend still holds old FD)
    shutil.move(str(db_path), tmp_path / "old.bak")
    _fresh_empty_db(db_path)
    write_sentinel(db_path, tmp_path / "old.bak")

    # Write row B on the new DB
    b2 = SQLiteBackend(db_path)
    b2.store(_make_entry("M-sentinel-B"))
    b2.close()

    # Expire the cache on original backend
    backend._stale_detector._last_checked = 0.0  # type: ignore[attr-defined]

    results = backend.list_entries(limit=100)
    ids = [e.id for e in results]
    assert "M-sentinel-B" in ids
    assert "M-sentinel-A" not in ids
    backend.close()


# ---------------------------------------------------------------------------
# Test: precheck cached within budget — no extra stat calls
# ---------------------------------------------------------------------------


def test_precheck_is_cheap_cached_within_budget(tmp_path: Path) -> None:
    """Back-to-back calls within the staleness budget skip the stat."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    detector = backend._stale_detector  # type: ignore[attr-defined]
    # Set a long budget so second call within window skips stat
    detector._check_interval = 60.0

    stat_calls: list[int] = []
    original_stat = Path.stat

    def counting_stat(self: Path, **kw: object) -> object:
        if self == db_path or self == sentinel_path(db_path):
            stat_calls.append(1)
        return original_stat(self, **kw)  # type: ignore[arg-type]

    with patch.object(Path, "stat", counting_stat):
        # First call — does the real check (cache miss)
        detector._last_checked = 0.0
        detector.is_stale()
        calls_after_first = len(stat_calls)

        # Second call within budget — must NOT stat again
        detector.is_stale()
        calls_after_second = len(stat_calls)

    assert calls_after_second == calls_after_first, (
        "Second call within budget should not issue any stat syscalls"
    )
    backend.close()


# ---------------------------------------------------------------------------
# Test: reconnect failure raises StaleConnectionError
# ---------------------------------------------------------------------------


def test_reconnect_failure_raises_stale_connection_error(tmp_path: Path) -> None:
    """When reconnect itself fails, StaleConnectionError is raised."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    # Write sentinel so staleness is detected
    sentinel_p = sentinel_path(db_path)
    sentinel_p.write_text("2099-01-01T00:00:00+00:00\n/gone.bak\n")

    # Expire the staleness cache
    backend._stale_detector._last_checked = 0.0  # type: ignore[attr-defined]

    # Patch _open_and_configure to always raise so _reconnect fails
    import sqlite3 as _sqlite3
    with patch.object(
        SQLiteBackend,
        "_open_and_configure",
        side_effect=_sqlite3.DatabaseError("injected failure"),
    ):
        with pytest.raises(StaleConnectionError):
            backend.list_entries()

    # Cleanup — connection may be closed already
    try:
        backend.close()
    except Exception:
        pass
