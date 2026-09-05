"""PRD-CORE-231-FR02: ``verification_status`` survives the storage round trip.

The FR08 auto-stale verdict was previously set on the in-memory recall payload
only. These tests pin the durable half of the contract: ``update()`` writes it,
a fresh connection reads it back, and clearing it writes ``NULL`` rather than
leaving the previous verdict in place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


def _entry(entry_id: str = "M-VS-100") -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(id=entry_id, content="anchored claim", created_at=now, updated_at=now)


def test_default_is_none(tmp_path: Path) -> None:
    """A freshly-stored entry records no adverse verdict."""
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(_entry())
    stored = backend.get("M-VS-100", namespace="default")
    assert stored is not None
    assert stored.verification_status is None


def test_persists_stale_on_threshold(tmp_path: Path) -> None:
    """``update(verification_status='stale')`` persists across a reconnect."""
    db_path = tmp_path / "m.db"
    backend = SQLiteBackend(db_path)
    backend.store(_entry())

    updated = backend.update("M-VS-100", verification_status="stale", namespace="default")
    assert updated is not None
    assert updated.verification_status == "stale"
    backend.close()

    # Fresh connection == process restart: the verdict must still be there.
    reopened = SQLiteBackend(db_path)
    reread = reopened.get("M-VS-100", namespace="default")
    assert reread is not None
    assert reread.verification_status == "stale"


def test_clearing_writes_null(tmp_path: Path) -> None:
    """Passing ``None`` overwrites a previous 'stale' verdict (last-write-wins)."""
    db_path = tmp_path / "m.db"
    backend = SQLiteBackend(db_path)
    backend.store(_entry())
    backend.update("M-VS-100", verification_status="stale", namespace="default")

    cleared = backend.update("M-VS-100", verification_status=None, namespace="default")
    assert cleared is not None
    assert cleared.verification_status is None
    backend.close()

    reopened = SQLiteBackend(db_path)
    reread = reopened.get("M-VS-100", namespace="default")
    assert reread is not None
    assert reread.verification_status is None


def test_out_of_contract_value_is_rejected(tmp_path: Path) -> None:
    """An unknown literal is refused at write time, never persisted."""
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(_entry())

    with pytest.raises(StorageError):
        backend.update("M-VS-100", verification_status="bogus", namespace="default")

    stored = backend.get("M-VS-100", namespace="default")
    assert stored is not None
    assert stored.verification_status is None


def test_hand_edited_column_degrades_to_none(tmp_path: Path) -> None:
    """A junk value written behind the backend's back reads back as ``None``."""
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(_entry())
    with backend._lock:
        backend._conn.execute("UPDATE memories SET verification_status = 'wat' WHERE id = ?", ("M-VS-100",))
        backend._conn.commit()

    stored = backend.get("M-VS-100", namespace="default")
    assert stored is not None
    assert stored.verification_status is None


def test_yaml_backend_round_trips_verification_status(tmp_path: Path) -> None:
    """The YAML backend keeps field parity with SQLite for the new column."""
    backend = YAMLBackend(tmp_path / "entries")
    backend.store(_entry("M-VS-YAML"))

    backend.update("M-VS-YAML", verification_status="stale", namespace="default")
    reread = YAMLBackend(tmp_path / "entries").get("M-VS-YAML", namespace="default")
    assert reread is not None
    assert reread.verification_status == "stale"
