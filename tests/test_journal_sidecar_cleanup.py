"""PRD-CORE-306 B71-59b (worker-1): ``restore_from_snapshot`` and ``recover_db``
clear a stale rollback ``-journal`` sidecar, not just ``-wal``/``-shm`` — a
stale journal replayed against a freshly-restored/recovered base file would
corrupt it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trw_memory.storage._recovery import recover_db
from trw_memory.storage._snapshot import restore_from_snapshot
from trw_memory.storage.sqlite_backend import SQLiteBackend


def test_restore_from_snapshot_clears_stale_journal(tmp_path: Path) -> None:
    base_dir = tmp_path
    snapshots_dir = base_dir / "memory" / "snapshots" / "daily"
    snapshots_dir.mkdir(parents=True)
    snapshot = snapshots_dir / "2026-09-01.db"
    snapshot.write_bytes(b"snapshot contents")

    db_path = base_dir / "memory.db"
    db_path.write_bytes(b"stale live db")
    journal_path = Path(f"{db_path}-journal")
    journal_path.write_bytes(b"stale rollback journal from a prior crash")
    assert journal_path.exists()

    restore_from_snapshot(base_dir, snapshot, db_path)

    assert not journal_path.exists(), "restore left a stale -journal sidecar in place"
    assert db_path.read_bytes() == b"snapshot contents"


def test_recover_db_clears_stale_journal(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()
    journal_path = Path(f"{db_path}-journal")
    journal_path.write_bytes(b"stale rollback journal")
    assert journal_path.exists()

    recover_db(db_path, dbapi=sqlite3, recovery_policy="empty_ok", corrupt_backup_keep=5, rebuild_from_cold=False)

    assert not journal_path.exists(), "recover_db left a stale -journal sidecar in place"
