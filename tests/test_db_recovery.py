"""Tests for SQLite database corruption detection and auto-recovery.

Covers:
- integrity_check detects healthy and corrupt databases
- recover_db salvages rows from a corrupt database
- recover_db creates an empty database when salvage fails
- __init__ auto-recovers on corruption
- check_integrity static utility
- WAL/SHM files cleaned up on recovery
- Backup rotation
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _make_entry(entry_id: str = "test-1", content: str = "hello") -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="detail text",
        tags=["test"],
        importance=0.5,
        status=MemoryStatus.ACTIVE,
        namespace="default",
        source="agent",
        created_at=now,
        updated_at=now,
    )


def _corrupt_db(db_path: Path) -> None:
    """Corrupt a database file by overwriting its header and B-tree pages."""
    data = db_path.read_bytes()
    # Overwrite the SQLite header (first 100 bytes) and first page
    corrupted = b"\x00\xff\xfe\xfd" * 512 + data[2048:]
    db_path.write_bytes(corrupted)


class TestIntegrityCheck:
    def test_healthy_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ok.db"
        backend = SQLiteBackend(db_path)
        assert backend._run_integrity_check() is True
        backend.close()

    def test_corrupt_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bad.db"
        # Create a valid database first, then corrupt it
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id TEXT)")
        conn.execute("INSERT INTO t VALUES ('a')")
        conn.commit()
        conn.close()
        _corrupt_db(db_path)

        # Open raw connection and check
        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is False


class TestCheckIntegrityStatic:
    def test_healthy(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ok.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id TEXT)")
        conn.close()

        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is True
        assert result["detail"] == "ok"
        assert result["db_path"] == str(db_path)

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        db_path = tmp_path / "does-not-exist.db"
        # sqlite3 creates the file on connect, so this returns ok
        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is True

    def test_corrupt(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id TEXT)")
        for i in range(100):
            conn.execute("INSERT INTO t VALUES (?)", (f"row-{i}",))
        conn.commit()
        conn.close()
        _corrupt_db(db_path)

        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is False


class TestRecoverDb:
    def test_recovers_rows_from_corrupt_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        # Create a valid backend and store an entry
        backend = SQLiteBackend(db_path)
        entry = _make_entry("L-0001", "important learning")
        backend.store(entry)
        backend.close()

        # Corrupt it
        _corrupt_db(db_path)

        # Recovery should produce a new valid database
        new_conn = SQLiteBackend.recover_db(db_path)
        new_conn.close()

        # Verify backup was created
        backup = db_path.with_suffix(".db.corrupt.bak")
        assert backup.exists()

        # The new database should be valid
        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is True

    def test_cleans_wal_shm_files(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        backend = SQLiteBackend(db_path)
        backend.store(_make_entry())
        backend.close()

        # Create fake WAL/SHM files
        wal = tmp_path / "memory.db-wal"
        shm = tmp_path / "memory.db-shm"
        wal.write_bytes(b"wal data")
        shm.write_bytes(b"shm data")

        _corrupt_db(db_path)
        new_conn = SQLiteBackend.recover_db(db_path)
        new_conn.close()

        assert not wal.exists()
        assert not shm.exists()

    def test_backup_rotation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"

        # First corruption + recovery
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
        conn.close()
        _corrupt_db(db_path)
        c1 = SQLiteBackend.recover_db(db_path)
        c1.close()

        backup1 = db_path.with_suffix(".db.corrupt.bak")
        assert backup1.exists()

        # Second corruption + recovery — old backup should be rotated
        _corrupt_db(db_path)
        c2 = SQLiteBackend.recover_db(db_path)
        c2.close()

        rotated = db_path.with_suffix(".db.corrupt.bak.1")
        assert rotated.exists()
        assert backup1.exists()

    def test_empty_db_when_salvage_fails(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        # Write total garbage — not even a valid SQLite file
        db_path.write_bytes(b"\x00" * 4096)

        new_conn = SQLiteBackend.recover_db(db_path)
        new_conn.close()

        result = SQLiteBackend.check_integrity(db_path)
        assert result["ok"] is True


class TestInitAutoRecovery:
    def test_auto_recovers_on_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"

        # Create a valid DB, store data, close
        backend = SQLiteBackend(db_path)
        entry = _make_entry("L-recover", "should survive")
        backend.store(entry)
        backend.close()

        # Corrupt it (header overwrite — _db_has_data returns False)
        _corrupt_db(db_path)

        # Constructor should detect corruption and auto-recover
        backend2 = SQLiteBackend(db_path)
        # Database should be healthy after recovery
        assert backend2._run_integrity_check() is True
        assert backend2.recovered is True
        assert backend2.integrity_warning is False
        backend2.close()

        # Backup should exist
        assert db_path.with_suffix(".db.corrupt.bak").exists()

    def test_preserves_db_with_data_on_transient_failure(self, tmp_path: Path, monkeypatch: object) -> None:
        """When quick_check fails but DB has readable data, open anyway
        instead of destroying the database with auto-recovery."""
        from unittest.mock import patch

        db_path = tmp_path / "memory.db"

        # Create valid DB with data
        backend = SQLiteBackend(db_path)
        entry = _make_entry("L-preserve", "must not be lost")
        backend.store(entry)
        backend.close()

        # Simulate transient quick_check failure by patching
        # _open_and_configure to always raise, while _db_has_data
        # returns True (DB file is fine, just quick_check fails)
        def _failing_open(db_path_arg: Path) -> None:
            raise sqlite3.DatabaseError("simulated transient failure")

        with patch.object(SQLiteBackend, "_open_and_configure", staticmethod(_failing_open)):
            backend2 = SQLiteBackend(db_path)
            # Should NOT auto-recover — DB has data
            assert backend2.recovered is False
            assert backend2.integrity_warning is True

            # Data should still be accessible
            result = backend2.get("L-preserve")
            assert result is not None
            assert result.content == "must not be lost"
            backend2.close()

            # No backup should be created
            assert not db_path.with_suffix(".db.corrupt.bak").exists()

    def test_db_has_data_returns_false_for_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        # Create empty DB (just schema, no rows)
        backend = SQLiteBackend(db_path)
        backend.close()
        assert SQLiteBackend._db_has_data(db_path) is False

    def test_db_has_data_returns_true_for_populated(self, tmp_path: Path) -> None:
        db_path = tmp_path / "populated.db"
        backend = SQLiteBackend(db_path)
        backend.store(_make_entry("L-data", "has data"))
        backend.close()
        assert SQLiteBackend._db_has_data(db_path) is True

    def test_db_has_data_returns_false_for_garbage(self, tmp_path: Path) -> None:
        db_path = tmp_path / "garbage.db"
        db_path.write_bytes(b"\x00" * 4096)
        assert SQLiteBackend._db_has_data(db_path) is False
