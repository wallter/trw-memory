"""Diagnostics may mutate a private copy, never initialize or harden the source."""

from __future__ import annotations

import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from trw_memory.storage.readonly_snapshot import readonly_memory_snapshot


def test_snapshot_includes_wal_and_cleans_up_without_source_mutation(tmp_path: Path) -> None:
    path = tmp_path / "source ?#.db"
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
        writer.execute("INSERT INTO memories VALUES ('committed-in-wal')")
        writer.commit()
        path.chmod(0o640)
        before = path.read_bytes()
        with readonly_memory_snapshot(path, temporary_root=tmp_path) as snapshot:
            assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
            assert stat.S_IMODE(snapshot.parent.stat().st_mode) == 0o700
            with closing(sqlite3.connect(snapshot)) as copy:
                assert copy.execute("SELECT id FROM memories").fetchall() == [("committed-in-wal",)]
                copy.execute("DELETE FROM memories")
            assert writer.execute("SELECT count(*) FROM memories").fetchone() == (1,)
        assert not snapshot.parent.exists()
        assert path.read_bytes() == before
        assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_source_is_not_initialized(tmp_path: Path, exists: bool) -> None:
    source = tmp_path / "source.db"
    if exists:
        source.touch(mode=0o640)
    with pytest.raises((sqlite3.Error, ValueError)):
        with readonly_memory_snapshot(source, temporary_root=tmp_path):
            pytest.fail("Uninitialized source accepted")
    assert source.exists() is exists
    if exists:
        assert source.read_bytes() == b""
        assert stat.S_IMODE(source.stat().st_mode) == 0o640
    assert not list(tmp_path.glob("trw-memory-audit-*"))


def test_consumer_failure_removes_private_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE memories (id TEXT)")
    with pytest.raises(RuntimeError, match="consumer failed"):
        with readonly_memory_snapshot(source, temporary_root=tmp_path) as snapshot:
            raise RuntimeError("consumer failed")
    assert not snapshot.parent.exists()


def test_expired_copy_budget_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE memories (id TEXT)")
    times = iter([0.0, 31.0])
    monkeypatch.setattr("trw_memory.storage.readonly_snapshot.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="time budget"):
        with readonly_memory_snapshot(source, temporary_root=tmp_path):
            pytest.fail("Expired snapshot accepted")
    assert not list(tmp_path.glob("trw-memory-audit-*"))


def test_analysis_does_not_hold_source_read_transaction(tmp_path: Path) -> None:
    source = tmp_path / "live.db"
    with closing(sqlite3.connect(source, timeout=0.1)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE memories (id TEXT)")
        writer.execute("INSERT INTO memories VALUES ('before')")
        writer.commit()
        with readonly_memory_snapshot(source, temporary_root=tmp_path) as snapshot:
            writer.execute("INSERT INTO memories VALUES ('after')")
            writer.commit()
            assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
            with closing(sqlite3.connect(snapshot)) as copy:
                assert copy.execute("SELECT id FROM memories").fetchall() == [("before",)]


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(tmp_path: Path, budget: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        with readonly_memory_snapshot(tmp_path / "missing.db", timeout_seconds=budget):
            pytest.fail("Invalid budget accepted")


def test_exclusive_lock_times_out_without_leaking_copy(tmp_path: Path) -> None:
    source = tmp_path / "locked.db"
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("CREATE TABLE memories (id TEXT)")
        writer.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with readonly_memory_snapshot(source, temporary_root=tmp_path, timeout_seconds=0.05):
                pytest.fail("Locked database accepted")
        assert time.monotonic() - started < 2.0  # Scheduling tolerance, not a hard SLA.
        assert not list(tmp_path.glob("trw-memory-audit-*"))


@pytest.mark.parametrize("writable_directory", [True, False])
def test_committed_wal_without_shm_is_read_or_fails_cleanly(tmp_path: Path, writable_directory: bool) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    path = directory / "memory.db"
    # A terminated fixture process leaves a committed WAL without any live owner.
    code = (
        "import sqlite3,sys,os; c=sqlite3.connect(sys.argv[1]); "
        "c.execute('PRAGMA journal_mode=WAL'); c.execute('CREATE TABLE memories (id TEXT)'); "
        "c.execute(\"INSERT INTO memories VALUES ('in-wal')\"); c.commit(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, timeout=10)
    Path(str(path) + "-shm").unlink()
    before = path.read_bytes()
    path.chmod(0o400)
    if not writable_directory:
        directory.chmod(0o500)
    try:
        try:
            with readonly_memory_snapshot(path, temporary_root=tmp_path) as snapshot:
                with closing(sqlite3.connect(snapshot)) as copy:
                    assert copy.execute("SELECT id FROM memories").fetchall() == [("in-wal",)]
        except sqlite3.OperationalError:
            if writable_directory:
                raise
            # SQLite may require a writable directory to reconstruct its shm.
        assert path.read_bytes() == before
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        assert not list(tmp_path.glob("trw-memory-audit-*"))
    finally:
        directory.chmod(0o700)
        path.chmod(0o600)
