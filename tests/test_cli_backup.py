"""Tests for PRD-CORE-311 FR02: `trw-memory backup create` (local-only leg)."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from trw_memory.cli_storage import handle_backup
from trw_memory.models.config import MemoryConfig


def _args(tmp_path: Path, db: Path) -> argparse.Namespace:
    return argparse.Namespace(namespace="default", db=str(db), backup_action="create")


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS memories (x INTEGER)")
    conn.execute("INSERT INTO memories VALUES (1)")
    conn.commit()
    conn.close()


def test_backup_create_prints_archive_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "memory.db"
    _make_db(db_path)

    rc = handle_backup(_args(tmp_path, db_path), config_cls=MemoryConfig)

    assert rc == 0
    out = capsys.readouterr().out
    assert "Created backup archive:" in out
    printed_path = Path(out.strip().rsplit(": ", 1)[1])
    assert printed_path.exists()
    assert printed_path.name.endswith(".db.gz")


def test_backup_create_refuses_beside_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A running daemon on the same store makes `backup create` exit 1 with the existing refusal."""
    from trw_memory.daemon import DaemonInfo

    db_path = tmp_path / "memory.db"
    _make_db(db_path)

    live_info = DaemonInfo(pid=1, url="http://127.0.0.1:1/mcp", started_at="2026-01-01T00:00:00Z", version="1.0.0")
    monkeypatch.setattr("trw_memory.cli_client.read_live_discovery", lambda paths: live_info)

    rc = handle_backup(_args(tmp_path, db_path), config_cls=MemoryConfig)

    assert rc == 1
    err = capsys.readouterr().err
    assert "daemon" in err.lower()
    backups_dir = tmp_path / "memory" / "backups"
    assert not backups_dir.exists() or not any(backups_dir.glob("*.db.gz"))


def test_backup_create_succeeds_once_daemon_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same store, with the daemon record gone, lets `backup create` succeed (paired with the refusal case)."""
    from trw_memory.daemon import DiscoveryAbsent

    db_path = tmp_path / "memory.db"
    _make_db(db_path)

    monkeypatch.setattr("trw_memory.cli_client.read_live_discovery", lambda paths: DiscoveryAbsent(reason="no record"))

    rc = handle_backup(_args(tmp_path, db_path), config_cls=MemoryConfig)

    assert rc == 0
    assert "Created backup archive:" in capsys.readouterr().out
