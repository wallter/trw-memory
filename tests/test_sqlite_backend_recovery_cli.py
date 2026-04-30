"""Focused sqlite3 CLI fallback tests for SQLite backend recovery."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_sqlite_backend_recovery_support import (
    _corrupt_sqlite_master,
    _make_fake_row,
    _populate_db,
)


def test_fr04_recover_cli_salvage_succeeds_when_select_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    now = "2026-04-13T00:00:00+00:00"
    fake_row = _make_fake_row(
        {
            "id": "L-rescued-via-cli",
            "content": "rescued content",
            "created_at": now,
            "updated_at": now,
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT id FROM memories").fetchone()[0] == "L-rescued-via-cli"
    finally:
        conn.close()


def test_recovery_drops_unknown_columns_preventing_sql_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    now = "2026-04-18T00:00:00+00:00"
    poisoned_row = _make_fake_row(
        {
            "id": "L-legit",
            "content": "legit content",
            "created_at": now,
            "updated_at": now,
            "id, content); DROP TABLE memories; --": "payload",
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [poisoned_row]),
    )
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT id FROM memories").fetchone()[0] == "L-legit"
    finally:
        conn.close()


def test_fr04_recover_cli_unavailable_falls_through_to_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_fnf(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("sqlite3 not installed")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_timeout_falls_through_to_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="sqlite3", timeout=30)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_nonzero_exit_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _nonzero(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=1, stdout=b"", stderr=b"err")

    monkeypatch.setattr(subprocess, "run", _nonzero)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_full_executescript_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    dump_sql = b"""
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO memories (id, content, created_at, updated_at)
VALUES ('L-from-dump', 'rescued from cli dump', '2026-04-13T00:00:00+00:00', '2026-04-13T00:00:00+00:00');
"""

    def _valid_dump(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=0, stdout=dump_sql, stderr=b"")

    monkeypatch.setattr(subprocess, "run", _valid_dump)
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        row = conn.execute("SELECT id, content FROM memories").fetchone()
        assert row[0] == "L-from-dump"
        assert row[1] == "rescued from cli dump"
    finally:
        conn.close()


def test_fr04_recover_cli_malformed_dump_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _bad_dump(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=0, stdout=b"THIS IS NOT VALID SQL AT ALL;;", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _bad_dump)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_empty_dump_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _empty_stdout(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=0, stdout=b"   \n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _empty_stdout)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_nfr03_subprocess_called_without_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    recorded: dict[str, Any] = {}

    def _record(*args: Any, **kwargs: Any) -> Any:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0] if args else [], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _record)
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    cmd = recorded["args"][0]
    assert isinstance(cmd, list)
    assert cmd[0] == "sqlite3"
    assert recorded["kwargs"].get("shell", False) is False
