"""Shared helpers for split sqlite-backend recovery tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage import sqlite_backend as _sqlite_backend_module
from trw_memory.storage.sqlite_backend import SQLiteBackend

_TIMESTAMPED_BACKUP_RE = _sqlite_backend_module._TIMESTAMPED_BACKUP_RE


class _FakeRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __iter__(self) -> Any:
        return iter(self._data.values())


def _make_fake_row(data: dict[str, Any]) -> _FakeRow:
    return _FakeRow(data)


def _find_timestamped_backup(parent: Path) -> Path:
    matches = sorted(p for p in parent.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name))
    assert len(matches) == 1, f"expected exactly one timestamped backup in {parent}, got {[p.name for p in matches]}"
    return matches[0]


def _make_entry(entry_id: str = "L-0001", content: str = "test content") -> MemoryEntry:
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


def _corrupt_sqlite_master(db_path: Path) -> None:
    data = db_path.read_bytes()
    db_path.write_bytes(b"\x00\xff\xfe\xfd" * 512 + data[2048:])


def _populate_db(db_path: Path, *, entries: int = 3) -> None:
    backend = SQLiteBackend(db_path)
    for idx in range(entries):
        backend.store(_make_entry(entry_id=f"L-{idx:04d}", content=f"row {idx}"))
    backend.close()


class _FrozenDateTime:
    _frozen_utc: datetime = datetime(2026, 4, 13, 14, 37, 14, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        if tz is None:
            return cls._frozen_utc.replace(tzinfo=None)
        return cls._frozen_utc.astimezone(tz) if tz != timezone.utc else cls._frozen_utc

    @classmethod
    def set(cls, value: datetime) -> None:
        cls._frozen_utc = value


def _freeze_utc(monkeypatch: pytest.MonkeyPatch, when: datetime) -> None:
    _FrozenDateTime.set(when)
    monkeypatch.setattr(_sqlite_backend_module, "datetime", _FrozenDateTime)


def _write_timestamped_backup(parent: Path, ts_suffix: str, content: bytes = b"data") -> Path:
    path = parent / f"memory.db.corrupt.{ts_suffix}.bak"
    path.write_bytes(content)
    return path


def _write_legacy_backup(parent: Path, name: str, content: bytes = b"legacy") -> Path:
    path = parent / name
    path.write_bytes(content)
    return path


def _trigger_recovery(db_path: Path, monkeypatch: pytest.MonkeyPatch, *, keep_n: int = 5) -> Path:
    _populate_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    now = "2026-04-13T00:00:00+00:00"
    fake_row = _make_fake_row({"id": "L-ok", "content": "ok", "created_at": now, "updated_at": now})
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", corrupt_backup_keep=keep_n)
    conn.close()
    return _find_timestamped_backup(db_path.parent)
