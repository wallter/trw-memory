"""Shared helpers for split encryption tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.keys import clear_key_cache

_KEY_LENGTH = 32


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


def _make_entry(
    entry_id: str = "enc-test-1",
    content: str = "test content",
    detail: str = "",
    namespace: str = "default",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        namespace=namespace,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


class _StaticCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None


class _RotatingSQLCipherConnection:
    def __init__(
        self,
        conn: sqlite3.Connection,
        statements: list[str],
        db_path: Path,
        *,
        integrity_result: str = "ok",
        mutate_on_rekey: bytes | None = None,
        wal_checkpoint_busy: bool = False,
        raise_on_rekey: bool = False,
    ) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_statements", statements)
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_integrity_result", integrity_result)
        object.__setattr__(self, "_mutate_on_rekey", mutate_on_rekey)
        object.__setattr__(self, "_wal_checkpoint_busy", wal_checkpoint_busy)
        object.__setattr__(self, "_raise_on_rekey", raise_on_rekey)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._conn, name, value)

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor | _StaticCursor:
        self._statements.append(sql)
        normalized = sql.strip().upper()
        if normalized.startswith("PRAGMA WAL_CHECKPOINT") and self._wal_checkpoint_busy:
            # Simulate another connection holding the WAL (busy=1).
            return _StaticCursor([(1, 0, 0)])
        if normalized.startswith("PRAGMA REKEY") and self._raise_on_rekey:
            # Simulate a SQLCipher driver that echoes the failing SQL — which
            # for a rekey embeds the new key hex — straight into the exception.
            raise sqlite3.OperationalError(f"near rekey: syntax error in {sql!r}")
        if normalized.startswith("PRAGMA REKEY") and self._mutate_on_rekey is not None:
            with self._db_path.open("ab") as handle:
                handle.write(self._mutate_on_rekey)
            return _StaticCursor([])
        if normalized.startswith("PRAGMA INTEGRITY_CHECK"):
            return _StaticCursor([(self._integrity_result,)])
        return self._conn.execute(sql, *args)


class _RotatingSQLCipherDBAPI:
    Error = sqlite3.Error
    DatabaseError = sqlite3.DatabaseError

    def __init__(
        self,
        statements: list[str],
        *,
        integrity_result: str = "ok",
        mutate_on_rekey: bytes | None = None,
        wal_checkpoint_busy: bool = False,
        raise_on_rekey: bool = False,
    ) -> None:
        self._statements = statements
        self._integrity_result = integrity_result
        self._mutate_on_rekey = mutate_on_rekey
        self._wal_checkpoint_busy = wal_checkpoint_busy
        self._raise_on_rekey = raise_on_rekey

    def connect(self, database: str, **kwargs: object) -> _RotatingSQLCipherConnection:
        conn = sqlite3.connect(database, **kwargs)
        return _RotatingSQLCipherConnection(
            conn,
            self._statements,
            Path(database),
            integrity_result=self._integrity_result,
            mutate_on_rekey=self._mutate_on_rekey,
            wal_checkpoint_busy=self._wal_checkpoint_busy,
            raise_on_rekey=self._raise_on_rekey,
        )


def _load_real_sqlcipher_driver_or_skip() -> object:
    from trw_memory.storage.sqlite_backend import _import_sqlcipher_driver

    try:
        return _import_sqlcipher_driver()
    except Exception as exc:
        pytest.skip(f"real SQLCipher driver unavailable: {exc}")
