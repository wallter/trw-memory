"""SQLite connection-management helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — the public API surface (``SQLiteBackend._connect``,
``SQLiteBackend._open_and_configure``,
``SQLiteBackend._open_without_integrity_check``,
``SQLiteBackend._db_has_data``) is preserved by parent re-export
delegators.

4 helpers:

- ``connect`` — base ``dbapi.connect`` with WAL/synchronous defaults
  + sqlcipher key-pragma application when ``sqlcipher_key_hex`` is
  provided.
- ``open_and_configure`` — open + WAL mode + retry-once quick_check.
- ``open_without_integrity_check`` — open without quick_check (used
  when DB has data but quick_check fails transiently).
- ``db_has_data`` — non-destructive row-count probe.

Extracted as PRD-DIST-245 Phase 1 batch 82.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _apply_sqlcipher_pragmas_safe(conn: Any) -> None:
    """Apply the sqlcipher KDF + cipher pragmas via the parent module."""
    from trw_memory.storage import sqlite_backend as _sqlite_backend_module

    _sqlite_backend_module._apply_sqlcipher_pragmas(conn)


def connect(
    db_path: Path,
    *,
    dbapi: Any,
    timeout: float,
    check_same_thread: bool,
    cached_statements: int | None = None,
    sqlcipher_key_hex: str | None = None,
) -> Any:
    """Base sqlite connection with WAL/synchronous defaults + optional sqlcipher key."""
    kwargs: dict[str, object] = {
        "timeout": timeout,
        "check_same_thread": check_same_thread,
    }
    if cached_statements is not None:
        kwargs["cached_statements"] = cached_statements
    conn = dbapi.connect(str(db_path), **kwargs)
    conn.row_factory = sqlite3.Row
    if sqlcipher_key_hex is not None:
        if len(sqlcipher_key_hex) != 64 or any(ch not in "0123456789abcdef" for ch in sqlcipher_key_hex):
            raise ValueError("sqlcipher_key_hex must be a 64-character lowercase hex string")
        conn.execute(f"PRAGMA key = \"x'{sqlcipher_key_hex}'\"")
        _apply_sqlcipher_pragmas_safe(conn)
        conn.execute("SELECT count(*) FROM sqlite_master")
    return conn


def open_and_configure(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> Any:
    """Open a connection with WAL mode and run a quick integrity check.

    Retries once on quick_check failure to handle transient WAL contention
    (e.g., MCP server mid-checkpoint while trw-maintain opens the DB).

    Raises:
        sqlite3.DatabaseError: If the database fails integrity check twice.
    """
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
        sqlcipher_key_hex=sqlcipher_key_hex,
    )
    conn.execute("PRAGMA busy_timeout = 30000")
    wal_result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if wal_result and wal_result[0] != "wal":
        logger.warning("wal_mode_not_enabled", got=wal_result[0])
    sync_result = conn.execute("PRAGMA synchronous=NORMAL").fetchone()
    if sync_result and sync_result[0] not in ("1", 1):
        logger.warning("synchronous_normal_not_set", got=sync_result[0] if sync_result else None)

    for attempt in range(2):
        rows = conn.execute("PRAGMA quick_check").fetchall()
        if len(rows) == 1 and rows[0][0] == "ok":
            return conn
        if attempt == 0:
            logger.warning(
                "integrity_check_retry",
                db=str(db_path),
                detail=rows[0][0] if rows else "empty",
            )
            time.sleep(1.0)

    conn.close()
    raise sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)")


def open_without_integrity_check(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> Any:
    """Open a connection skipping integrity check (transient WAL contention path)."""
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
        sqlcipher_key_hex=sqlcipher_key_hex,
    )
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_has_data(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> bool:
    """Probe whether the DB at ``db_path`` has any rows in ``memories``.

    Non-destructive: prevents auto-recovery from destroying a DB that has
    data but fails quick_check due to transient WAL contention.
    """
    try:
        conn = connect(
            db_path,
            dbapi=dbapi,
            timeout=5.0,
            check_same_thread=True,
            sqlcipher_key_hex=sqlcipher_key_hex,
        )
        try:
            count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
            conn.close()
            return bool(count > 0)
        except sqlite3.Error:
            conn.close()
            return False
    except sqlite3.Error:
        return False
