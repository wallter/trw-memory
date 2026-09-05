"""SQLite connection-management helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — the public API surface (``SQLiteBackend._connect``,
``SQLiteBackend._open_and_configure``,
``SQLiteBackend._open_without_integrity_check``,
``SQLiteBackend._db_has_data``) is preserved by parent re-export
delegators.

5 helpers:

- ``connect`` — base ``dbapi.connect`` with WAL/synchronous defaults
  + sqlcipher key-pragma application when ``sqlcipher_key_hex`` is
  provided.
- ``open_and_configure`` — open + WAL mode + retry-once quick_check.
- ``open_without_integrity_check`` — open without quick_check (reserved
  for explicit SQLite lock/busy contention; structural quick_check failures
  must recover instead of continuing against a damaged B-tree).
- ``open_probe`` — short-lived open for the two probes below, carrying the
  same PRAGMA profile as every other open path.
- ``db_has_data`` — non-destructive row-count probe.

Extracted as PRD-DIST-245 Phase 1 batch 82.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Cap WAL file growth so a stalled checkpoint cannot let the WAL grow unbounded
# (a large stale WAL widens the window for WAL-reset inconsistency). 64 MiB.
#
# Read this together with trw-mcp's ``wal_checkpoint_threshold_mb`` (default
# 10 MB), which is the size at which trw-mcp DECIDES a checkpoint is due. The
# two numbers are 6.4x apart and mean different things, and the gap is exactly
# what an operator sees on an engine below SQLite 3.51.3: PASSIVE is the only
# permitted mode there (``_wal_checkpoint.normalize_mode``), PASSIVE writes
# frames back but never truncates, so the file climbs to THIS cap and stays,
# while the 10 MB trigger keeps firing to no visible effect. That is not drift
# between the two constants — it is the documented consequence of the engine
# gate, and the remedy is the engine upgrade named in
# ``_wal_checkpoint.WAL_RESET_UNSAFE_REMEDY``.
WAL_JOURNAL_SIZE_LIMIT_BYTES = 67108864
# Lock-wait window applied to every open path so a transient checkpoint/writer
# does not raise "database is locked" immediately.
_BUSY_TIMEOUT_MS = 30000


# 64 MiB shared page cache — trades memory for fewer disk reads on hot pages.
# Negative value = KiB (SQLite convention); 65536 KiB = 64 MiB.
_CACHE_SIZE_KB = -65536
# 1 GiB memory-map for read I/O — OS maps the file into virtual address space
# so reads bypass the kernel page-cache round-trip. Only benefits file-backed
# databases; effectively a no-op on `:memory:` connections.
_MMAP_SIZE_BYTES = 1073741824  # 1 GiB
# Raise WAL auto-checkpoint threshold from the default 1000 pages (4 MiB) to
# 4000 pages (16 MiB). Reduces checkpoint pressure during bulk writes without
# risking runaway WAL growth (cap is still enforced by journal_size_limit=64MiB).
# NOTE: keep this below journal_size_limit so the WAL cap never fires mid-write.
# The existing single-connection checkpoint serialiser (_wal_checkpoint.py) is
# unchanged — this PRAGMA only adjusts the automatic background trigger point.
_WAL_AUTOCHECKPOINT_PAGES = 4000


def apply_open_pragmas(conn: Any, *, verify: bool = False) -> None:
    """Apply the standard durable-open PRAGMA profile to *conn*.

    The single source of truth for the open profile shared by
    ``open_and_configure``, ``open_without_integrity_check``, and the recovered
    connection in ``_recovery._open_recovered_conn``: busy_timeout, WAL journal
    mode, NORMAL synchronous, and a bounded WAL size limit.

    When *verify* is True the WAL/synchronous results are checked and a warning
    is logged if the engine did not honour them (used on the primary open path).
    """
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    wal_result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if verify and wal_result and wal_result[0] != "wal":
        logger.warning("wal_mode_not_enabled", got=wal_result[0])
    sync_result = conn.execute("PRAGMA synchronous=NORMAL").fetchone()
    if verify and sync_result and sync_result[0] not in ("1", 1):
        logger.warning("synchronous_normal_not_set", got=sync_result[0] if sync_result else None)
    conn.execute(f"PRAGMA journal_size_limit = {WAL_JOURNAL_SIZE_LIMIT_BYTES}")
    # Performance tuning: cache + mmap + checkpoint threshold.
    # safe for all connection types; mmap_size is effectively a no-op on :memory:.
    conn.execute(f"PRAGMA cache_size = {_CACHE_SIZE_KB}")
    conn.execute(f"PRAGMA mmap_size = {_MMAP_SIZE_BYTES}")
    conn.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
    conn.execute("PRAGMA temp_store = MEMORY")


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
    # Use the caller-provided ``dbapi`` for the Row factory so the type
    # matches the cursor. With the pysqlite3 shim live, ``dbapi`` is
    # usually pysqlite3 — but tests can pass stdlib ``sqlite3`` explicitly
    # to drive deterministic exception classes, and any cross-module row
    # factory would raise ``TypeError: Row() argument 1 must be
    # sqlite3.Cursor, not pysqlite3.dbapi2.Cursor`` (or vice versa).
    conn.row_factory = getattr(dbapi, "Row", sqlite3.Row)
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
    try:
        apply_open_pragmas(conn, verify=True)

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
        raise sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.close()
        raise


def open_without_integrity_check(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> Any:
    """Open a connection skipping integrity check for explicit lock/busy contention only."""
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
        sqlcipher_key_hex=sqlcipher_key_hex,
    )
    apply_open_pragmas(conn)
    return conn


#: Lock-wait for the short-lived probe opens below. Deliberately shorter than
#: the 30 s the full open paths allow: a probe that cannot get in promptly
#: should report that, not stall a caller's boot.
_PROBE_TIMEOUT_SECONDS = 5.0


def open_probe(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> Any:
    """Open a short-lived probe connection with the standard PRAGMA profile.

    The shared open path for :func:`check_integrity` and :func:`db_has_data`.
    Both used to call :func:`connect` directly and skip
    :func:`apply_open_pragmas`, so a probe connection ran without
    ``journal_size_limit``: it could append WAL frames (a probe still runs the
    recovery/checkpoint machinery of the engine) with the 64 MiB cap that every
    other open path sets left unset. A per-callsite PRAGMA list is exactly how
    that cap gets opted out of by accident, so there is one open path instead.
    """
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check_same_thread=True,
        sqlcipher_key_hex=sqlcipher_key_hex,
    )
    try:
        apply_open_pragmas(conn)
    except BaseException:
        with contextlib.suppress(Exception):
            conn.close()
        raise
    return conn


def check_integrity(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> dict[str, object]:
    """Check database integrity without opening a full backend.

    Re-exported as ``SQLiteBackend.check_integrity`` for back-compat.

    Returns:
        Dict with ``ok`` (bool), ``detail`` (str), and ``db_path``.
    """
    conn: Any = None
    try:
        conn = open_probe(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
        rows = conn.execute("PRAGMA quick_check").fetchall()
        healthy = len(rows) == 1 and rows[0][0] == "ok"
        return {"ok": healthy, "detail": rows[0][0] if rows else "empty", "db_path": str(db_path)}
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "detail": str(exc), "db_path": str(db_path)}
    finally:
        # Close in finally so a non-sqlite exception (KeyboardInterrupt, MemoryError)
        # during quick_check cannot leak the connection.
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()


def db_has_data(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
) -> bool:
    """Probe whether the DB at ``db_path`` has any rows in ``memories``.

    Non-destructive: this proves rows are readable; it does not prove the
    database is structurally healthy after a failed quick_check.
    """
    conn: Any = None
    try:
        conn = open_probe(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
        count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        return bool(count > 0)
    except sqlite3.Error:
        return False
    finally:
        # Close in finally so an unexpected (non-sqlite) exception cannot leak it.
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
