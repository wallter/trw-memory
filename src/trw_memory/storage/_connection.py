"""SQLite connection-management helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — the public API surface (``SQLiteBackend._connect``,
``SQLiteBackend._open_and_configure``,
``SQLiteBackend._open_without_integrity_check``,
``SQLiteBackend._db_has_data``) is preserved by parent re-export
delegators.

5 helpers:

- ``connect`` — base ``dbapi.connect`` with WAL/synchronous defaults.
- ``open_and_configure`` — open + WAL mode + retry-once quick_check (once
  per process per store file when the caller asks for ``check_once``).
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
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import structlog

from trw_memory._live_stores import connect_registered, note_store_sidecars

logger = structlog.get_logger(__name__)

# Trim the WAL back on reset so a stalled checkpoint cannot let the WAL grow unbounded
# (a large stale WAL widens the window for WAL-reset inconsistency). 64 MiB.
#
# Read this together with trw-mcp's ``wal_checkpoint_threshold_mb`` (default
# 10 MB), which is the size at which trw-mcp DECIDES a checkpoint is due. The
# two numbers are 6.4x apart and mean different things, and the gap is exactly
# what an operator sees on an engine below SQLite 3.51.3: PASSIVE is the only
# permitted mode there (``_wal_checkpoint.normalize_mode``), PASSIVE writes
# frames back but never truncates, so the file climbs toward THIS target and stays,
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
# risking runaway WAL growth (file is still trimmed toward journal_size_limit=64MiB on each reset).
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
    # WAL mode opens -wal/-shm on the first read; take one, then record their inodes
    # so an alias of either is refused even if the store's path is replaced later (C15).
    conn.execute("PRAGMA schema_version").fetchone()
    main = next((row[2] for row in conn.execute("PRAGMA database_list").fetchall() if row[1] == "main"), "")
    if main:
        note_store_sidecars(main)


def _file_backed(db_path: Path) -> bool:
    """Whether *db_path* names a real on-disk file (not ``:memory:``/``file::memory:...``)."""
    name = str(db_path)
    return name != ":memory:" and not name.startswith("file::memory:")


def connect(
    db_path: Path,
    *,
    dbapi: Any,
    timeout: float,
    check_same_thread: bool,
    cached_statements: int | None = None,
) -> Any:
    """Base sqlite connection with WAL/synchronous defaults.

    A file-backed open goes through :func:`connect_registered`, which also runs
    PRD-SEC-016's identity check: the store's inode is pinned before the
    driver's by-path open and compared after it, and a store swapped for a
    different file in between is refused (``StorageError``). A race inside
    SQLite's own C-level ``open()`` stays the residual PRD-SEC-016 records for
    G4 (same user or root, who can already open the store file directly).
    """
    kwargs: dict[str, object] = {
        "timeout": timeout,
        "check_same_thread": check_same_thread,
    }
    if cached_statements is not None:
        kwargs["cached_statements"] = cached_statements
    # Registered under the fd lock: nothing may open this inode by descriptor once a
    # connection can hold its locks (C15, see _live_stores).
    conn = (
        connect_registered(db_path, dbapi, str(db_path), **kwargs)
        if _file_backed(db_path)
        else dbapi.connect(str(db_path), **kwargs)
    )
    # Use the caller-provided ``dbapi`` for the Row factory so the type
    # matches the cursor. With the pysqlite3 shim live, ``dbapi`` is
    # usually pysqlite3 — but tests can pass stdlib ``sqlite3`` explicitly
    # to drive deterministic exception classes, and any cross-module row
    # factory would raise ``TypeError: Row() argument 1 must be
    # sqlite3.Cursor, not pysqlite3.dbapi2.Cursor`` (or vice versa).
    conn.row_factory = getattr(dbapi, "Row", sqlite3.Row)
    if (deadline := untrusted_deadline(db_path)) is not None:
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        # Before any statement runs. 3.11+: the import refuses Python 3.10 before any open.
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, UNTRUSTED_LENGTH_LIMIT)  # type: ignore[attr-defined]
    return conn


#: Store files a caller is reading that trw-memory did not write (an import's private copy), by real
#: path, each with its monotonic deadline. Every connection opened on one, from any thread -- the
#: schema check's, the backend's own opens (migrations, snapshot source), a stale-handle reopen, the
#: UTF-8 fallback's second connection -- runs no statement past that deadline (``OperationalError:
#: interrupted``) and builds or reads no value past ``UNTRUSTED_LENGTH_LIMIT``. Keyed by file, not by
#: context, so no connection opened on the copy escapes it (rc8 C12). The handler stays until close.
_UNTRUSTED: dict[str, list[float]] = {}
_UNTRUSTED_LOCK = threading.Lock()


@contextlib.contextmanager
def untrusted_store(db_path: Path | str, deadline: float) -> Iterator[None]:
    """Treat *db_path* as a store trw-memory did not write until the block ends (see ``_UNTRUSTED``)."""
    key = os.path.realpath(db_path)
    with _UNTRUSTED_LOCK:
        _UNTRUSTED.setdefault(key, []).append(deadline)
    try:
        yield
    finally:
        with _UNTRUSTED_LOCK:  # another registration of the same file keeps it registered
            if (deadlines := _UNTRUSTED.get(key)) is not None:
                deadlines.remove(deadline)
                if not deadlines:
                    del _UNTRUSTED[key]


def untrusted_deadline(db_path: Path | str) -> float | None:
    """The earliest deadline under which *db_path* is open as an untrusted store, or ``None``."""
    if not _UNTRUSTED or not _file_backed(Path(db_path)):
        return None
    with _UNTRUSTED_LOCK:
        deadlines = _UNTRUSTED.get(os.path.realpath(db_path))
        return min(deadlines) if deadlines else None


#: The largest string, BLOB or row such a connection may build or read (``SQLITE_LIMIT_LENGTH``), so
#: no expression in a store trw-memory did not write can allocate past it (rc8 C12). The largest
#: value trw-memory writes is a ``vec_memories`` chunk: 1024 vectors of ``dim`` float32s, 1.5 MiB at
#: the default 384 dimensions and 12 MiB at 3072; rows of text are far smaller.
UNTRUSTED_LENGTH_LIMIT = 16 * 1024 * 1024


#: Store files whose ``quick_check`` passed in this process: (realpath, st_dev, st_ino).
_VERIFIED_STORES: set[tuple[str, int, int]] = set()
_VERIFIED_LOCK = threading.Lock()


def _store_identity(db_path: Path) -> tuple[str, int, int] | None:
    """The file's identity, or ``None`` when it cannot be stat'ed (checked on every open)."""
    try:
        stat = os.stat(db_path)
    except (OSError, ValueError):  # trw-fail-silent-allow: no identity means the check runs every time
        return None
    return (os.path.realpath(db_path), stat.st_dev, stat.st_ino)


def mark_verified(db_path: Path, verified: bool = True) -> None:
    """Record (or retire) that *db_path*'s ``quick_check`` passed here, so a ``check_once`` open skips it."""
    identity = _store_identity(db_path)
    if identity is not None:
        with _VERIFIED_LOCK:
            (_VERIFIED_STORES.add if verified else _VERIFIED_STORES.discard)(identity)


def open_and_configure(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    check_once: bool = False,
) -> Any:
    """Open a connection with WAL mode and run a quick integrity check.

    ``PRAGMA quick_check`` reads the whole file: about 320 ms per open at 20,000
    rows. Every open runs it by default, so a writer never opens a store that
    has gone bad since. ``check_once`` is for the daemon's recall path
    (PRD-CORE-298 FR05), which opened the store twice per call. There the check
    runs once per process per store file -- realpath, device and inode, so a
    replaced file is checked again. A failed check is never recorded, so a
    corrupt store fails on every open. The trade-off, in one sentence: a store
    corrupted in place after this process verified it is not caught by a
    ``check_once`` open, including the recall's access-count update, until the
    next default open, the stale-handle probe, or the opt-in background checker.

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
    )
    try:
        apply_open_pragmas(conn, verify=True)
        identity = _store_identity(db_path)
        if check_once and identity is not None and identity in _VERIFIED_STORES:
            return conn

        for attempt in range(2):
            rows = conn.execute("PRAGMA quick_check").fetchall()
            if len(rows) == 1 and rows[0][0] == "ok":
                if identity is not None:
                    with _VERIFIED_LOCK:
                        _VERIFIED_STORES.add(identity)
                return conn
            if attempt == 0:
                logger.warning(
                    "integrity_check_retry",
                    db=str(db_path),
                    detail=rows[0][0] if rows else "empty",
                )
                time.sleep(1.0)
        raise IntegrityCheckFailed("database disk image is malformed (quick_check failed twice)")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.close()
        raise


def open_without_integrity_check(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
) -> Any:
    """Open a connection skipping integrity check for explicit lock/busy contention only."""
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
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
) -> Any:
    """Open a short-lived probe connection with the standard PRAGMA profile.

    The shared open path for :func:`check_integrity` and :func:`db_has_data`.
    Both used to call :func:`connect` directly and skip
    :func:`apply_open_pragmas`, so a probe connection ran without
    ``journal_size_limit``: it could append WAL frames (a probe still runs the
    recovery/checkpoint machinery of the engine) with the 64 MiB truncation target that every
    other open path sets left unset. A per-callsite PRAGMA list is exactly how
    that cap gets opted out of by accident, so there is one open path instead.
    """
    conn = connect(
        db_path,
        dbapi=dbapi,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check_same_thread=True,
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
) -> dict[str, object]:
    """Check database integrity without opening a full backend.

    Re-exported as ``SQLiteBackend.check_integrity`` for back-compat.

    Returns:
        Dict with ``ok`` (bool), ``detail`` (str), and ``db_path``.
    """
    conn: Any = None
    try:
        conn = open_probe(db_path, dbapi=dbapi)
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


#: SQLite primary result codes; every extended code keeps its primary one in the low byte.
_SQLITE_IOERR, _SQLITE_CORRUPT, _SQLITE_NOTADB = 10, 11, 26
#: How a damaged file reads when the error carries no result code (Python 3.10, or raised here).
_CORRUPTION_MESSAGES = ("database disk image is malformed", "file is not a database")


class IntegrityCheckFailed(sqlite3.DatabaseError):
    """``PRAGMA quick_check`` reported damage on both attempts of an open."""


def classify_open_error(exc: BaseException) -> Literal["io", "lock", "corrupt", "other"]:
    """What a failed open says about the store. Only ``"corrupt"`` is grounds to quarantine it.

    Lives here, at the lowest layer that needs it, so ``db_has_data`` and
    ``open_connection_with_recovery`` cannot drift apart on what "transient"
    means -- they make the same destructive decision together.

    - ``"io"``: ``SQLITE_IOERR`` and its extended codes, a read or write failed.
      It says nothing about what the file holds. It is how SQLite below 3.51.3
      reports the WAL-reset race when two processes write one WAL store (L-8QV8):
      ``quick_check`` raised ``vtable constructor failed: memories_fts`` on a
      healthy store, and recovery then quarantined it. Needs ``sqlite_errorcode``
      (Python 3.11+); on 3.10 no error reads as I/O.
    - ``"lock"``: lock/busy, another connection holds the store.
    - ``"corrupt"``: the file's content is damaged -- ``SQLITE_CORRUPT`` (and its
      ``CORRUPT_*`` codes), ``SQLITE_NOTADB``, or a failed ``quick_check``.
    - ``"other"``: out of descriptors, disk full, read-only, ``locking protocol``,
      a schema change: the machine or another connection failed, not the file.
      Quarantining on those moved healthy stores aside under connections that
      kept committing into the moved file.
    """
    code = getattr(exc, "sqlite_errorcode", None)
    primary = code & 0xFF if isinstance(code, int) else None
    message = str(exc).lower()
    # A result code outranks the words: a CORRUPT error may quote "database is locked".
    if primary == _SQLITE_IOERR:
        return "io"
    if isinstance(exc, IntegrityCheckFailed) or primary in (_SQLITE_CORRUPT, _SQLITE_NOTADB):
        return "corrupt"
    if "locked" in message or "busy" in message:
        return "lock"
    if primary is None and message.startswith(_CORRUPTION_MESSAGES):
        return "corrupt"
    return "other"


def db_has_data(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
) -> bool | None:
    """Rows in ``memories``? ``True``/``False``, or ``None`` when UNKNOWN.

    Non-destructive: this proves rows are readable; it does not prove the
    database is structurally healthy after a failed quick_check.

    ``None`` is the load-bearing third answer and it exists because of one
    caller. ``open_connection_with_recovery`` asks this on the lock-contention
    branch, and a busy database makes the PROBE fail too — so a blanket ``False``
    told that caller "this store has no rows", and it took the destructive branch:
    rename the live database to ``.corrupt.bak`` and initialise a blank schema. A
    populated store was wiped because the machine was busy, which is exactly when
    it is most likely to happen.

    A structural failure still returns ``False``: a garbage file genuinely has no
    readable rows and the recovery path is designed for it. Only lock/busy — a
    transient condition that says nothing about content — is ``None``.

    Found by a cross-family audit 2026-09-12. Since 2026-09-24 that branch no
    longer asks at all: an EMPTY store under contention is a new one being
    created, and recovering it lost the writes of the connections creating it.
    Lock/busy never reaches recovery; this probe now only feeds the corruption
    log line.
    """
    conn: Any = None
    try:
        conn = open_probe(db_path, dbapi=dbapi)
        count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        return bool(count > 0)
    except sqlite3.Error as exc:
        if classify_open_error(exc) == "lock":
            logger.warning("db_has_data_probe_locked", db=str(db_path), error=str(exc))
            return None
        # trw-fail-silent-allow: a structurally unreadable file has no readable rows (see docstring)
        return False
    finally:
        # Close in finally so an unexpected (non-sqlite) exception cannot leak it.
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
