"""SQLiteBackend.__init__ helper steps.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``__init__`` becomes a sequence of 4 helper calls
that mutate the backend instance.

4 helpers covering the side-init concerns:

- ``open_connection_with_recovery`` — open + WAL + auto-recovery on
  quick_check corruption failures; explicit lock/busy failures may still
  fall back to opening without a quick_check after a data-presence probe,
  and an I/O error is retried, then opened the same way, never recovered.
  Returns ``(conn, integrity_warning, recovered)``.
- ``load_vec_extension`` — load sqlite-vec when available; populate
  vec_index/vec_memories tables; flip ``_vec_available``.
- ``start_integrity_scheduler`` — PRD-INFRA-063 periodic quick_check
  scheduler (fail-open).

Extracted as PRD-DIST-245 Phase 1 batch 88.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._connection import is_io_error, is_lock_contention_error
from trw_memory.storage._recovery import classify_recovery_preflight, write_recovery_state
from trw_memory.storage._schema import ensure_schema, ensure_vec_table

try:
    import sqlite_vec
except ImportError:  # pragma: no cover — optional dep
    sqlite_vec = None

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


#: Re-exported from ``_connection`` so this module and ``db_has_data`` cannot
#: drift on what counts as transient: they make the same destructive decision
#: together, and the whole defect was the two halves disagreeing.
_is_lock_contention_error = is_lock_contention_error

#: Opens tried when the checked open fails with an I/O error, and the wait before each retry (x attempt).
_IO_ERROR_ATTEMPTS = 3
_IO_ERROR_BACKOFF_SECONDS = 0.5


def _open_checked(backend: SQLiteBackend, db_path: Path, *, dbapi: Any, sqlcipher_key_hex: str | None) -> Any:
    """The integrity-checked open, retried while it fails with an I/O error."""
    once = getattr(backend, "_check_integrity_once", False)
    for attempt in range(1, _IO_ERROR_ATTEMPTS + 1):
        try:
            if sqlcipher_key_hex is None:
                return backend._open_and_configure(db_path, check_once=once)
            return backend._open_and_configure(
                db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex, check_once=once
            )
        except sqlite3.DatabaseError as exc:
            if not is_io_error(exc) or attempt == _IO_ERROR_ATTEMPTS:
                raise
            logger.warning("db_open_io_error_retry", db=str(db_path), attempt=attempt, error=str(exc))
            time.sleep(_IO_ERROR_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable: the last attempt returns or raises")  # pragma: no cover


def open_connection_with_recovery(
    backend: SQLiteBackend,
    db_path: Path,
    *,
    dbapi: Any,
    sqlcipher_key_hex: str | None,
    recovery_policy: str,
    corrupt_backup_keep: int,
    rebuild_from_cold: bool,
    recovery_inline_max_bytes: int = 64 * 1024 * 1024,
) -> tuple[Any, bool, bool]:
    """Open connection with WAL + auto-recovery on quick_check failure.

    Returns ``(conn, integrity_warning, recovered)``. A failed ``PRAGMA
    quick_check`` after retry is treated as corruption even when rows remain
    readable; a row-count probe proves data exists, not that the B-tree is
    healthy. Explicit lock/busy failures keep the non-destructive fallback
    path, and so does an I/O error that outlasts its retries: it is never
    corruption, so the store is opened unchecked and never quarantined.
    """
    integrity_warning = False
    recovered = False
    preflight = classify_recovery_preflight(db_path, inline_max_bytes=recovery_inline_max_bytes)
    backend.recovery_preflight = preflight
    if preflight.classification == "hard_fail" and recovery_policy == "strict":
        raise CorruptDatabaseUnsalvageableError(
            f"memory recovery previously hard-failed for {db_path}",
            backup_path=preflight.state_path,
        )
    try:
        conn = _open_checked(backend, db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
    except sqlite3.DatabaseError as exc:
        if is_io_error(exc):
            # Never the recovery branch, whatever the probe says: renaming a store
            # because a read failed is how a healthy one was lost (L-8QV8).
            logger.warning(
                "db_integrity_check_io_error",
                db=str(db_path),
                action="open_anyway",
                error=str(exc),
                hint="quick_check hit an I/O error on every attempt; opening without it, the file untouched",
            )
            write_recovery_state(
                db_path,
                status="degraded_open_with_background_recovery",
                reason="sqlite_io_error",
                db_size_bytes=preflight.db_size_bytes,
            )
            conn = backend._open_without_integrity_check(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
            ensure_schema(conn)
            return conn, True, False
        # `is not False`, deliberately, not a truth test. Under contention the
        # PROBE is locked too and returns None (UNKNOWN), and the safe reading of
        # "I could not check" is "assume there is something to lose" — the
        # degraded open below is non-destructive, while the `else` branch renames
        # the live database and initialises a blank schema. A plain truth test
        # sent UNKNOWN down the destructive path, so a populated store was wiped
        # precisely when the machine was busy.
        if (
            _is_lock_contention_error(exc)
            and backend._db_has_data(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex) is not False
        ):
            logger.warning(
                "db_integrity_check_deferred_due_to_lock",
                db=str(db_path),
                action="open_anyway",
                hint=("quick_check could not complete because SQLite reported lock/busy; opening without probe"),
            )
            write_recovery_state(
                db_path,
                status="degraded_open_with_background_recovery",
                reason="sqlite_lock_or_busy",
                db_size_bytes=preflight.db_size_bytes,
            )
            conn = backend._open_without_integrity_check(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
            integrity_warning = True
        else:
            has_data = backend._db_has_data(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
            logger.exception(
                "db_corrupt_detected",
                db=str(db_path),
                action="auto_recover",
                has_data=has_data,
                reason=str(exc),
            )
            if preflight.classification == "degraded_open_with_background_recovery":
                write_recovery_state(
                    db_path,
                    status="degraded_open_with_background_recovery",
                    reason=preflight.reason,
                    db_size_bytes=preflight.db_size_bytes,
                )
                raise CorruptDatabaseUnsalvageableError(
                    "database recovery requires background recovery outside startup budget",
                    backup_path=preflight.state_path,
                ) from exc
            try:
                write_recovery_state(
                    db_path,
                    status="running",
                    reason="inline_recovery_started",
                    db_size_bytes=preflight.db_size_bytes,
                )
                conn = backend.recover_db(
                    db_path,
                    dbapi=dbapi,
                    sqlcipher_key_hex=sqlcipher_key_hex,
                    recovery_policy=recovery_policy,  # type: ignore[arg-type]
                    corrupt_backup_keep=corrupt_backup_keep,
                    rebuild_from_cold=rebuild_from_cold,
                )
                write_recovery_state(
                    db_path,
                    status="recovered",
                    reason="inline_recovery_succeeded",
                    db_size_bytes=preflight.db_size_bytes,
                )
                recovered = True
            except CorruptDatabaseUnsalvageableError:
                write_recovery_state(
                    db_path,
                    status="hard_fail",
                    reason="inline_recovery_failed",
                    db_size_bytes=preflight.db_size_bytes,
                )
                raise
    ensure_schema(conn)
    return conn, integrity_warning, recovered


def load_vec_extension(conn: Any, db_path: Path, dim: int) -> bool:
    """Load sqlite-vec extension and ensure vec tables; return availability flag.

    Fail-open: AttributeError surfaces on Python builds without
    SQLITE_ENABLE_LOAD_EXTENSION (common on macOS system Python). Returns
    False so the caller flips ``_vec_available=False`` and BM25 keeps working.
    """
    if sqlite_vec is None:
        logger.debug("sqlite_vec_unavailable", reason="not_installed")
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        ensure_vec_table(conn, dim)
        logger.debug("sqlite_vec_loaded", db=str(db_path))
        return True
    except (sqlite3.Error, OSError, AttributeError) as exc:
        logger.warning(
            "sqlite_vec_load_failed",
            db=str(db_path),
            reason=type(exc).__name__,
            detail=str(exc),
            hint=("Python lacks SQLite load_extension support; vector search disabled, BM25 still works"),
        )
        return False


def start_integrity_scheduler(
    db_path: Path,
    *,
    interval_minutes: int,
    on_regression: Any,
) -> Any:
    """PRD-INFRA-063 periodic integrity scheduler (fail-open observability)."""
    try:
        from trw_memory.storage._integrity_scheduler import IntegrityScheduler

        scheduler = IntegrityScheduler(
            db_path,
            interval_minutes=interval_minutes,
            on_regression=on_regression,
        )
        scheduler.start()
        return scheduler
    except Exception:  # justified: observability scheduler must not block open
        logger.debug("integrity_scheduler_unavailable", db=str(db_path), exc_info=True)
        return None
