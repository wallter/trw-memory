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
from trw_memory.storage._connection import classify_open_error
from trw_memory.storage._recovery import classify_recovery_preflight, write_recovery_state
from trw_memory.storage._schema import ensure_schema, ensure_vec_table

try:
    import sqlite_vec
except ImportError:  # pragma: no cover — optional dep
    sqlite_vec = None

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


#: Opens tried when the checked open fails with an I/O error or lock/busy, and the
#: wait before each retry (x attempt). Lock/busy waits are short: the case they
#: exist for is two first opens of a NEW store both switching it to WAL, where
#: SQLite answers "database is locked" at once instead of calling the busy
#: handler (it would deadlock), and the other opener is done within milliseconds.
_IO_ERROR_ATTEMPTS = 3
_IO_ERROR_BACKOFF_SECONDS = 0.5
_LOCK_BACKOFF_SECONDS = 0.05


def _open_checked(backend: SQLiteBackend, db_path: Path) -> Any:
    """The integrity-checked open, retried while it fails with an I/O error or lock/busy."""
    once = getattr(backend, "_check_integrity_once", False)
    for attempt in range(1, _IO_ERROR_ATTEMPTS + 1):
        try:
            return backend._open_and_configure(db_path, check_once=once)
        except sqlite3.DatabaseError as exc:
            kind = classify_open_error(exc)
            if kind not in ("io", "lock") or attempt == _IO_ERROR_ATTEMPTS:
                raise
            locked = kind == "lock"
            logger.warning(
                "db_open_lock_retry" if locked else "db_open_io_error_retry",
                db=str(db_path),
                attempt=attempt,
                error=str(exc),
            )
            time.sleep((_LOCK_BACKOFF_SECONDS if locked else _IO_ERROR_BACKOFF_SECONDS) * attempt)
    raise AssertionError("unreachable: the last attempt returns or raises")  # pragma: no cover


def open_connection_with_recovery(
    backend: SQLiteBackend,
    db_path: Path,
    *,
    dbapi: Any,
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
        conn = _open_checked(backend, db_path)
    except sqlite3.DatabaseError as exc:
        kind = classify_open_error(exc)
        # An I/O error is never the recovery branch, whatever the probe says:
        # renaming a store because a read failed is how a healthy one was lost
        # (L-8QV8). Nor is lock/busy, whatever the store holds. It used to
        # recover when the data probe said "no rows" -- and a NEW store has no
        # rows while other connections are creating it. Recovery then renamed
        # the file to ``.corrupt.bak`` and unlinked its WAL under those
        # connections, whose next commits returned success into files no reader
        # would open again: the daemon's silent write loss. A store that really
        # is damaged fails its next open with a corruption error and is
        # recovered then. Both open unchecked, the file untouched.
        if kind in ("io", "lock"):
            logger.warning(
                "db_integrity_check_deferred",
                db=str(db_path),
                cause=kind,
                action="open_anyway",
                error=str(exc),
            )
            write_recovery_state(
                db_path,
                status="degraded_open_with_background_recovery",
                reason="sqlite_io_error" if kind == "io" else "sqlite_lock_or_busy",
                db_size_bytes=preflight.db_size_bytes,
            )
            conn = backend._open_without_integrity_check(db_path, dbapi=dbapi)
            integrity_warning = True
        elif kind == "other":
            # Out of descriptors, disk full, read-only, ``locking protocol``: the
            # machine or another connection failed, not the file. Quarantining on
            # those moved healthy stores aside the same way. Surface it; the file stays.
            logger.warning("db_open_failed_not_corruption", db=str(db_path), error=str(exc))
            raise
        else:
            has_data = backend._db_has_data(db_path, dbapi=dbapi)
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
