"""SQLiteBackend.__init__ helper steps.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``__init__`` becomes a sequence of 4 helper calls
that mutate the backend instance.

4 helpers covering the side-init concerns:

- ``open_connection_with_recovery`` — open + WAL + auto-recovery on
  quick_check failure (transient-WAL-contention guard via db_has_data).
  Returns ``(conn, integrity_warning, recovered)``.
- ``load_vec_extension`` — load sqlite-vec when available; populate
  vec_index/vec_memories tables; flip ``_vec_available``.
- ``register_writer_registry`` — PRD-INFRA-064 advisory writer
  registry (fail-open: never blocks open).
- ``start_integrity_scheduler`` — PRD-INFRA-063 periodic quick_check
  scheduler (fail-open).

Extracted as PRD-DIST-245 Phase 1 batch 88.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.storage._schema import ensure_schema, ensure_vec_table

try:
    import sqlite_vec
except ImportError:  # pragma: no cover — optional dep
    sqlite_vec = None

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


def open_connection_with_recovery(
    backend: SQLiteBackend,
    db_path: Path,
    *,
    dbapi: Any,
    sqlcipher_key_hex: str | None,
    recovery_policy: str,
    corrupt_backup_keep: int,
    rebuild_from_cold: bool,
) -> tuple[Any, bool, bool]:
    """Open connection with WAL + auto-recovery on quick_check failure.

    Returns ``(conn, integrity_warning, recovered)``. The recovery path
    is gated by ``db_has_data`` so transient WAL contention doesn't
    destroy a DB with real rows.
    """
    integrity_warning = False
    recovered = False
    try:
        if sqlcipher_key_hex is None:
            conn = backend._open_and_configure(db_path)
        else:
            conn = backend._open_and_configure(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
    except sqlite3.DatabaseError:
        if backend._db_has_data(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex):
            logger.warning(
                "db_integrity_check_failed_but_has_data",
                db=str(db_path),
                action="open_anyway",
                hint=("quick_check failed but DB has rows — likely transient WAL contention, not corruption"),
            )
            conn = backend._open_without_integrity_check(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
            integrity_warning = True
        else:
            logger.exception("db_corrupt_detected", db=str(db_path), action="auto_recover")
            conn = backend.recover_db(
                db_path,
                dbapi=dbapi,
                sqlcipher_key_hex=sqlcipher_key_hex,
                recovery_policy=recovery_policy,  # type: ignore[arg-type]
                corrupt_backup_keep=corrupt_backup_keep,
                rebuild_from_cold=rebuild_from_cold,
            )
            recovered = True
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


def register_writer_registry(db_path: Path, warn_threshold: int) -> Any:
    """PRD-INFRA-064 advisory writer registry (fail-open)."""
    try:
        from trw_memory.storage._writer_registry import WriterRegistry

        registry = WriterRegistry(db_path, warn_threshold=warn_threshold)
        registry.register()
        return registry
    except Exception:  # justified: advisory-only invariant — never block open
        logger.debug("writer_registry_unavailable", db=str(db_path), exc_info=True)
        return None


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
