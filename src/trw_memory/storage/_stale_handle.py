"""Stale-handle detection + reconnect + integrity-check helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — class methods become 1-line delegators.

4 helpers covering the connection-resilience boundary:

- ``handle_integrity_regression`` — IntegrityScheduler callback;
  flips ``integrity_warning`` on the backend.
- ``reconnect`` — close + reopen + ensure_schema; mutates
  ``backend._conn`` and increments ``backend.reconnect_count``.
- ``ensure_connection_fresh`` — best-effort stale probe; calls
  ``reconnect`` when ``backend._stale_detector.is_stale()``.
- ``run_integrity_check`` — PRAGMA quick_check probe.

Extracted as PRD-DIST-245 Phase 1 batch 89.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StaleConnectionError
from trw_memory.storage._schema import ensure_schema

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


def handle_integrity_regression(backend: SQLiteBackend) -> None:
    """IntegrityScheduler callback: flip ``integrity_warning`` flag.

    Same flag used by the transient-WAL-contention path so external code
    can observe both regressions uniformly.
    """
    backend.integrity_warning = True


def reconnect(backend: SQLiteBackend) -> None:
    """Close the current connection and reopen against the current DB file.

    Called when ``ensure_connection_fresh`` detects a stale handle.

    Raises:
        StaleConnectionError: If reopening the connection fails.
    """
    try:
        with contextlib.suppress(sqlite3.Error):
            backend._conn.close()
        if backend._sqlcipher_key_hex is not None:
            backend._conn = backend._open_and_configure(
                backend._db_path,
                dbapi=backend._dbapi,
                sqlcipher_key_hex=backend._sqlcipher_key_hex,
            )
        else:
            backend._conn = backend._open_and_configure(backend._db_path)
        ensure_schema(backend._conn)
        backend._stale_detector.reset()
        backend.reconnect_count += 1
        logger.info(
            "memory_stale_handle_reconnected",
            db_path=str(backend._db_path),
            reconnect_count=backend.reconnect_count,
        )
    except Exception as exc:
        raise StaleConnectionError(
            f"Failed to reopen stale connection to {backend._db_path}: {exc}",
            path=str(backend._db_path),
        ) from exc


def ensure_connection_fresh(backend: SQLiteBackend) -> None:
    """Best-effort stale-handle probe; reconnect if stale.

    Steady-state cost is effectively zero — the check is cached for
    ``TRW_MEMORY_STALE_HANDLE_CHECK_SECS`` (default 1s) so a warm
    kernel page cache amortises the syscall.

    Raises:
        StaleConnectionError: If a stale handle is detected and
            reconnect fails.
    """
    if backend._stale_detector.is_stale():
        reconnect(backend)


def run_integrity_check(backend: SQLiteBackend) -> bool:
    """Run PRAGMA quick_check; return True when the DB is healthy."""
    try:
        rows = backend._conn.execute("PRAGMA quick_check").fetchall()
        return len(rows) == 1 and rows[0][0] == "ok"
    except sqlite3.DatabaseError:
        return False
