"""Stale-handle detection + reconnect + integrity-check helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — class methods become 1-line delegators.

4 helpers covering the connection-resilience boundary:

- ``handle_integrity_regression`` — IntegrityScheduler callback;
  flips ``integrity_warning`` on the backend.
- ``reconnect`` — validate a replacement connection, then atomically swap it;
  ``backend._conn`` and increments ``backend.reconnect_count``.
- ``ensure_connection_fresh`` — best-effort stale probe; calls
  ``reconnect`` when ``backend._stale_detector.is_stale()``.
- ``run_integrity_check`` — PRAGMA quick_check probe.

Extracted as PRD-DIST-245 Phase 1 batch 89.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StaleConnectionError
from trw_memory.storage._init_helpers import load_vec_extension
from trw_memory.storage._permissions import harden_db_file_mode, prepare_db_file_mode
from trw_memory.storage._schema import ensure_fts_table, ensure_schema

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
    with backend._lock:
        old_conn = backend._conn
        if backend._skip_commit_depth != 0 or old_conn.in_transaction:
            raise StaleConnectionError(
                "Refusing to reconnect while a transaction is active",
                path=str(backend._db_path),
            )
        candidate = None
        try:
            prepare_db_file_mode(backend._db_path)
            if backend._sqlcipher_key_hex is not None:
                candidate = backend._open_and_configure(
                    backend._db_path,
                    dbapi=backend._dbapi,
                    sqlcipher_key_hex=backend._sqlcipher_key_hex,
                )
            else:
                candidate = backend._open_and_configure(backend._db_path)
            ensure_schema(candidate)
            vec_available = load_vec_extension(candidate, backend._db_path, backend._dim)
            fts_available = ensure_fts_table(candidate)
            harden_db_file_mode(backend._db_path)
            backend._stale_detector.reset()
        except Exception as exc:
            if candidate is not None:
                with contextlib.suppress(Exception):
                    candidate.close()
            raise StaleConnectionError(
                f"Failed to reopen stale connection to {backend._db_path}: {exc}",
                path=str(backend._db_path),
            ) from exc
        backend._conn = candidate
        backend._vec_available = vec_available
        backend._fts_available = fts_available
        backend.reconnect_count += 1
        with contextlib.suppress(Exception):
            old_conn.close()
        logger.info(
            "memory_stale_handle_reconnected",
            db_path=str(backend._db_path),
            reconnect_count=backend.reconnect_count,
        )


def ensure_connection_fresh(backend: SQLiteBackend) -> None:
    """Best-effort stale-handle probe; reconnect if stale.

    Steady-state cost is effectively zero — the check is cached for
    ``TRW_MEMORY_STALE_HANDLE_CHECK_SECS`` (default 1s) so a warm
    kernel page cache amortises the syscall.

    Raises:
        StaleConnectionError: If a stale handle is detected and
            reconnect fails.
    """
    with backend._lock:
        if backend._stale_detector.is_stale():
            reconnect(backend)


@contextmanager
def fresh_connection(backend: SQLiteBackend) -> Iterator[None]:
    """Serialize one operation with stale detection and connection replacement."""
    with backend._lock:
        if backend._skip_commit_depth == 0:
            ensure_connection_fresh(backend)
        yield


def run_integrity_check(backend: SQLiteBackend) -> bool:
    """Run PRAGMA quick_check; return True when the DB is healthy."""
    try:
        rows = backend._conn.execute("PRAGMA quick_check").fetchall()
        return len(rows) == 1 and rows[0][0] == "ok"
    except sqlite3.DatabaseError:
        return False
