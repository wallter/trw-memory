"""Stale connection detector for SQLiteBackend.

Detects when a DB file was replaced beneath an open connection (e.g., after
``recover_db`` moved the corrupt file aside) and signals the backend to
reconnect so it reads the fresh inode rather than the corrupt one.

Cross-process signal:
    ``recover_db`` writes ``memory.db.recovered_at`` next to the DB file.
    Consumers compare the sentinel's mtime against ``_opened_at_mtime``; if
    the sentinel is newer, the connection is stale.

Belt-and-suspenders:
    On every precheck we also compare the DB file's inode against
    ``_opened_inode`` captured at open time.

Cost control:
    The check is cached for ``check_interval_secs`` (default 1.0, tunable
    via the ``TRW_MEMORY_STALE_HANDLE_CHECK_SECS`` env var).  Back-to-back
    calls within the budget skip the stat calls entirely, keeping overhead
    well below 100µs on a warm kernel page cache.

Guarantee:
    This is a *best-effort* probe, NOT a consistency guarantee.  Writes to
    the new inode that arrived between the last precheck and now will still be
    missed until the next precheck window opens.  Recovery events are rare
    (once per corruption incident) and the budget is 1 s, so the practical
    miss window is small.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default staleness budget in seconds — overridable via env.
_DEFAULT_CHECK_SECS = float(os.environ.get("TRW_MEMORY_STALE_HANDLE_CHECK_SECS", "1.0"))

# Name of the sentinel file written by recover_db.
SENTINEL_NAME = "memory.db.recovered_at"


def sentinel_path(db_path: Path) -> Path:
    """Return the sentinel path for *db_path*."""
    return db_path.parent / SENTINEL_NAME


def write_sentinel(db_path: Path, backup_path: Path) -> None:
    """Write the cross-process recovery sentinel next to *db_path*.

    Called by :meth:`SQLiteBackend.recover_db` immediately after the corrupt
    file has been moved aside and the fresh DB is in place.

    Args:
        db_path: Path of the (now fresh) database file.
        backup_path: Path the corrupt file was moved to (for the log breadcrumb).
    """
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    sentinel = sentinel_path(db_path)
    try:
        sentinel.write_text(f"{ts}\n{backup_path}\n", encoding="utf-8")
        logger.info(
            "recovery_sentinel_written",
            sentinel=str(sentinel),
            ts=ts,
            backup_path=str(backup_path),
        )
    except OSError:
        logger.warning(
            "recovery_sentinel_write_failed",
            sentinel=str(sentinel),
            exc_info=True,
        )


class StaleHandleDetector:
    """Cheap, cached stale-connection detector.

    Instantiate once per :class:`SQLiteBackend` at open time.  Call
    :meth:`is_stale` before each public read method; if it returns ``True``,
    the backend should reconnect and then call :meth:`reset` with the new
    open-time stats.

    Args:
        db_path: Path to the SQLite DB file (used for inode + sentinel checks).
        check_interval_secs: How long to cache a "fresh" result.  Defaults to
            ``TRW_MEMORY_STALE_HANDLE_CHECK_SECS`` env var (default 1.0).
    """

    def __init__(
        self,
        db_path: Path,
        check_interval_secs: float | None = None,
    ) -> None:
        self._db_path = db_path
        self._sentinel_path = sentinel_path(db_path)
        self._check_interval = check_interval_secs if check_interval_secs is not None else _DEFAULT_CHECK_SECS

        # Capture open-time stats (both signals).
        self._opened_at_mtime: float = time.monotonic()
        self._opened_inode: int | None = None
        self._sentinel_mtime_at_open: float | None = None

        self._capture_open_stats()

        # Cache: last time we performed a real stat.
        self._last_checked: float = 0.0

    def _capture_open_stats(self) -> None:
        """Record inode + sentinel mtime at (re-)open time."""
        try:
            st = self._db_path.stat()
            self._opened_inode = st.st_ino
            self._opened_at_mtime = st.st_mtime
        except OSError:
            self._opened_inode = None
            self._opened_at_mtime = 0.0

        try:
            self._sentinel_mtime_at_open = self._sentinel_path.stat().st_mtime
        except OSError:
            self._sentinel_mtime_at_open = None

    def reset(self) -> None:
        """Re-capture stats after a successful reconnect."""
        self._capture_open_stats()
        self._last_checked = time.monotonic()

    def is_stale(self) -> bool:
        """Return True if the connection appears stale.

        Checks are cached for ``check_interval_secs``; returns False
        immediately on a warm cache hit without any syscalls.
        """
        now = time.monotonic()
        if now - self._last_checked < self._check_interval:
            return False

        # Belt 1: sentinel mtime
        try:
            sentinel_mtime = self._sentinel_path.stat().st_mtime
            if self._sentinel_mtime_at_open is None or sentinel_mtime > self._sentinel_mtime_at_open:
                logger.warning(
                    "memory_stale_handle_detected",
                    action="memory_stale_handle_detected",
                    signal="sentinel_mtime",
                    sentinel_mtime=sentinel_mtime,
                    opened_sentinel_mtime=self._sentinel_mtime_at_open,
                    db_path=str(self._db_path),
                )
                return True
        except OSError:
            pass  # Sentinel absent is fine — no recovery has happened

        # Belt 2: inode change
        if self._opened_inode is not None:
            try:
                current_inode = self._db_path.stat().st_ino
                if current_inode != self._opened_inode:
                    logger.warning(
                        "memory_stale_handle_detected",
                        action="memory_stale_handle_detected",
                        signal="inode_change",
                        old_inode=self._opened_inode,
                        new_inode=current_inode,
                        db_path=str(self._db_path),
                    )
                    return True
            except OSError:
                pass  # DB temporarily absent — report as stale so caller reconnects

        self._last_checked = now
        return False
