"""Periodic integrity check scheduler (PRD-INFRA-063 / B2).

Opt-in daemon thread running ``PRAGMA quick_check`` on a dedicated read-only
connection at a configurable interval. Observability ONLY — a failed check
sets ``integrity_warning=True`` on the owning backend and logs
``db_integrity_regression_detected``; it NEVER triggers auto-recovery.

Key invariants (do not relax without updating PRD-INFRA-063):

- Default interval is 0 (DISABLED). The scheduler only starts when the
  config knob ``memory_integrity_check_interval_minutes`` is > 0. This is
  the regression guard for the sprint exit criterion ``opt-in-defaults``.
- Uses a DEDICATED read-only connection, NEVER the backend's primary write
  connection. Sharing the connection would serialize writes behind the
  quick_check (which can take seconds on large DBs) and is explicitly
  rejected by the mitigation plan section B2.
- Thread is a ``daemon`` thread — interpreter exit does not block on it.
- Thread is stopped cleanly on :meth:`IntegrityScheduler.stop` via an
  :class:`threading.Event`; the loop wakes from its wait() as soon as the
  event is set rather than sleeping the full interval.
- Failures to open the read-only connection or run the probe are logged at
  DEBUG/WARNING and the loop continues — one flaky check does not stop the
  scheduler.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

__all__ = ["IntegrityScheduler"]

logger = structlog.get_logger(__name__)


class IntegrityScheduler:
    """Background PRAGMA quick_check runner on an isolated read-only connection.

    Args:
        db_path: Path to the SQLite database file.
        interval_minutes: Interval between checks, in minutes. ``0`` disables
            the scheduler entirely — :meth:`start` becomes a no-op.
        on_regression: Optional callback invoked with the :class:`~pathlib.Path`
            and the ``quick_check`` detail string when a regression is
            detected. Typically wired to set ``integrity_warning=True`` on the
            owning :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend`.

    Attributes:
        last_check_at: Timestamp of the last check, in epoch seconds, or
            ``None`` before the first check completes. Read by C3's
            session-start health dashboard.
        last_check_ok: Last observed result (True/False), or ``None`` before
            the first check.
    """

    __slots__ = (
        "_db_path",
        "_interval_seconds",
        "_on_regression",
        "_stop_event",
        "_thread",
        "_lock",
        "last_check_at",
        "last_check_ok",
    )

    def __init__(
        self,
        db_path: Path,
        interval_minutes: int,
        on_regression: "callable[[Path, str], None] | None" = None,  # type: ignore[valid-type]
    ) -> None:
        self._db_path = db_path
        self._interval_seconds: float = max(0.0, float(interval_minutes) * 60.0)
        self._on_regression = on_regression
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_check_at: float | None = None
        self.last_check_ok: bool | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background check thread if interval > 0.

        No-op when the configured interval is ``0`` (opt-in default).
        Calling :meth:`start` twice is safe; the second call is ignored.
        """
        with self._lock:
            if self._interval_seconds <= 0:
                logger.debug(
                    "integrity_scheduler_disabled",
                    db=str(self._db_path),
                    reason="interval_minutes=0",
                )
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"trw-memory-integrity-check[{self._db_path.name}]",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "integrity_scheduler_started",
                db=str(self._db_path),
                interval_minutes=self._interval_seconds / 60.0,
            )

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to stop and join with ``timeout`` seconds.

        Safe to call multiple times; safe to call before :meth:`start`.
        """
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.debug(
                "integrity_scheduler_stop_timed_out",
                db=str(self._db_path),
            )

    @property
    def is_running(self) -> bool:
        """True when the background thread is alive."""
        with self._lock:
            t = self._thread
        return t is not None and t.is_alive()

    def run_once(self) -> bool:
        """Run one probe synchronously and return the result.

        Useful for tests and for the C3 session-start dashboard which wants
        a fresh reading without waiting for the next scheduled tick.
        """
        ok, detail = self._probe()
        self.last_check_at = time.time()
        self.last_check_ok = ok
        self._write_sentinel()
        self._report(ok, detail)
        return ok

    # ------------------------------------------------------------------
    # Sentinel file (PRD-INFRA-063 + PRD-INFRA-068 cross-PRD contract)
    # ------------------------------------------------------------------

    def _write_sentinel(self) -> None:
        """Write ``<db_parent>/.integrity_last_check`` for C3 to read.

        The C3 session-start dashboard reads this file to report
        ``last_integrity_check_age_minutes``. Failures are silent — the
        dashboard gracefully degrades to ``None`` when the file is missing.
        """
        if self.last_check_at is None:
            return
        sentinel = self._db_path.parent / ".integrity_last_check"
        try:
            sentinel.write_text(f"{self.last_check_at:.6f}\n")
        except OSError:
            # Non-fatal: sentinel is an observability artifact only.
            return

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Wait one interval BEFORE the first check so that opening the DB
        # and calling start() doesn't spike immediate work.
        while not self._stop_event.wait(self._interval_seconds):
            ok, detail = self._probe()
            self.last_check_at = time.time()
            self.last_check_ok = ok
            self._write_sentinel()
            self._report(ok, detail)

    def _probe(self) -> tuple[bool, str]:
        """Run a single PRAGMA quick_check on a dedicated read-only connection.

        Returns:
            ``(ok, detail)`` where ``ok`` is True on a clean check and
            ``detail`` is the raw ``quick_check`` output string (or an
            error description on connect failure).
        """
        conn: sqlite3.Connection | None = None
        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
            # Belt-and-suspenders: enforce read-only at the PRAGMA layer too.
            # Reconfirms the URI mode=ro intent and blocks any accidental write.
            with contextlib.suppress(sqlite3.Error):
                conn.execute("PRAGMA query_only = 1")
            rows = conn.execute("PRAGMA quick_check").fetchall()
            detail = rows[0][0] if rows else "empty"
            ok = len(rows) == 1 and rows[0][0] == "ok"
            return ok, str(detail)
        except sqlite3.Error as exc:
            # Treat connect/query errors as a regression signal rather than
            # silently skipping — the scheduler exists to surface problems.
            return False, f"sqlite_error: {exc}"
        finally:
            if conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()

    def _report(self, ok: bool, detail: str) -> None:
        if ok:
            logger.debug(
                "integrity_scheduler_check_ok",
                db=str(self._db_path),
                detail=detail,
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
            return

        logger.warning(
            "db_integrity_regression_detected",
            db=str(self._db_path),
            detail=detail,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
        if self._on_regression is not None:
            with contextlib.suppress(Exception):
                self._on_regression(self._db_path, detail)
