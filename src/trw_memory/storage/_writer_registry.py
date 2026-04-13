"""Multi-writer advisory registry for :class:`SQLiteBackend` (PRD-INFRA-064 / B3).

Records the calling process's pid in ``<db_path>.writers/<pid>.lock`` so that
concurrent writer count is observable in logs at open time. Advisory ONLY — the
registry NEVER refuses ``open()``. The user's deliberate 9-session workflow must
not be broken by this diagnostic layer.

Key invariants (do not relax without updating PRD-INFRA-064):

- Registry failures (FS errors, stale-pid detection failures, unwritable parent)
  MUST be swallowed and logged at DEBUG/INFO. They MUST NOT raise from the
  public API. This is the regression guard for the sprint exit criterion
  ``advisory-only-b3``.
- On non-POSIX platforms (Windows), ``/proc/<pid>`` does not exist; fall back
  to best-effort ``psutil``-free liveness check (treat files older than 7 days
  as stale; younger files kept). The 7-day horizon is conservative because
  stale ``.lock`` files are harmless — they only inflate the concurrent-writer
  count until evicted.
- The 2026-04-12 incident post-mortem inferred "9 concurrent writers" from
  timing correlation. This registry makes that signal first-class: on every
  open, log ``concurrent_writers=N peer_pids=[...]`` at INFO (or WARNING when
  ``N > threshold``).
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import os
import sys
import threading
import time
from pathlib import Path

import structlog

__all__ = ["WriterRegistry"]

logger = structlog.get_logger(__name__)

# Lock files older than this are considered stale on non-POSIX hosts where
# ``/proc/<pid>`` is unavailable. Chosen to be longer than any reasonable
# process lifetime but shorter than "user manually copied the DB dir" scenarios.
_STALE_LOCK_MAX_AGE_SECONDS: float = 7 * 24 * 3600.0


class WriterRegistry:
    """Advisory writer-count registry for one :class:`SQLiteBackend` instance.

    The registry is a directory sibling of the DB file: ``<db_path>.writers/``
    (e.g. ``/path/to/memory.db.writers/``). Each live writer drops a
    ``<pid>.lock`` file on open, prunes stale peer files, and logs the
    resulting concurrent-writer count. On close/del, its own lock file is
    removed.

    Args:
        db_path: Path to the SQLite DB file. The ``.writers/`` sibling
            directory is created under the DB's parent directory.
        warn_threshold: Threshold above which the open log escalates from
            INFO to WARNING (concurrent_writers > warn_threshold).

    Attributes:
        concurrent_writers: Number of live peer writers at registration time,
            including self. Set by :meth:`register`.
        peer_pids: Sorted list of peer pids (including self) observed at
            registration time.
    """

    __slots__ = (
        "_db_path",
        "_registry_dir",
        "_lock_file",
        "_warn_threshold",
        "_registered",
        "_atexit_handle",
        "_lock",
        "concurrent_writers",
        "peer_pids",
    )

    def __init__(self, db_path: Path, warn_threshold: int = 4) -> None:
        self._db_path = db_path
        self._registry_dir: Path = db_path.parent / f"{db_path.name}.writers"
        self._lock_file: Path = self._registry_dir / f"{os.getpid()}.lock"
        self._warn_threshold = warn_threshold
        self._registered = False
        self._atexit_handle: object | None = None
        # Guards concurrent cleanup between __del__, close(), and atexit.
        self._lock = threading.Lock()
        self.concurrent_writers: int = 0
        self.peer_pids: list[int] = []

    def register(self) -> None:
        """Drop a ``<pid>.lock`` file and log the current peer count.

        All steps are fail-open. A failure at any stage logs at DEBUG/INFO
        and returns — the registry never propagates an exception. After
        this call the instance's ``concurrent_writers`` and ``peer_pids``
        fields reflect the best-effort observation at this moment in time.
        """
        with self._lock:
            if self._registered:
                return
            try:
                self._registry_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.info(
                    "writer_registry_mkdir_failed",
                    db=str(self._db_path),
                    registry_dir=str(self._registry_dir),
                    error=str(exc),
                    outcome="advisory_skipped",
                )
                return

            # Prune stale peer locks before counting. Best-effort.
            try:
                self._prune_stale_peers()
            except OSError as exc:
                logger.debug(
                    "writer_registry_prune_failed",
                    db=str(self._db_path),
                    error=str(exc),
                )

            # Create OUR lock via O_EXCL. If our pid somehow already has a
            # lockfile (same process re-opening DB), fall through — the
            # existing file is fine.
            try:
                fd = os.open(
                    str(self._lock_file),
                    os.O_CREAT | os.O_WRONLY | os.O_EXCL,
                    0o644,
                )
                try:
                    os.write(fd, f"{os.getpid()}\n{time.time():.6f}\n".encode())
                finally:
                    os.close(fd)
            except FileExistsError:
                # Same-pid re-open — harmless. Touch mtime.
                with contextlib.suppress(OSError):
                    os.utime(str(self._lock_file), None)
            except OSError as exc:
                logger.info(
                    "writer_registry_lock_create_failed",
                    db=str(self._db_path),
                    lock_file=str(self._lock_file),
                    error=str(exc),
                    outcome="advisory_skipped",
                )
                return

            self._registered = True
            # Atexit cleanup as belt-and-suspenders to __del__.
            self._atexit_handle = atexit.register(self._unregister_silent)

            # Count live peers AFTER our own lock is in place so concurrent
            # opens see consistent state.
            peers = self._enumerate_live_peers()
            self.peer_pids = peers
            self.concurrent_writers = len(peers)

            if self.concurrent_writers > self._warn_threshold:
                logger.warning(
                    "high_concurrent_writer_count_detected",
                    db=str(self._db_path),
                    concurrent_writers=self.concurrent_writers,
                    peer_pids=peers,
                    threshold=self._warn_threshold,
                )
            else:
                logger.info(
                    "writer_registry_registered",
                    db=str(self._db_path),
                    concurrent_writers=self.concurrent_writers,
                    peer_pids=peers,
                )

    def close(self) -> None:
        """Explicit cleanup — remove our ``<pid>.lock`` file.

        Safe to call multiple times. Idempotent. Errors logged at DEBUG and
        swallowed.
        """
        self._unregister_silent()

    def _unregister_silent(self) -> None:
        with self._lock:
            if not self._registered:
                return
            self._registered = False
            try:
                self._lock_file.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(
                    "writer_registry_unlink_failed",
                    lock_file=str(self._lock_file),
                    error=str(exc),
                )
            if self._atexit_handle is not None:
                with contextlib.suppress(Exception):
                    atexit.unregister(self._atexit_handle)  # type: ignore[arg-type]
                self._atexit_handle = None

    def __del__(self) -> None:
        # Best-effort; may run during interpreter shutdown when Path/os
        # are partially torn down. Swallow everything.
        with contextlib.suppress(Exception):
            self._unregister_silent()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enumerate_live_peers(self) -> list[int]:
        """Return sorted list of pids whose ``<pid>.lock`` exists and is live."""
        try:
            files = list(self._registry_dir.glob("*.lock"))
        except OSError:
            return []

        pids: list[int] = []
        for f in files:
            pid = _parse_pid_from_lockname(f.name)
            if pid is None:
                continue
            if _pid_is_live(pid, f):
                pids.append(pid)
        return sorted(set(pids))

    def _prune_stale_peers(self) -> None:
        """Unlink ``<pid>.lock`` files whose pids no longer exist.

        Best-effort; any individual unlink failure is logged at DEBUG and
        skipped.
        """
        try:
            files = list(self._registry_dir.glob("*.lock"))
        except OSError:
            return

        for f in files:
            pid = _parse_pid_from_lockname(f.name)
            if pid is None:
                # Unparseable name — leave it alone.
                continue
            if pid == os.getpid():
                # Never prune our own lockfile from this code path; register()
                # owns creation and close() owns removal.
                continue
            if _pid_is_live(pid, f):
                continue
            try:
                f.unlink(missing_ok=True)
                logger.debug(
                    "writer_registry_pruned_stale",
                    lock_file=str(f),
                    stale_pid=pid,
                )
            except OSError as exc:
                logger.debug(
                    "writer_registry_prune_unlink_failed",
                    lock_file=str(f),
                    error=str(exc),
                )


def _parse_pid_from_lockname(name: str) -> int | None:
    """Return pid from ``<pid>.lock`` or None if unparseable."""
    if not name.endswith(".lock"):
        return None
    stem = name[: -len(".lock")]
    try:
        pid = int(stem)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_is_live(pid: int, lock_file: Path) -> bool:
    """Check whether ``pid`` refers to a currently running process.

    Uses ``/proc/<pid>`` on Linux. On other platforms falls back to
    ``os.kill(pid, 0)`` semantics with EPERM→live, ESRCH→dead. If neither
    check is available, treats lock files younger than
    :data:`_STALE_LOCK_MAX_AGE_SECONDS` as live (conservative).
    """
    if sys.platform.startswith("linux"):
        return Path(f"/proc/{pid}").exists()

    # POSIX (macOS) — signal 0 probe.
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            if exc.errno == errno.EPERM:
                return True
            # EINVAL or other — fall through to mtime heuristic.
        else:
            return True

    # Windows or unknown — mtime heuristic.
    try:
        age = time.time() - lock_file.stat().st_mtime
    except OSError:
        return False
    return age < _STALE_LOCK_MAX_AGE_SECONDS
