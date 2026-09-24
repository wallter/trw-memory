"""Best-effort process-liveness check shared by daemon discovery.

Split out of the PRD-INFRA-064 writer registry (removed by PRD-CORE-280
FR01; its successor lock went with PRD-CORE-298 FR01, since the daemon is the
one writer): ``daemon/_discovery.py`` uses it to decide whether a daemon's
recorded pid is still running.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path

__all__ = ["_pid_is_live"]

# Lock/marker files older than this are considered stale on non-POSIX hosts
# where ``/proc/<pid>`` is unavailable. Chosen to be longer than any
# reasonable process lifetime but shorter than "user manually copied the
# directory" scenarios.
_STALE_LOCK_MAX_AGE_SECONDS: float = 7 * 24 * 3600.0


def _pid_is_live(pid: int, lock_file: Path) -> bool:
    """Check whether ``pid`` refers to a currently running process.

    Uses ``/proc/<pid>`` on Linux. On other platforms falls back to
    ``os.kill(pid, 0)`` semantics with EPERM→live, ESRCH→dead. If neither
    check is available, treats *lock_file* younger than
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
