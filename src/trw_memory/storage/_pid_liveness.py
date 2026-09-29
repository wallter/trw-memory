"""Best-effort process-liveness check shared by daemon discovery.

Split out of the PRD-INFRA-064 writer registry (removed by PRD-CORE-280
FR01; its successor lock went with PRD-CORE-298 FR01, since the daemon is the
one writer): ``daemon/_discovery.py`` uses it to decide whether a daemon's
recorded pid is still running.
"""

from __future__ import annotations

import errno
import os
import struct
import sys
import time
from pathlib import Path

__all__ = ["_pid_is_live", "is_process_live", "process_start"]

#: ``SZOMB`` in XNU's ``sys/proc.h``, and ``p_stat``'s byte offset in ``struct
#: extern_proc`` (``p_un`` 16, two pointers, ``p_flag``) on 64-bit macOS. ``p_un``
#: opens with ``p_starttime``, a ``struct timeval`` (8-byte seconds, 4-byte micros).
_DARWIN_SZOMB = 5
_DARWIN_P_STAT_OFFSET = 36
_DARWIN_STARTTIME = struct.Struct("=qi")
#: ``starttime`` is field 22 of ``/proc/<pid>/stat``: index 19 after the ``)`` that ends field 2 (state is 0).
_LINUX_STARTTIME = 19
#: ``sizeof(struct kinfo_proc)`` on 64-bit macOS.
_DARWIN_KINFO_PROC_SIZE = 648

# Lock/marker files older than this are considered stale on non-POSIX hosts
# where ``/proc/<pid>`` is unavailable. Chosen to be longer than any
# reasonable process lifetime but shorter than "user manually copied the
# directory" scenarios.
_STALE_LOCK_MAX_AGE_SECONDS: float = 7 * 24 * 3600.0


def _is_zombie(pid: int) -> bool:
    """Whether *pid* has exited and waits to be reaped; ``False`` when it cannot tell."""
    if sys.platform.startswith("linux"):
        return (_linux_stat(pid) or [""])[0] == "Z"
    kinfo = _darwin_kinfo(pid) if sys.platform == "darwin" else None
    return kinfo is not None and kinfo[_DARWIN_P_STAT_OFFSET] == _DARWIN_SZOMB


def process_start(pid: int) -> str | None:
    """*pid*'s start as the OS recorded it, or ``None`` when it cannot be read (PRD-CORE-310 FR01).

    Two reads agree exactly for one process and differ for a later one that reused its
    pid. Never derived from the wall clock, which a clock step would shift: macOS
    stores the start in the process, and Linux counts ticks from a boot named by its id.
    """
    if sys.platform.startswith("linux"):
        fields = _linux_stat(pid) or []
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        except OSError:  # trw-fail-silent-allow: None is "unknown", and callers keep the pid-only answer
            return None
        return f"linux:{boot}:{fields[_LINUX_STARTTIME]}" if len(fields) > _LINUX_STARTTIME else None
    kinfo = _darwin_kinfo(pid) if sys.platform == "darwin" else None
    if kinfo is None:
        return None
    seconds, micros = _DARWIN_STARTTIME.unpack_from(kinfo)
    return f"darwin:{seconds}.{micros:06d}"


def _linux_stat(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat``'s fields after the parenthesised command (which may hold spaces), or ``None``."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:  # trw-fail-silent-allow: a pid gone mid-read is "unknown"; each caller keeps its own answer
        return None
    return stat[stat.rfind(")") + 2 :].split()


def _darwin_kinfo(pid: int) -> bytes | None:
    """*pid*'s ``struct kinfo_proc`` via ``sysctl(KERN_PROC_PID)``, or ``None``."""
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:  # trw-fail-silent-allow: None means "state unknown", which keeps the pre-zombie-check answer
        return None
    ctl_kern, kern_proc, kern_proc_pid = 1, 14, 1
    mib = (ctypes.c_int * 4)(ctl_kern, kern_proc, kern_proc_pid, pid)
    buf = ctypes.create_string_buffer(_DARWIN_KINFO_PROC_SIZE)
    size = ctypes.c_size_t(_DARWIN_KINFO_PROC_SIZE)
    if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) != 0 or size.value < _DARWIN_P_STAT_OFFSET + 1:
        return None
    return buf.raw


def _pid_is_live(pid: int, lock_file: Path) -> bool:
    """Check whether ``pid`` refers to a currently running process.

    A zombie is not live: it has exited, holds no store and serves nothing, but
    ``/proc/<pid>`` exists and ``kill(pid, 0)`` succeeds until its parent reaps it.
    Reading it as live left every client "unreachable" behind a crashed daemon
    whose parent never reaped it (2026-09-25).

    Uses ``/proc/<pid>`` on Linux. On other platforms falls back to
    ``os.kill(pid, 0)`` semantics with EPERM→live, ESRCH→dead. If neither
    check is available, treats *lock_file* younger than
    :data:`_STALE_LOCK_MAX_AGE_SECONDS` as live (conservative).
    """
    if sys.platform.startswith("linux"):
        return Path(f"/proc/{pid}").exists() and not _is_zombie(pid)

    # POSIX (macOS) — signal 0 probe.
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            if exc.errno == errno.EPERM:
                return not _is_zombie(pid)
            # EINVAL or other — fall through to mtime heuristic.
        else:
            return not _is_zombie(pid)

    # Windows or unknown — mtime heuristic.
    try:
        age = time.time() - lock_file.stat().st_mtime
    except OSError:
        return False
    return age < _STALE_LOCK_MAX_AGE_SECONDS


def is_process_live(pid: int, start: str | None, lock_file: Path) -> bool:
    """Whether the process *pid* that started at *start* still runs (PRD-CORE-310 FR01).

    ``start`` is a :func:`process_start` reading taken earlier, or ``None`` when
    none was. A different reading now means the pid was reused, so that process is
    gone. A reading that cannot be taken keeps the pid's own answer: "dead" needs
    positive evidence, or a caller could start a second daemon beside a live one.
    """
    if not _pid_is_live(pid, lock_file):
        return False
    now = process_start(pid) if start is not None else None
    return now is None or now == start
