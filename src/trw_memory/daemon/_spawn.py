"""Starting the daemon so it is nobody's child -- PRD-CORE-310 FR03.

A client that finds no daemon starts one, and that daemon then serves every
session on the machine. So it must not belong to the client that happened to
start it. When the client was its parent, the client had to reap it for its
whole life (a crashed daemon it never reaped stayed a zombie and held the slot,
2026-09-25), and a harness that kills an MCP server's process tree took the
shared daemon with it.

A short-lived launcher (a fresh interpreter, so no fork of this multi-threaded
process) starts the daemon in its own session and exits at once. The client
waits for the launcher, and the daemon's parent is init, or the session's
subreaper, from its first instant. A daemon launched here is then named by its
pid and its OS start (:class:`SpawnedDaemon`), so stopping one that never
published cannot signal a later process that reused the pid.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from trw_memory.daemon._discovery import DaemonInfo, DiscoveryInvalid, read_live_discovery
from trw_memory.daemon._paths import DaemonPaths, open_private_log
from trw_memory.daemon._versions import is_older
from trw_memory.exceptions import DaemonUnreachableError
from trw_memory.storage._pid_liveness import is_process_live, process_start

__all__ = ["OutdatedDaemonStop", "SpawnedDaemon", "start_daemon_detached", "stop_outdated_daemon"]

logger = structlog.get_logger(__name__)

#: The auto-started daemon's argv after the interpreter: the module entry point,
#: so auto-start does not depend on the console script being on ``PATH``.
_DAEMON_ARGV = ("-m", "trw_memory.server", "serve", "http")

#: Starts ``argv[1:]`` in a new session, prints its pid, and exits without waiting for it.
_LAUNCHER = (
    "import subprocess, sys\n"
    "daemon = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,"
    " start_new_session=True)\n"
    "print(daemon.pid)\n"
)

#: The launcher only starts a process; one that takes this long is stuck, not slow.
_LAUNCH_TIMEOUT_SECONDS = 30.0

#: How long a stopped daemon gets to exit on SIGTERM before SIGKILL, and on SIGKILL before giving up.
_STOP_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class SpawnedDaemon:
    """A daemon this process launched, named by pid and OS start."""

    pid: int
    start: str | None
    lock_file: Path

    def running(self) -> bool:
        """Whether this very process still runs (a reused pid is another process)."""
        return is_process_live(self.pid, self.start, self.lock_file)

    def proven(self) -> bool:
        """Whether the pid is still THIS process by its start read NOW: the precondition of every signal.

        :meth:`running` keeps a live pid when its start cannot be read (so no second daemon starts beside
        a live one); a signal needs the opposite default, since an unreadable start proves nothing.
        A pid of 1 or less (init, or a process group) is never a daemon, whatever its start reads.
        """
        return self.pid > 1 and self.start is not None and process_start(self.pid) == self.start and self.running()

    def stop(self) -> bool:
        """SIGTERM it, then SIGKILL after the grace; whether it was still running.

        Each signal is sent only while :meth:`proven`, so a pid reused (or unreadable) since is never signalled.
        """
        if self.start is None:
            logger.warning("daemon_stop_refused_unknown_start", pid=self.pid)
            return False
        if not self.proven():
            return False
        for signum in (signal.SIGTERM, signal.SIGKILL):
            if not self.proven():
                break
            with contextlib.suppress(ProcessLookupError):  # it exited between the check and the signal
                os.kill(self.pid, signum)
            deadline = time.monotonic() + _STOP_GRACE_SECONDS
            while self.running() and time.monotonic() < deadline:
                time.sleep(0.05)
        return True


def start_daemon_detached(paths: DaemonPaths) -> SpawnedDaemon:
    """Launch a daemon that is not this process's child, and return it.

    Its stderr goes to :attr:`DaemonPaths.start_log`, emptied on each start, so a
    start that stalls or crashes before publishing leaves a reason behind. The
    daemon installs no log handlers, so stderr carries warnings and tracebacks only.

    Raises:
        DaemonUnreachableError: The launcher failed or did not report a pid.
    """
    logger.info("daemon_auto_start", discovery=str(paths.discovery))
    log = open_private_log(paths.start_log)
    try:
        launched = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and module constants
            [sys.executable, "-I", "-c", _LAUNCHER, sys.executable, *_DAEMON_ARGV],  # -I: no cwd module shadows
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            check=True,
            timeout=_LAUNCH_TIMEOUT_SECONDS,
        )
        pid = int(launched.stdout.split()[-1])  # the last line: a site hook may print before it
        if pid <= 1:  # init or a process group: a stubbed launcher (int(MagicMock()) == 1), never a daemon
            raise ValueError(f"the launcher reported pid {pid}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise DaemonUnreachableError(
            f"the trw-memory daemon launcher failed ({type(exc).__name__}); its stderr is in {paths.start_log}"
        ) from exc
    finally:
        os.close(log)
    return SpawnedDaemon(pid, process_start(pid), paths.lock)


@dataclass(frozen=True)
class OutdatedDaemonStop:
    """What :func:`stop_outdated_daemon` did, and the detail an operator acts on."""

    outcome: Literal["absent", "current", "not_older", "changed", "stopped", "unproven", "invalid"]
    detail: str


def stop_outdated_daemon(
    paths: DaemonPaths, version: str, *, expect: DaemonInfo | None = None, older_only: bool = False
) -> OutdatedDaemonStop:
    """Stop the daemon when it serves a trw-memory other than *version* (PRD-INFRA-200 FR02).

    A client refuses only a MAJOR mismatch, so after a minor or patch upgrade the
    old daemon kept serving old code until someone killed it. The record is read
    here, at signal time, never trusted from an earlier read; a record naming a
    dead, zombie or reused pid is absent, and one whose OS start is missing (a 4.0
    daemon) or cannot be read now is ``unproven``: its pid may name any process,
    so it is never signalled and the detail is the manual remedy.

    *expect* is the instance the caller observed: a record naming another pid or OS start
    at signal time is ``changed`` and never signalled. *older_only* stops only a daemon
    provably older than *version* (numeric order; an unparseable version is not older),
    so an equal or newer daemon is ``current`` or ``not_older`` and left running.
    """
    found = read_live_discovery(paths)
    if isinstance(found, DiscoveryInvalid):
        return OutdatedDaemonStop("invalid", f"{found.path} cannot be trusted ({found.reason})")
    if not isinstance(found, DaemonInfo):
        return OutdatedDaemonStop("absent", found.reason)
    if expect is not None and (found.pid, found.process_start) != (expect.pid, expect.process_start):
        return OutdatedDaemonStop(
            "changed", f"the record now names pid {found.pid}, not the observed pid {expect.pid}; nothing was signalled"
        )
    if found.version == version:
        return OutdatedDaemonStop("current", f"pid {found.pid} already serves trw-memory {version}")
    if older_only and not is_older(found.version, version):
        return OutdatedDaemonStop(
            "not_older", f"pid {found.pid} serves trw-memory {found.version}, not older than {version}; left running"
        )
    daemon = SpawnedDaemon(found.pid, found.process_start, paths.lock)
    if not daemon.proven():
        return OutdatedDaemonStop(
            "unproven", f"trw-memory {found.version} daemon: {found.stop_remedy(paths.discovery)}"
        )
    if not daemon.stop():  # it exited, or its identity changed, between the check and the first signal
        return OutdatedDaemonStop("unproven", f"pid {found.pid} was not signalled: its identity was not proven")
    logger.info("daemon_outdated_stopped", pid=found.pid, served=found.version, installed=version)
    return OutdatedDaemonStop("stopped", f"stopped pid {found.pid} (trw-memory {found.version}; installed {version})")
