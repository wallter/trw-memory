"""Test support: stop the memory daemons a test session started. Not runtime API.

A client that finds no daemon starts one (``start_daemon_detached``), detached,
so it outlives the test that triggered it: one ``-n 8`` run once left ~450 of
them, and on 2026-09-26 the host ran 15 orphans at load >20. Test daemons detach
on purpose, so a parent pid says nothing about ownership. This module finds a
daemon four ways:

* by the discovery file it publishes under a test's tmp tree;
* by placement: a working directory, ``TRW_USER_DIR`` or ``HOME`` under a root
  (a daemon that stalls before publishing has no discovery file, 2026-09-24);
* by owner: every test session exports its own ``TRW_PYTEST_DAEMON_OWNER``
  token (:func:`tag_daemon_ownership`), which an auto-started daemon inherits,
  so a session finds the daemons it started wherever they were placed;
* by the spawn handle a test's own client returned (:func:`stop_spawned`).

The session also caps a daemon's idle life at 60 s, so one a killed session
could not reap still exits. Only a process whose command line is the daemon's is
signalled, so a recycled pid is never hit. Stdlib only, and no pytest import:
this module ships in the wheel.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "IDLE_VARIABLE",
    "OWNER_VARIABLE",
    "TEST_IDLE_SECONDS",
    "SessionSweep",
    "daemon_env_passthrough",
    "daemon_pids_owned_by",
    "daemon_pids_placed_under",
    "daemon_pids_under",
    "daemon_placement",
    "reap_daemons_under",
    "stop_spawned",
    "sweep_session_daemons",
    "tag_daemon_ownership",
]

#: Set by the test session that owns a daemon; an auto-started daemon inherits it.
OWNER_VARIABLE = "TRW_PYTEST_DAEMON_OWNER"
#: ``MemoryConfig.memory_daemon_idle_shutdown_seconds`` from the environment.
IDLE_VARIABLE = "MEMORY_DAEMON_IDLE_SHUTDOWN_SECONDS"
#: The field's floor, instead of the 30-minute production default.
TEST_IDLE_SECONDS = "60"

_DISCOVERY_FILE = "daemon.json"
_DAEMON_ARGV_MARK = "trw_memory.server serve"
_GRACE_SECONDS = 5.0
#: The variables that place a daemon's store; an auto-started daemon inherits them.
_PLACING_VARIABLES = ("TRW_USER_DIR", "HOME")


class _Stoppable(Protocol):
    """The part of ``trw_memory.daemon._spawn.SpawnedDaemon`` this module uses."""

    @property
    def pid(self) -> int: ...

    def stop(self) -> bool: ...


def tag_daemon_ownership() -> str:
    """Give this process its own daemon-owner token, and cap daemon idle life; return the token.

    Call it once per pytest process (the xdist controller and each worker alike),
    from ``pytest_configure``. A test that needs another idle value sets its own.
    """
    token = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    os.environ[OWNER_VARIABLE] = token
    os.environ.setdefault(IDLE_VARIABLE, TEST_IDLE_SECONDS)
    return token


def daemon_env_passthrough() -> dict[str, str]:
    """The owner token and idle cap from this process, for a child env built from an allowlist.

    Merge it into a sanitized child env so a daemon the child starts is still this
    session's. It passes exactly these two variables, and only when set.
    """
    return {name: os.environ[name] for name in (OWNER_VARIABLE, IDLE_VARIABLE) if name in os.environ}


def _run(argv: list[str]) -> str:
    try:
        # Fixed argv: pids and flags only, never a shell or test-supplied text.
        return subprocess.run(argv, capture_output=True, text=True, check=False).stdout  # noqa: S603
    except OSError:  # trw-fail-silent-allow: ps/lsof unavailable: no pid is known to be a daemon, so none is signalled
        return ""


class _Census:
    """One read of the host's process table, shared by every pass of a sweep.

    ``ps`` runs once (lazily) and ``lsof`` once, for every daemon's cwd together, however many
    daemons are alive: a per-daemon call cost ~0.2 s each with foreign daemons on the host.
    -ww: Linux ps cuts piped output to $COLUMNS, which pytest sets. On macOS ``-E`` appends
    the environment to each command line; a value containing a space is cut short there, and
    tmp paths contain none.
    """

    def __init__(self, previous: _Census | None = None) -> None:
        self._commands: dict[int, str] | None = None
        # A daemon's cwd never changes, so a later census reuses the earlier one's answers.
        self._cwds: dict[int, str | None] = dict(previous._cwds) if previous else {}

    @property
    def commands(self) -> dict[int, str]:
        """Every live daemon on the host: pid -> its command line (plus environment on macOS)."""
        if self._commands is None:
            self._commands = {}
            # No /proc (macOS): -E appends each process's environment, which _environment reads.
            environ = [] if os.path.isdir("/proc") else ["-E"]
            for line in _run(["ps", "-ww", *environ, "-axo", "pid=,command="]).splitlines():
                pid_text, _, command = line.strip().partition(" ")
                if _DAEMON_ARGV_MARK in command and pid_text.isdigit():
                    self._commands[int(pid_text)] = command
        return self._commands

    def environment(self, pid: int) -> dict[str, str]:
        """*pid*'s placing variables and owner token: ``/proc`` on Linux, the census line on macOS."""
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:  # trw-fail-silent-allow: no /proc (macOS): the census line carries the environment
            raw = self.commands.get(pid, "").split()
        found: dict[str, str] = {}
        for entry in raw:
            name, sep, value = entry.partition("=")
            if sep and (name in _PLACING_VARIABLES or name == OWNER_VARIABLE):
                found.setdefault(name, value)
        return found

    def cwd(self, pid: int) -> str | None:
        try:
            return str(Path(os.readlink(f"/proc/{pid}/cwd")).resolve())
        except OSError:  # trw-fail-silent-allow: no /proc (macOS): one lsof for every daemon
            pass
        if pid not in self._cwds:
            self._load_cwds([p for p in self.commands if p not in self._cwds] or [pid])
        return self._cwds.get(pid)

    def _load_cwds(self, pids: list[int]) -> None:
        current: int | None = None
        found: dict[int, str] = {}
        for line in _run(["lsof", "-a", "-d", "cwd", "-p", ",".join(map(str, pids)), "-Fn"]).splitlines():
            if line.startswith("p") and line[1:].isdigit():
                current = int(line[1:])
            elif line.startswith("n") and current is not None:
                found.setdefault(current, str(Path(line[1:]).resolve()))
        for pid in pids:
            self._cwds[pid] = found.get(pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # trw-fail-silent-allow: no such process means not alive
        return False
    except PermissionError:
        return True
    return True


def _discoveries_under(root: Path, census: _Census) -> dict[int, Path]:
    """Each live daemon published under *root*, and the discovery file that names it."""
    pids: dict[int, Path] = {}
    if not root.is_dir():
        return pids
    for discovery in root.rglob(_DISCOVERY_FILE):
        try:
            pid = int(json.loads(discovery.read_text(encoding="utf-8"))["pid"])
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ):  # trw-fail-silent-allow: an unreadable or partial discovery file names no daemon to stop
            continue
        if pid > 0 and pid != os.getpid() and pid in census.commands:
            pids[pid] = discovery
    return pids


def daemon_pids_under(root: Path) -> list[int]:
    """Pids of live daemons whose discovery file lies under *root*."""
    return list(_discoveries_under(root, _Census()))


def daemon_pids_placed_under(root: Path, census: _Census | None = None) -> list[int]:
    """Pids of live daemons whose working directory, ``TRW_USER_DIR`` or ``HOME`` lies under *root*.

    Catches a daemon with no discovery file: one that stalled before publishing,
    or whose file a fixture already removed with its tmp tree.
    """
    census = census or _Census()
    wanted = root.resolve()
    pids: list[int] = []
    for pid in census.commands:
        environment = census.environment(pid)
        environment.pop(OWNER_VARIABLE, None)
        cwd = census.cwd(pid)
        places = [*environment.values(), *([cwd] if cwd is not None else [])]
        if any(Path(place).resolve().is_relative_to(wanted) for place in places):
            pids.append(pid)
    return pids


def daemon_pids_owned_by(owner: str, census: _Census | None = None) -> list[int]:
    """Pids of live daemons that inherited ``OWNER_VARIABLE=<owner>`` from the session that started them."""
    census = census or _Census()
    return [pid for pid in census.commands if census.environment(pid).get(OWNER_VARIABLE) == owner]


def daemon_placement(pid: int, discovery: Path | None = None, census: _Census | None = None) -> str:
    """Where *pid* was started from: its discovery file, owner, ``HOME``, ``TRW_USER_DIR`` and cwd.

    Each lies under the tmp tree of the test that started it, so it names that test.
    """
    census = census or _Census()
    places = {**census.environment(pid), "cwd": census.cwd(pid) or "?"}
    if discovery is not None:
        places["discovery"] = str(discovery)
    return " ".join(f"{name}={value}" for name, value in sorted(places.items()))


def reap_daemons_under(
    *roots: Path,
    wait: bool = False,
    by_process: bool = False,
    placements: dict[int, str] | None = None,
    owner: str | None = None,
    census: _Census | None = None,
) -> list[int]:
    """Stop every daemon published (or, with *by_process*, placed) under any of *roots*.

    SIGTERM lets the daemon remove its discovery file and lock. With *wait*, a
    daemon still alive after a grace period gets SIGKILL; a per-test reap does not
    wait, and the session-end sweep does. *placements*, when given, receives each
    pid's :func:`daemon_placement` before it is signalled, so a leak report can
    name the test that started it. *owner*, when given, also stops every daemon
    that inherited that session's ``OWNER_VARIABLE``. *census* shares one process-table
    read across the passes (see :class:`_Census`); without it the reap takes its own.
    """
    census = census or _Census()
    published = {pid: path for root in roots for pid, path in _discoveries_under(root, census).items()}
    pids = set(published)
    if by_process:
        pids |= {pid for root in roots for pid in daemon_pids_placed_under(root, census)}
    if owner is not None:
        pids |= set(daemon_pids_owned_by(owner, census))
    pids.discard(os.getpid())
    if placements is not None:
        placements.update({pid: daemon_placement(pid, published.get(pid), census) for pid in pids})
    _signal_all(pids, signal.SIGTERM)
    if wait:
        deadline = time.monotonic() + _GRACE_SECONDS
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
        _signal_all([pid for pid in pids if _alive(pid)], signal.SIGKILL)
    return sorted(pids)


def _signal_all(pids: Iterable[int], signum: signal.Signals) -> None:
    for pid in pids:
        if pid <= 1:  # init or a process group, from a corrupt discovery file: never signalled
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:  # trw-fail-silent-allow: the process already exited, which is the goal
            continue


def stop_spawned(daemons: Sequence[_Stoppable]) -> list[int]:
    """Stop the daemons a test's own client spawned; the pids that were still running.

    A reap by discovery file misses a daemon that has not published yet (an
    auto-start takes seconds). The spawn's own handle (pid plus OS start) needs no
    discovery file and no process scan.
    """
    return [daemon.pid for daemon in daemons if daemon.stop()]


@dataclass(frozen=True)
class SessionSweep:
    """What a session-end sweep found: daemons it had to stop, and daemons still alive after it."""

    basetemp: Path
    #: Pids this process's sweep stopped. Under xdist a worker hands these to the controller.
    leaked: list[int]
    #: Pids alive after the sweep: the sweep itself regressed.
    survivors: list[int]
    placements: dict[int, str] = field(default_factory=dict)

    def report(self, handed_over: Sequence[int] = (), handed_over_survivors: Sequence[int] = ()) -> list[str]:
        """The failure lines for this sweep plus what xdist workers handed over; empty when clean.

        A worker's own exit status never reaches the controller, so a worker hands both its leaks
        (*handed_over*) and its sweep's survivors (*handed_over_survivors*) up for the controller to fail on.
        """
        leaked = [*self.leaked, *handed_over]
        survivors = [*self.survivors, *handed_over_survivors]
        lines: list[str] = []
        if leaked:
            lines.append(
                f"FAIL: {len(leaked)} leaked memory daemon(s) stopped at session end under {self.basetemp}: "
                f"pids {leaked}"
            )
            lines += [f"  leaked daemon {pid}: {self.placements.get(pid, '?')}" for pid in leaked]
        if survivors:
            lines.append(f"FAIL: memory daemon(s) survived the session-end sweep: pids {survivors}")
        return lines


def sweep_session_daemons(basetemp: Path, owner: str | None) -> SessionSweep:
    """Stop every daemon published or placed under *basetemp*, or owned by *owner*, then census survivors.

    A daemon still found afterwards means the sweep itself regressed, and the
    caller fails the session: this is the survivor guard both suites share.
    """
    placements: dict[int, str] = {}
    census = _Census()
    leaked = reap_daemons_under(basetemp, wait=True, by_process=True, placements=placements, owner=owner, census=census)
    after = _Census(census)  # the daemons alive now: one fresh ps, the cwds already known
    survivors = sorted(
        {*daemon_pids_placed_under(basetemp, after), *(daemon_pids_owned_by(owner, after) if owner else [])}
    )
    return SessionSweep(basetemp, leaked, survivors, placements)
