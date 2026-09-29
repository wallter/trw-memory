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


def _ps(*args: str) -> str:
    try:
        # Fixed argv: pids and flags only, never a shell or test-supplied text.
        return subprocess.run(["ps", "-ww", *args], capture_output=True, text=True, check=False).stdout  # noqa: S603, S607
    except OSError:  # trw-fail-silent-allow: ps unavailable: no pid is known to be a daemon, so none is signalled
        return ""


def _is_daemon(pid: int) -> bool:
    return _DAEMON_ARGV_MARK in _ps("-o", "command=", "-p", str(pid))


def _daemon_pids() -> list[int]:
    """Every live daemon on the host. -ww: Linux ps cuts piped output to $COLUMNS, which pytest sets."""
    pids: list[int] = []
    for line in _ps("-axo", "pid=,command=").splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if _DAEMON_ARGV_MARK in command and pid_text.isdigit():
            pids.append(int(pid_text))
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # trw-fail-silent-allow: no such process means not alive
        return False
    except PermissionError:
        return True
    return True


def _discoveries_under(root: Path) -> dict[int, Path]:
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
        if pid > 0 and pid != os.getpid() and _is_daemon(pid):
            pids[pid] = discovery
    return pids


def daemon_pids_under(root: Path) -> list[int]:
    """Pids of live daemons whose discovery file lies under *root*."""
    return list(_discoveries_under(root))


def _environment(pid: int) -> dict[str, str]:
    """*pid*'s placing variables and owner token: ``/proc`` on Linux, ``ps -E`` on macOS.

    ``ps -E`` appends the environment to the command line space-separated, so a
    value containing a space is cut short there; tmp paths contain none.
    """
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace").split("\0")
    except OSError:  # trw-fail-silent-allow: no /proc (macOS): ps -E below
        raw = _ps("-E", "-o", "command=", "-p", str(pid)).split()
    found: dict[str, str] = {}
    for entry in raw:
        name, sep, value = entry.partition("=")
        if sep and (name in _PLACING_VARIABLES or name == OWNER_VARIABLE):
            found.setdefault(name, value)
    return found


def _cwd(pid: int) -> str | None:
    try:
        return str(Path(os.readlink(f"/proc/{pid}/cwd")).resolve())
    except OSError:  # trw-fail-silent-allow: no /proc (macOS): fall through to lsof
        pass
    try:
        out = subprocess.run(  # noqa: S603 -- fixed argv, the pid is an int
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:  # trw-fail-silent-allow: lsof unavailable: the cwd is unknown, so it places nothing
        return None
    names = [line[1:] for line in out.splitlines() if line.startswith("n")]
    return str(Path(names[0]).resolve()) if names else None


def daemon_pids_placed_under(root: Path) -> list[int]:
    """Pids of live daemons whose working directory, ``TRW_USER_DIR`` or ``HOME`` lies under *root*.

    Catches a daemon with no discovery file: one that stalled before publishing,
    or whose file a fixture already removed with its tmp tree.
    """
    wanted = root.resolve()
    pids: list[int] = []
    for pid in _daemon_pids():
        environment = _environment(pid)
        environment.pop(OWNER_VARIABLE, None)
        cwd = _cwd(pid)
        places = [*environment.values(), *([cwd] if cwd is not None else [])]
        if any(Path(place).resolve().is_relative_to(wanted) for place in places):
            pids.append(pid)
    return pids


def daemon_pids_owned_by(owner: str) -> list[int]:
    """Pids of live daemons that inherited ``OWNER_VARIABLE=<owner>`` from the session that started them."""
    return [pid for pid in _daemon_pids() if _environment(pid).get(OWNER_VARIABLE) == owner]


def daemon_placement(pid: int, discovery: Path | None = None) -> str:
    """Where *pid* was started from: its discovery file, owner, ``HOME``, ``TRW_USER_DIR`` and cwd.

    Each lies under the tmp tree of the test that started it, so it names that test.
    """
    places = {**_environment(pid), "cwd": _cwd(pid) or "?"}
    if discovery is not None:
        places["discovery"] = str(discovery)
    return " ".join(f"{name}={value}" for name, value in sorted(places.items()))


def reap_daemons_under(
    *roots: Path,
    wait: bool = False,
    by_process: bool = False,
    placements: dict[int, str] | None = None,
    owner: str | None = None,
) -> list[int]:
    """Stop every daemon published (or, with *by_process*, placed) under any of *roots*.

    SIGTERM lets the daemon remove its discovery file and lock. With *wait*, a
    daemon still alive after a grace period gets SIGKILL; a per-test reap does not
    wait, and the session-end sweep does. *placements*, when given, receives each
    pid's :func:`daemon_placement` before it is signalled, so a leak report can
    name the test that started it. *owner*, when given, also stops every daemon
    that inherited that session's ``OWNER_VARIABLE``.
    """
    published = {pid: path for root in roots for pid, path in _discoveries_under(root).items()}
    pids = set(published)
    if by_process:
        pids |= {pid for root in roots for pid in daemon_pids_placed_under(root)}
    if owner is not None:
        pids |= set(daemon_pids_owned_by(owner))
    pids.discard(os.getpid())
    if placements is not None:
        placements.update({pid: daemon_placement(pid, published.get(pid)) for pid in pids})
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
    leaked = reap_daemons_under(basetemp, wait=True, by_process=True, placements=placements, owner=owner)
    survivors = sorted({*daemon_pids_placed_under(basetemp), *(daemon_pids_owned_by(owner) if owner else [])})
    return SessionSweep(basetemp, leaked, survivors, placements)
