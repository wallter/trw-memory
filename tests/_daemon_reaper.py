"""Stop the memory daemons a test left running (PRD-INFRA-196-FR07).

Byte-identical in shape to ``trw-mcp/tests/_daemon_reaper.py`` (that module's
own docstring explains why: trw-mcp and trw-memory ship independently to
PyPI, so a new cross-package test dependency for ~140 lines is not worth it).
A client that finds no daemon starts one (``start_daemon_detached``),
detached, with a 30-minute idle exit; a test that reaches the store without
stopping it leaves that daemon running past its own tmp tree.

A daemon is identified by the discovery file it publishes in its user memory
directory, and at session end also by a working directory, ``TRW_USER_DIR`` or
``HOME`` under the basetemp: a daemon that stalls before publishing has no
discovery file. Only a process whose command line is the daemon's is
signalled, so a recycled pid is never hit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

_DISCOVERY_FILE = "daemon.json"
_DAEMON_ARGV_MARK = "trw_memory.server serve"
_GRACE_SECONDS = 5.0
#: The variables that place a daemon's store; an auto-started daemon inherits them.
_PLACING_VARIABLES = ("TRW_USER_DIR", "HOME")


def _is_daemon(pid: int) -> bool:
    try:
        command = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout
    except (
        OSError
    ):  # trw-fail-silent-allow: ps unavailable: the pid is not known to be a daemon, so it is never signalled
        return False
    return _DAEMON_ARGV_MARK in command


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # trw-fail-silent-allow: no such process means not alive
        return False
    except PermissionError:
        return True
    return True


def daemon_pids_under(root: Path) -> list[int]:
    """Pids of live daemons whose discovery file lies under *root*."""
    pids: list[int] = []
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
            pids.append(pid)
    return pids


def daemon_pids_placed_under(root: Path) -> list[int]:
    """Pids of live daemons whose working directory, ``TRW_USER_DIR`` or ``HOME`` lies under *root*.

    Catches a daemon with no discovery file: one that stalled before publishing,
    or whose file a fixture already removed with its tmp tree. An auto-started
    daemon inherits the cwd and environment of the process that started it.
    """
    # -ww: Linux ps cuts piped output to $COLUMNS (pytest sets it), which drops the command-line mark.
    listing = subprocess.run(["ps", "-ww", "-axo", "pid=,command="], capture_output=True, text=True, check=False).stdout
    wanted = root.resolve()
    pids: list[int] = []
    for line in listing.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if _DAEMON_ARGV_MARK not in command or not pid_text.isdigit():
            continue
        cwd = _cwd(int(pid_text))
        places = [*_placing_environment(int(pid_text)).values(), *([cwd] if cwd is not None else [])]
        if any(Path(place).resolve().is_relative_to(wanted) for place in places):
            pids.append(int(pid_text))
    return pids


def _placing_environment(pid: int) -> dict[str, str]:
    """*pid*'s ``TRW_USER_DIR`` and ``HOME``: ``/proc`` on Linux, ``ps -E`` on macOS.

    ``ps -E`` appends the environment to the command line space-separated, so a
    value containing a space is cut short there; tmp paths contain none.
    """
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace").split("\0")
    except OSError:  # trw-fail-silent-allow: no /proc (macOS): ps -E below
        raw = subprocess.run(
            ["ps", "-ww", "-E", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout.split()
    found: dict[str, str] = {}
    for entry in raw:
        name, sep, value = entry.partition("=")
        if sep and name in _PLACING_VARIABLES:
            found.setdefault(name, value)
    return found


def _cwd(pid: int) -> str | None:
    try:
        return str(Path(os.readlink(f"/proc/{pid}/cwd")).resolve())
    except OSError:  # trw-fail-silent-allow: no /proc (macOS): fall through to lsof
        pass
    out = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True, check=False
    ).stdout
    names = [line[1:] for line in out.splitlines() if line.startswith("n")]
    return str(Path(names[0]).resolve()) if names else None


def reap_daemons_under(*roots: Path, wait: bool = False, by_process: bool = False) -> list[int]:
    """Stop every daemon published (or, with *by_process*, placed) under any of *roots*.

    SIGTERM lets the daemon remove its discovery file and lock. With *wait*, a
    daemon still alive after a grace period gets SIGKILL; a per-test reap does not
    wait, and the session-end sweep does.
    """
    pids = {pid for root in roots for pid in daemon_pids_under(root)}
    if by_process:
        pids |= {pid for root in roots for pid in daemon_pids_placed_under(root)}
    pids.discard(os.getpid())
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:  # trw-fail-silent-allow: the process already exited, which is the goal
            continue
    if wait:
        deadline = time.monotonic() + _GRACE_SECONDS
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
        for pid in pids:
            if _alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:  # trw-fail-silent-allow: the process already exited, which is the goal
                    pass
    return sorted(pids)
