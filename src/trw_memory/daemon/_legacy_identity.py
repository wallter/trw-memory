"""Proving a 4.0 daemon is the process its record names, without an OS start (E2E-INC-134).

A 4.0 record carries no ``process_start``, so ``SpawnedDaemon.proven`` can never vouch for its pid,
and an upgrade over a live 4.0 daemon refused to stop it: every first 8.x install after 7.0.x ended
``daemon_version_mismatch``. The pid cannot be trusted on its own (a reboot reuses pids), so this
module asks the evidence that does not need the start. ALL of these must hold, else nothing is signalled:

1. the daemon's own socket answers and reports its pid and version (a 4.x version gate names both
   when it refuses a call from another major, before the call runs, so the probe applies nothing);
2. the reported pid is the discovery record's pid, and the reported version is the record's;
3. ``ps`` shows that pid's command is ``python -m trw_memory.server``, owned by this user.

Only then is the daemon sent SIGTERM (a 4.0 daemon offers no drain handshake) and waited for.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import signal
import subprocess
import time

import httpx
import structlog

from trw_memory.daemon._discovery import VERSION_HEADER, DaemonInfo
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.storage._pid_liveness import is_process_live

__all__ = ["legacy_identity_gap", "stop_legacy_daemon"]

logger = structlog.get_logger(__name__)

#: How the 4.x version gate names itself when it refuses a call: ``... daemon (pid N) serves V, but ...``.
_GATE_REPORT = re.compile(r"daemon_version_mismatch: this trw-memory daemon \(pid (\d+)\) serves (\S+?),")
#: A tool no daemon serves: the gate refuses by the caller's major before the tool is looked up.
_PROBE_TOOL = "memory_identity_probe"
_PROBE_TIMEOUT_SECONDS = 5.0
_PS_TIMEOUT_SECONDS = 5.0
#: A 4.0 daemon finishes its calls in flight on SIGTERM; past this it is left running and reported.
_EXIT_WAIT_SECONDS = 15.0
_POLL_SECONDS = 0.05


def socket_report(info: DaemonInfo, token: str | None, mine: str) -> tuple[int, str] | None:
    """The ``(pid, version)`` the daemon at ``info.url`` reports about itself, or ``None`` when it says nothing.

    One stateless JSON-RPC ``tools/call`` of a tool nobody serves, carrying *mine* as the client version.
    """
    headers = {"accept": "application/json, text/event-stream", VERSION_HEADER: mine}
    # Deferred like the drain call's: fastmcp's import is paid only by a run that reaches a 4.0 record.
    from fastmcp.client.auth import BearerAuth

    auth = BearerAuth(token) if token is not None else None
    call = {"name": _PROBE_TOOL, "arguments": {}}
    try:
        response = httpx.post(
            info.url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": call},
            headers=headers,
            auth=auth,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:  # trw-fail-silent-allow: no report is unproven; the caller refuses
        logger.info("daemon_legacy_probe_unanswered", pid=info.pid, error=type(exc).__name__)
        return None
    texts = [str(part.get("text", "")) for part in ((body.get("result") or {}).get("content") or [])]
    texts.append(str((body.get("error") or {}).get("message", "")))
    for text in texts:
        found = _GATE_REPORT.search(text)
        if found:
            return int(found.group(1)), found.group(2)
    return None


def process_identity(pid: int) -> tuple[int, str] | None:
    """``(owner uid, command line)`` of *pid* as ``ps`` shows it, or ``None`` when ``ps`` cannot say."""
    try:
        shown = subprocess.run(  # noqa: S603 -- fixed argv, the pid is an int
            ["/bin/ps", "-o", "uid=", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=_PS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # trw-fail-silent-allow: None is "unproven", and the caller refuses
        return None
    fields = shown.stdout.strip().split(None, 1)
    if shown.returncode != 0 or len(fields) != 2 or not fields[0].isdigit():
        return None
    return int(fields[0]), fields[1]


def _is_trw_memory_server(command: str) -> bool:
    """Whether *command* is ``<python> [flags] -m trw_memory.server ...``."""
    argv = command.split()
    return (
        bool(argv)
        and os.path.basename(argv[0]).lower().startswith("python")
        and any(a == "-m" and b == "trw_memory.server" for a, b in itertools.pairwise(argv))
    )


def legacy_identity_gap(info: DaemonInfo, token: str | None, mine: str) -> str:
    """Why *info* (a record without an OS start) is not proven to be a trw-memory daemon, else ``""``."""
    reported = socket_report(info, token, mine)
    if reported is None:
        return f"its socket {info.url} did not report a pid and version"
    if reported != (info.pid, info.version):
        return f"its socket reports pid {reported[0]} serving {reported[1]}, not the record's pid {info.pid} serving {info.version}"
    owned = process_identity(info.pid)
    if owned is None:
        return f"ps cannot show process {info.pid}"
    if owned[0] != os.getuid():
        return f"process {info.pid} is owned by uid {owned[0]}, not this user"
    if not _is_trw_memory_server(owned[1]):
        return f"process {info.pid} is not `python -m trw_memory.server`"
    return ""


def stop_legacy_daemon(info: DaemonInfo, paths: DaemonPaths, token: str | None, mine: str) -> str:
    """SIGTERM a 4.0 daemon proven by :func:`legacy_identity_gap` and wait for it; ``""`` when it exited, else why not."""
    gap = legacy_identity_gap(info, token, mine)
    if gap:
        return gap
    with contextlib.suppress(ProcessLookupError):  # it exited between the proof and the signal
        os.kill(info.pid, signal.SIGTERM)
    deadline = time.monotonic() + _EXIT_WAIT_SECONDS
    while is_process_live(info.pid, None, paths.lock):
        if time.monotonic() >= deadline:
            return f"process {info.pid} did not exit within {_EXIT_WAIT_SECONDS:g}s of SIGTERM"
        time.sleep(_POLL_SECONDS)
    logger.info("daemon_legacy_stopped", pid=info.pid, served=info.version)
    return ""
