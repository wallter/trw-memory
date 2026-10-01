"""Replacing a major-older daemon through its drain handshake -- DAEMON-AUTO-RESTART-ON-UPGRADE.

A client that finds a daemon of another major refuses it (PRD-CORE-302 C7), and
before this the user had to stop the old daemon by hand after every major
upgrade. :func:`replace_older_daemon` removes that step, but only when every
condition that makes it the software's own safe act holds, in this order:

1. the client is not pinned to one instance (a pinned caller checked THAT daemon);
2. auto-start is on (off means a supervisor owns the daemon);
3. the daemon is a MAJOR version older (a newer one is never retired);
4. it advertises the drain handshake (every 4.x and 5.0.0 daemon does not; a 4.x record, which has no OS start
   either, is instead stopped on the socket, record and ``ps`` proof of ``_legacy_identity``);
5. its pid and OS start prove it is the process its record names;
6. its drain key (:mod:`._drain_key`) is readable as this user's private file.

It then asks the daemon to drain (``memory_drain``, presenting that key), which finishes the calls in
flight and exits by itself: nothing is signalled (but a proven 4.x daemon, which has no handshake, gets
SIGTERM). A bounded wait for the record
to withdraw follows; a record still there refuses, so no daemon is ever started
beside it. Any failed condition returns the reason, and the caller keeps the
refusal it gave before, naming that reason.
"""

from __future__ import annotations

import math
import time
from typing import Any

import httpx
import structlog

from trw_memory.daemon._discovery import DRAIN_CAPABILITY, DRAIN_TOOL, VERSION_HEADER, DaemonInfo, read_live_discovery
from trw_memory.daemon._drain_key import read_drain_key
from trw_memory.daemon._legacy_identity import stop_legacy_daemon
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._spawn import SpawnedDaemon
from trw_memory.daemon._versions import major
from trw_memory.models.config import MemoryConfig

__all__ = ["drain_daemon", "replace_older_daemon", "withdrawn"]

logger = structlog.get_logger(__name__)

#: How often the withdrawal wait re-reads the discovery file.
_POLL_SECONDS = 0.05
#: Room past the drain's own deadline for the HTTP exchange that carries it.
_CALL_MARGIN_SECONDS = 5.0


def withdrawn(paths: DaemonPaths, observed: DaemonInfo) -> bool:
    """Whether the record no longer names *observed*: withdrawn, replaced, dead or untrusted.

    THE identity predicate for "has that daemon gone": the client's retry wait uses it too.
    """
    current = read_live_discovery(paths)
    return not isinstance(current, DaemonInfo) or (current.pid, current.started_at) != (
        observed.pid,
        observed.started_at,
    )


def await_withdrawal(paths: DaemonPaths, observed: DaemonInfo, timeout: float) -> bool:
    """Block up to *timeout* seconds until :func:`withdrawn`; whether it was."""
    deadline = time.monotonic() + timeout
    while not withdrawn(paths, observed):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def restart_blocker(info: DaemonInfo, config: MemoryConfig, *, pinned: bool, mine: str) -> str:
    """Why this client may not replace *info* (pinned, supervised or not major-older), else ``""``."""
    if pinned:
        return "this client is pinned to one daemon instance"
    if not config.memory_daemon_autostart:
        return "auto-start is off (MEMORY_DAEMON_AUTOSTART=false), so a supervisor owns the daemon"
    theirs, ours = major(info.version), major(mine)
    if theirs is None or ours is None or theirs >= ours:
        return f"trw-memory {info.version} is not a major version older than {mine}"
    return ""


def _request_drain(info: DaemonInfo, token: str, mine: str, deadline: float, admin_key: str) -> dict[str, Any]:
    """Call ``memory_drain`` on *info* in ONE request; its answer (``declined`` when the daemon refused).

    One stateless JSON-RPC POST, synchronous like the attach path, rather than an MCP client session:
    a session sends further requests after the call, and a draining daemon answers those 503.
    """
    # Deferred: this module is on the daemon client's import path, and fastmcp's import costs ~0.3 s
    # that the client's direct reads never need (PRD-CORE-333 S3b). DRAIN_TOOL comes from _discovery,
    # so _drain (a server tool, sqlite_vec -> numpy) is not imported here at all.
    from fastmcp.client.auth import BearerAuth

    call = {"name": DRAIN_TOOL, "arguments": {"deadline_seconds": deadline, "admin_key": admin_key}}
    response = httpx.post(
        info.url,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": call},
        auth=BearerAuth(token),  # the attach transport's own bearer scheme (``auth=token``), never a raw header
        headers={"accept": "application/json, text/event-stream", VERSION_HEADER: mine},
        timeout=deadline + _CALL_MARGIN_SECONDS,
    )
    response.raise_for_status()
    result = response.json().get("result") or {}
    if result.get("isError"):
        return {
            "status": "declined",
            "detail": " ".join(str(part.get("text", "")) for part in result.get("content", [])),
        }
    answer = result.get("structuredContent")
    return answer if isinstance(answer, dict) else {"status": "unparseable", "detail": response.text[:200]}


def _drain(info: DaemonInfo, paths: DaemonPaths, token: str, mine: str, timeout: float) -> str:
    """The drain handshake on *info* and the wait for its record to withdraw; ``""`` when the slot is free."""
    if DRAIN_CAPABILITY not in info.capabilities:
        if info.process_start is not None:
            return "it does not offer the drain handshake (trw-memory 5.0.0 and older)"
        return _stop_start_less(info, paths, token, mine, timeout)
    if not SpawnedDaemon(info.pid, info.process_start, paths.lock).proven():
        return f"process {info.pid} could not be proven to be the daemon its record names"
    admin_key = read_drain_key(paths)
    if admin_key is None:
        return f"its drain key {paths.drain_key} is absent, a symlink, or readable by other users"
    try:
        answer = _request_drain(info, token, mine, timeout, admin_key)
    except Exception as exc:  # the drain's answer was lost: only the record now says whether it took effect
        logger.warning("daemon_drain_unanswered", pid=info.pid, error=type(exc).__name__)
        answer = {"status": "unanswered"}
    status = answer.get("status")
    if status not in {"drained", "draining", "unanswered"}:
        return f"it did not drain ({status}: {answer.get('detail', '')})"
    if not await_withdrawal(paths, info, timeout):
        return (
            f"it did not withdraw its record within {timeout}s of the drain ({status}), so no daemon was "
            f"started beside it"
        )
    logger.info("daemon_replaced_older", pid=info.pid, served=info.version, installed=mine)
    return ""


def _stop_start_less(info: DaemonInfo, paths: DaemonPaths, token: str, mine: str, timeout: float) -> str:
    """A 4.0 daemon (no OS start, no drain handshake): stopped only on the socket, record and ``ps`` proof of
    :func:`stop_legacy_daemon`, the same one the installer's stop uses (E2E-INC-136); ``""`` when its slot is free."""
    gap = stop_legacy_daemon(info, paths, token, mine)
    if gap:
        return f"it does not offer the drain handshake (trw-memory 4.x) and its identity is not proven: {gap}"
    return "" if await_withdrawal(paths, info, timeout) else f"it did not exit within {timeout}s of the stop"


def drain_daemon(paths: DaemonPaths, *, token: str, mine: str, timeout: float | None = None) -> str:
    """Drain the live daemon whatever its version (a hot reload, not an upgrade); ``""`` on success, else why not.

    The same pid + ``process_start`` + drain-key checks as :func:`replace_older_daemon`, without its
    major-older gate. *timeout* bounds the drain and the withdrawal wait (default: the configured daemon
    startup timeout). Sync-only: it blocks with ``time.sleep``, so call it from a thread inside async code.
    Residual risk, pre-existing and shared with :func:`replace_older_daemon`: the pid proof and the drain
    POST are not atomic, so a pid reused in that gap is not re-proven.
    """
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
        raise ValueError(f"timeout must be a finite number of seconds above zero, got {timeout!r}")
    info = read_live_discovery(paths)
    if not isinstance(info, DaemonInfo):
        return "no live daemon record to drain"
    return _drain(info, paths, token, mine, timeout or MemoryConfig().memory_daemon_startup_timeout_seconds)


def replace_older_daemon(
    info: DaemonInfo, paths: DaemonPaths, config: MemoryConfig, token: str, *, pinned: bool, mine: str
) -> str:
    """Drain *info* and wait for its record to withdraw; ``""`` when the slot is free, else why not."""
    blocker = restart_blocker(info, config, pinned=pinned, mine=mine)
    return blocker or _drain(info, paths, token, mine, config.memory_daemon_startup_timeout_seconds)
