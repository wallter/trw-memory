"""The drain handshake: a daemon retires itself for a same-or-newer-major client -- DAEMON-AUTO-RESTART-ON-UPGRADE.

After a major upgrade the old daemon refuses the new client, and the only remedy
used to be the user stopping it by hand. ``memory_drain`` lets the daemon do it
itself, and only safely: it stops accepting calls (a new request is answered 503
at the door), waits for every call already in flight to finish, then asks the
server to exit, and the serving loop's normal shutdown withdraws the discovery
record after the store is released. When the deadline passes with another
session's call still in flight, it resumes serving and says so (``busy``): a
restart is never worth another session's write (CONSTITUTION HB-2).

The daemon advertises the handshake in its record (``capabilities``). A drain
must present the daemon's ``admin_key`` (:mod:`._drain_key`), which a checkout's
namespace grant never carries; the version header is a compatibility signal
only: the :class:`~trw_memory.daemon._version_gate.VersionGate` also refuses a
caller that does not claim a NEWER major, so an older client never retires it.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Protocol

import structlog
from fastmcp.exceptions import ToolError

from trw_memory.daemon._discovery import DRAIN_TOOL
from trw_memory.daemon._drain_key import keys_match, remove_drain_key
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.tools._types import McpServer

__all__ = ["DRAIN_TOOL", "DrainRefusedError", "arm_drain", "disarm_drain", "drain", "register_drain_tool"]

logger = structlog.get_logger(__name__)

#: The longest a drain waits for in-flight calls, whatever the caller asks.
MAX_DRAIN_SECONDS = 60.0
_POLL_SECONDS = 0.05


class _Door(Protocol):
    """The part of ``_serve._IdleTracker`` a drain uses."""

    in_flight: int
    draining: bool


class _Exit(Protocol):
    """The part of ``uvicorn.Server`` a drain uses."""

    should_exit: bool


class DrainRefusedError(ToolError):
    """``memory_drain`` without the daemon's key: nothing was changed."""


@dataclass(frozen=True)
class _Armed:
    door: _Door
    server: _Exit
    key: str | None
    paths: DaemonPaths | None


#: The serving daemon's door and server; ``None`` outside ``serve_loopback`` (stdio, tests in-process).
_armed: _Armed | None = None


def arm_drain(door: _Door, server: _Exit, key: str | None, paths: DaemonPaths | None) -> None:
    """Make ``memory_drain`` act on this daemon's *door* and *server*, for a caller presenting *key*."""
    global _armed
    _armed = _Armed(door, server, key, paths)


def disarm_drain() -> None:
    """Forget the daemon a drain acts on (its serving loop has ended)."""
    global _armed
    _armed = None


async def drain(deadline_seconds: float, admin_key: str = "") -> dict[str, object]:
    """Stop taking calls, wait up to *deadline_seconds* for the ones in flight, then exit; what was done.

    Raises:
        DrainRefusedError: *admin_key* is not the daemon's drain key; nothing was changed.
    """
    armed = _armed
    if armed is None or armed.key is None:
        return {"status": "unsupported", "detail": "this process is not a loopback daemon; nothing was drained"}
    if not keys_match(admin_key, armed.key):
        logger.warning("daemon_drain_refused", reason="admin_key wrong" if admin_key else "admin_key missing")
        raise DrainRefusedError(
            "drain_refused: memory_drain needs this daemon's admin_key, which only a process of its OS user can "
            "read from its drain.key file; a checkout grant does not authorise it. Nothing was changed."
        )
    pid = os.getpid()
    if armed.door.draining:
        return {"status": "draining", "pid": pid, "detail": "a drain is already under way"}
    armed.door.draining = True
    started = time.monotonic()
    deadline = started + min(max(deadline_seconds, 0.0), MAX_DRAIN_SECONDS)
    while time.monotonic() < deadline:  # polled: the door counts requests, it signals nothing
        if armed.door.in_flight <= 1:  # 1: this drain's own request
            break
        await asyncio.sleep(_POLL_SECONDS)
    others, waited = armed.door.in_flight - 1, round(time.monotonic() - started, 3)
    if others > 0:
        armed.door.draining = False
        logger.info("daemon_drain_busy", in_flight=others, waited_seconds=waited)
        return {
            "status": "busy",
            "pid": pid,
            "in_flight": others,
            "waited_seconds": waited,
            "detail": f"{others} call(s) from other sessions were still in flight; the daemon resumed serving",
        }
    if armed.paths is not None:
        remove_drain_key(armed.paths, armed.key)
    armed.server.should_exit = True
    logger.info("daemon_drained", waited_seconds=waited)
    return {
        "status": "drained",
        "pid": pid,
        "waited_seconds": waited,
        "detail": "no other call in flight; the daemon exits and withdraws its discovery record",
    }


def register_drain_tool(server: McpServer) -> None:
    """Register ``memory_drain``. It runs on the event loop, never the lane, since it waits for the lane's calls."""

    @server.tool()
    async def memory_drain(deadline_seconds: float = 10.0, admin_key: str = "") -> dict[str, object]:
        """Retire this daemon for a same-or-newer-major client: finish in-flight calls, withdraw the record, exit.

        Returns ``drained`` (exiting), ``busy`` (a call outlived the deadline, so it kept serving),
        ``draining`` (already under way) or ``unsupported`` (not a loopback daemon).
        """
        return await drain(deadline_seconds, admin_key)
