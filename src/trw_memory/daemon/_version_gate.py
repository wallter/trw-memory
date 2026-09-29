"""The daemon refuses a client of another major version by name (PLAN W45).

``DaemonClient`` checks the daemon's major before it calls (PRD-CORE-302 C7), but
a client from before that check -- trw-memory 3.x under trw-mcp 6.1.0 -- attaches
to whatever daemon is running and calls with its own tool signatures, which a
4.x daemon answers with a contract error. So the daemon checks too: a current
client sends its version in :data:`VERSION_HEADER`, and a tool call without it,
or from another major, is refused with the upgrade to make.

``memory_drain`` is the one call that crosses majors, and only one way: a client
of the same or a NEWER major may ask this daemon to retire (DAEMON-AUTO-RESTART-ON-UPGRADE,
hot reload); an older or unversioned caller is refused, so no downgrade retires it. The
drain key (a same-user secret) is the authority; this gate only prevents a downgrade.
"""

from __future__ import annotations

import os

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers, get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp import types as mt

from trw_memory.daemon._discovery import AGENT_MUST_NOT_STOP, VERSION_HEADER
from trw_memory.daemon._drain import DRAIN_TOOL
from trw_memory.daemon._versions import major

__all__ = ["VERSION_HEADER", "VersionGate"]


def _major(version: str) -> str:
    """The raw leading dotted token: strict on purpose, so ``"5rc1"``, ``"unknown"`` and ``""`` match no major."""
    return version.split(".", 1)[0]


class VersionGate(Middleware):
    """Refuse tool calls from a client whose trw-memory major differs from the daemon's."""

    def __init__(self, daemon_version: str) -> None:
        self._version = daemon_version

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            get_http_request()
        except RuntimeError:  # stdio or in-process: no daemon client to check
            return await call_next(context)
        theirs = get_http_headers().get(VERSION_HEADER, "")
        if context.message.name == DRAIN_TOOL:  # the one call across majors, never from an older one
            ours, newer = major(self._version), major(theirs)
            if ours is None or newer is None or newer < ours:
                raise ToolError(
                    f"drain_refused: this trw-memory daemon (pid {os.getpid()}) serves {self._version}; only a "
                    f"client of the same or a newer major version may drain it, and the caller is {theirs or 'unversioned'}."
                )
            return await call_next(context)
        if _major(theirs) != _major(self._version):
            client = f"is trw-memory {theirs}" if theirs else "did not say its version (trw-memory 3.x or older)"
            raise ToolError(
                f"daemon_version_mismatch: this trw-memory daemon (pid {os.getpid()}) serves {self._version}, but the "
                f"calling client {client}; their tool signatures differ, so nothing was read or written. The user "
                f"should upgrade the client (pip install -U trw-mcp trw-memory, then reconnect the MCP server) or stop "
                f"this daemon (process {os.getpid()}) so the client's own version starts one. {AGENT_MUST_NOT_STOP}"
            )
        return await call_next(context)
