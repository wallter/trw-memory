"""The daemon refuses a client of another major version by name (PLAN W45).

``DaemonClient`` checks the daemon's major before it calls (PRD-CORE-302 C7), but
a client from before that check -- trw-memory 3.x under trw-mcp 6.1.0 -- attaches
to whatever daemon is running and calls with its own tool signatures, which a
4.x daemon answers with a contract error. So the daemon checks too: a current
client sends its version in :data:`VERSION_HEADER`, and a tool call without it,
or from another major, is refused with the upgrade to make.
"""

from __future__ import annotations

import os

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers, get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp import types as mt

__all__ = ["VERSION_HEADER", "VersionGate"]

#: The header a ``DaemonClient`` carries its trw-memory version in.
VERSION_HEADER = "x-trw-memory-version"


def _major(version: str) -> str:
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
        if _major(theirs) != _major(self._version):
            client = f"is trw-memory {theirs}" if theirs else "did not say its version (trw-memory 3.x or older)"
            raise ToolError(
                f"daemon_version_mismatch: this trw-memory daemon (pid {os.getpid()}) serves {self._version}, but the "
                f"calling client {client}; their tool signatures differ, so nothing was read or written. Upgrade the "
                f"client (pip install -U trw-mcp trw-memory, then reconnect the MCP server), or stop process "
                f"{os.getpid()} so the client's own version starts a daemon."
            )
        return await call_next(context)
