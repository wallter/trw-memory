"""Daemon reads sent as ONE JSON-RPC ``tools/call`` POST: no MCP session, no fastmcp (PRD-CORE-333 S3b).

The daemon serves stateless JSON (``_serve._build_app``), so an MCP session buys a read
nothing: a per-call session cost ``initialize``, ``notifications/initialized``, the call
and a ``tools/list``, and importing fastmcp cost about 0.3 s -- most of the
UserPromptSubmit hook's filtered read. :data:`DIRECT_TOOLS` holds only tools whose
result is a JSON object, which IS the ``structuredContent``; a wrapped result needs the
tool's output schema (a ``tools/list``) to unwrap. ``DaemonClient._call_once`` routes
them here after its usual discovery, version and instance checks.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx

from trw_memory.daemon._discovery import AGENT_MUST_NOT_STOP, VERSION_HEADER, DaemonInfo
from trw_memory.exceptions import DaemonProtocolError
from trw_memory.sync._remote_common import build_platform_headers

#: The reads a client sends as one POST: the prompt hook's filtered read (status, list_page) and
#: the edit hook's recall (status, anchored, recall). Every one is served as ``dict[str, object]``;
#: opening a session for recall and anchored cost the edit hook ~0.7 s of fastmcp import alone.
DIRECT_TOOLS = frozenset({"memory_status", "memory_list_page", "memory_anchored", "memory_recall"})

#: fastmcp's streamable-HTTP timeouts, so a direct read waits as a session call did.
_TIMEOUT = httpx.Timeout(30.0, read=300.0)


def answered(exc: BaseException) -> bool:
    """Whether *exc* is the daemon's refusal (fastmcp's ``ToolError``), without importing fastmcp.

    A process that never imported ``fastmcp.exceptions`` cannot hold one of its errors.
    """
    errors = sys.modules.get("fastmcp.exceptions")
    return errors is not None and isinstance(exc, errors.ToolError)


def _refused(message: str) -> Exception:
    """The daemon's refusal as the ``ToolError`` a session call raises (the error path pays the import)."""
    from fastmcp.exceptions import ToolError

    return ToolError(message)


async def post_tool(info: DaemonInfo, token: str, version: str, name: str, arguments: dict[str, Any]) -> Any:
    """Call *name* on *info*'s endpoint with one POST; its ``structuredContent`` object.

    The request carries what a session call carries -- the grant as a bearer token and
    the client's *version* (W45) -- so the daemon's auth, grant scope and version gate
    judge it the same. A 401 surfaces as ``httpx.HTTPStatusError`` and a connect failure
    as ``httpx.ConnectError``: ``DaemonClient.call_tool`` classifies both as it does a
    session's. A refusal (``isError``, or a JSON-RPC ``error``) is a ``ToolError``; a
    reply that is neither a refusal nor an object answer is a :class:`DaemonProtocolError`.
    """
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    # The bearer comes from the one sanctioned builder (the census in test_platform_trust*), which
    # attaches it only to a trusted or loopback host: the daemon's record is loopback-only (FIX-157).
    headers = {
        **build_platform_headers(token, info.url),
        "Accept": "application/json, text/event-stream",
        VERSION_HEADER: version,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
        response = await http.post(info.url, json=request, headers=headers)
    response.raise_for_status()
    return _answer(name, response)


def _answer(name: str, response: httpx.Response) -> dict[str, Any]:
    """The object *response* answers *name* with; the refusal it carries raised; anything else refused."""
    try:
        reply = response.json()
    except ValueError:
        raise _malformed(name, "a body that is not JSON") from None
    if not isinstance(reply, dict) or reply.get("jsonrpc") != "2.0" or reply.get("id") != 1:
        raise _malformed(name, "a JSON-RPC envelope that does not answer this request")
    if ("result" in reply) == ("error" in reply):
        raise _malformed(name, "an envelope with neither or both of result and error")
    if "error" in reply:
        error = reply["error"]
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            raise _malformed(name, "an error without a message")
        raise _refused(error["message"])
    result = reply["result"]
    if not isinstance(result, dict) or result.get("isError", False) not in (True, False):
        raise _malformed(name, "a result that is not a tool result")
    if result.get("isError"):
        raise _refused(_refusal_text(result.get("content")) or f"{name} refused")
    content = result.get("structuredContent")
    if not isinstance(content, dict):
        raise _malformed(name, f"structuredContent of type {type(content).__name__}, not an object")
    return content


def _refusal_text(content: object) -> str:
    """The text of a refusal's ``content``, whatever shape it arrived in: a block list, one block or a string."""
    if isinstance(content, str):
        return content
    blocks = content if isinstance(content, list) else [content]
    return "\n".join(
        block["text"] for block in blocks if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _malformed(name: str, what: str) -> DaemonProtocolError:
    return DaemonProtocolError(
        f"the trw-memory daemon answered {name} with {what}; nothing was read. The user can run "
        f"`trw-mcp doctor`, and restart the daemon if it keeps answering this way. {AGENT_MUST_NOT_STOP}"
    )
