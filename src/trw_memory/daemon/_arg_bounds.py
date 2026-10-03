"""Every served tool's arguments are bounded before the tool runs (rc9 sweep round 4).

The daemon serves every tenant from one process, so an argument a tool walks --
a list of ids, an acks map, a text it encodes -- is work a caller can make
unbounded, and six such findings had each been fixed (or missed) at its own call
site. :class:`ArgumentBounds` checks them in one place, as FastMCP middleware:
it sees a call's raw arguments before validation, any lane or pool, or the tool
body, and refuses anything past its bound with ``{"status": "invalid", "error":
"argument_too_large", "argument": ..., "limit": ...}``.

:func:`bound` gives the length limit of an argument (characters for a string,
items for a list or dict): the tool's own entry in :data:`OVERRIDES`, else the
entry for the argument's name in :data:`ARGUMENTS`. ``tests/test_arg_bounds.py``
fails when a served tool has a ``str``, ``list`` or ``dict`` argument with
neither, so a new tool cannot ship without deciding its bounds. Values nested
inside an argument take :data:`TEXT` and :data:`ITEMS` (dict keys :data:`NAME`),
and the whole request is capped at :data:`MAX_REQUEST_VALUES` values and
:data:`MAX_REQUEST_CHARS` characters. The walk is iterative and stops at the
first value past a bound, so it costs at most those totals.
"""

from __future__ import annotations

import json
from typing import Final

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp import types as mt
from starlette.requests import ClientDisconnect
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from trw_memory._sweep import MAX_TOKEN_CHARS
from trw_memory.models.memory import MAX_TEXT_FIELD_CHARS
from trw_memory.tools.checkout_import import IMPORT_MAX_IDS
from trw_memory.tools.recall_support import SURFACED_MAX
from trw_memory.tools.similar import MAX_SIMILAR_TEXT_CHARS
from trw_memory.tools.sync import MAX_SYNC_APPLY_MANY, MAX_SYNC_DIRTY_PAGE

__all__ = ["ARGUMENTS", "OVERRIDES", "ArgumentBounds", "bound", "call_with_body_cap"]

#: A name or path: a namespace (itself capped at 128), an entry id, an actor, a file path (PATH_MAX).
NAME: Final = 4_096
#: A text a tool stores, encodes or matches. ``max_entry_chars`` is capped to this, similar refuses
#: past 8,000 characters and recall cuts its query to 1,000. Same value the local store/update path
#: refuses past (PRD-CORE-331 FR07 / B71-94) -- imported, not restated, so the two cannot drift.
TEXT: Final = MAX_TEXT_FIELD_CHARS
#: The items of a list or the fields of an object: the largest page any daemon tool serves.
ITEMS: Final = 1_000
#: The whole request, counted as the walk goes: every value, and every character of every string.
#: Room for ``memory_import_checkout``'s 100,000 ids of up to 64 characters.
MAX_REQUEST_VALUES: Final = 200_000
MAX_REQUEST_CHARS: Final = 8 * 1024 * 1024
#: An HTTP request body, cut before it is parsed: room for the totals above with JSON escapes.
MAX_BODY_BYTES: Final = 16 * 1024 * 1024

_NAMES: tuple[str, ...] = ("namespace", "source", "destination", "memory_id", "learning_id", "entry_id")
_NAMES += ("remote_id", "actor", "status", "sort_by", "decision", "source_identity", "session_id", "expires")
_NAMES += ("source_path", "project_root", "if_revision", "file")
_ITEMS: tuple[str, ...] = ("ids", "remote_ids", "tags", "edge_types", "evidence", "assertions", "include_namespaces")
_ITEMS += ("include_source_kinds", "exclude_source_kinds", "after", "metadata", "learning", "entry", "patch")
_ITEMS += ("consolidation", "settings", "types")

#: The bound of an argument by its name, in every tool that takes it.
ARGUMENTS: Final[dict[str, int]] = {
    **dict.fromkeys(_NAMES, NAME),
    **dict.fromkeys(("content", "detail", "query", "text"), TEXT),
    **dict.fromkeys(_ITEMS, ITEMS),
    "cursor": MAX_TOKEN_CHARS,
}

#: Where a tool's argument needs another bound than its name's.
OVERRIDES: Final[dict[str, dict[str, int]]] = {
    "memory_import_checkout": {"ids": IMPORT_MAX_IDS},
    "memory_record_surfaced": {"ids": SURFACED_MAX},
    "memory_sync_mark_synced": {"acks": MAX_SYNC_DIRTY_PAGE},
    "memory_sync_apply_many": {"items": MAX_SYNC_APPLY_MANY},
    "memory_similar": {"text": MAX_SIMILAR_TEXT_CHARS},
    "memory_drain": {"admin_key": 128},  # a 64-hex drain key; anything longer is not one
}


def bound(tool: str, argument: str) -> int | None:
    """The length limit of *tool*'s *argument*; ``None`` if it has none (the census test's failure)."""
    return OVERRIDES.get(tool, {}).get(argument, ARGUMENTS.get(argument))


async def call_with_body_cap(app: ASGIApp, scope: Scope, receive: Receive, send: Send) -> None:
    """Run *app* on an HTTP request whose body may not pass :data:`MAX_BODY_BYTES`, buffering nothing.

    A declared ``content-length`` past it is answered 413 before any of the body is read (or the
    token checked). An undeclared (chunked) body is counted as the app reads it, after its own
    checks, and is cut with an ``http.disconnect`` at the cap. Starlette's body reader
    (``Request.stream()``) raises that as ``ClientDisconnect``; FastMCP's own request handler
    catches it and answers with ITS OWN plain 500, never re-raising, so we cannot simply catch
    the exception here -- by the time it would reach us it has already been turned into a sent
    response. Instead ``watching_send`` intercepts *app*'s outgoing ``http.response.start``: if
    the cut was ours (``over_cap``) and *app* has not yet sent a byte of its own response, we
    swap that 500 for a 413 naming the limit before anything reaches the real transport. If *app*
    had already started responding before the cap tripped, headers are already committed and no
    new status can be sent -- we let that response finish as-is (``started``). A handler that
    (unlike FastMCP's) leaves ``ClientDisconnect`` unhandled is still covered by the ``except``
    below, for the same two cases.
    """
    declared = dict(scope.get("headers") or ()).get(b"content-length", b"")
    if declared.isdigit() and (len(declared) > 9 or int(declared) > MAX_BODY_BYTES):  # no huge int() before auth
        await PlainTextResponse(f"request body over {MAX_BODY_BYTES} bytes", status_code=413)(scope, receive, send)
        return
    seen = 0
    over_cap = False
    started = False
    rewritten = False
    rewritten_body_sent = False

    async def capped() -> Message:
        nonlocal seen, over_cap
        message = await receive()
        seen += len(message.get("body", b""))
        if seen <= MAX_BODY_BYTES:
            return message
        over_cap = True
        return {"type": "http.disconnect"}

    async def watching_send(message: Message) -> None:
        nonlocal started, rewritten, rewritten_body_sent
        kind = message.get("type")
        if kind == "http.response.start":
            if over_cap and not started:
                rewritten = True
                await send(
                    {"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"text/plain")]}
                )
            else:
                await send(message)
            started = True
            return
        if kind == "http.response.body" and rewritten:
            if not rewritten_body_sent:
                rewritten_body_sent = True
                body = f"request body over {MAX_BODY_BYTES} bytes".encode()
                await send({"type": "http.response.body", "body": body, "more_body": False})
            return
        await send(message)

    try:
        await app(scope, capped, watching_send)
    except ClientDisconnect:
        if not over_cap:
            raise  # a real client disconnect, not ours to reinterpret
        if started:
            return  # headers already sent; too late for a new status, just close
        await PlainTextResponse(f"request body over {MAX_BODY_BYTES} bytes", status_code=413)(scope, receive, send)


def _oversized(tool: str, arguments: dict[str, object]) -> tuple[str, int] | None:
    """The first argument past its bound and that bound (``"(request)"`` for the totals); ``None`` if all fit."""
    stack: list[tuple[str, object, int | None]] = [(k, v, bound(tool, k)) for k, v in arguments.items()]
    values = chars = 0
    while stack:
        name, value, limit = stack.pop()
        values += 1
        if isinstance(value, str):
            chars += len(value)
            if len(value) > (limit or TEXT):
                return name, limit or TEXT
        elif isinstance(value, (list, dict)):
            if len(value) > (limit or ITEMS):
                return name, limit or ITEMS
            stack.extend((name, item, None) for item in (value.values() if isinstance(value, dict) else value))
            stack.extend((name, key, NAME) for key in (value if isinstance(value, dict) else ()))
        if values > MAX_REQUEST_VALUES or chars > MAX_REQUEST_CHARS:
            return "(request)", MAX_REQUEST_VALUES if values > MAX_REQUEST_VALUES else MAX_REQUEST_CHARS
    return None


class ArgumentBounds(Middleware):
    """Refuse a tool call whose arguments are past their bounds before the tool runs."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = context.message.name
        if (over := _oversized(name, dict(context.message.arguments or {}))) is None:
            return await call_next(context)
        refusal: dict[str, object] = {"status": "invalid", "error": "argument_too_large", "argument": over[0]}
        refusal["limit"] = over[1]
        tool = await context.fastmcp_context.fastmcp.get_tool(name) if context.fastmcp_context else None
        if tool is not None and (tool.output_schema or {}).get("x-fastmcp-wrap-result"):
            refusal = {"result": refusal}  # memory_update's union return is wrapped on the wire
        return ToolResult(content=[mt.TextContent(type="text", text=json.dumps(refusal))], structured_content=refusal)
