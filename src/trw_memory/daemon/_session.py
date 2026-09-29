"""How a daemon client reaches the daemon over MCP, and how it reads a failed request.

Split from ``client`` (PRD-CORE-333 S3b), which keeps the call policy: attach, retry,
replay and the fail-closed errors. Here: opening an MCP session (fastmcp loads only
when a call needs one; the direct reads in ``_direct`` never do) and classifying a
transport failure -- a 401 rejection, or a request that never left.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx

from trw_memory.daemon._discovery import VERSION_HEADER, DaemonInfo, refused_while_draining

if TYPE_CHECKING:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

#: HTTP status the daemon returns for a missing or wrong bearer token.
UNAUTHORIZED_STATUS = 401


def transport(info: DaemonInfo, token: str | None, version: str) -> StreamableHttpTransport:
    """*info*'s endpoint with *token* and the client's *version* (W45)."""
    from fastmcp.client.transports import StreamableHttpTransport

    return StreamableHttpTransport(url=info.url, auth=token, headers={VERSION_HEADER: version})


def open_session(info: DaemonInfo, token: str | None, version: str, **options: float) -> Client[Any]:
    """An MCP client over *info*'s endpoint; fastmcp loads only when a call needs a session."""
    from fastmcp import Client

    return Client(transport(info, token, version), **options)


def chain(exc: BaseException) -> Iterator[BaseException]:
    """*exc*, its causes and contexts, and the members of any exception group among them."""
    seen: set[int] = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        members = getattr(current, "exceptions", None)  # an exception group; Python 3.10 has no builtin
        if isinstance(members, tuple):
            pending.extend(member for member in members if isinstance(member, BaseException))


def is_unauthorized(exc: BaseException) -> bool:
    """Whether *exc* (or a cause in its chain) is a 401 rejection."""
    return any(
        getattr(getattr(current, "response", None), "status_code", None) == UNAUTHORIZED_STATUS
        for current in chain(exc)
    )


def never_sent(exc: BaseException) -> bool:
    """Whether the daemon never applied the request: the connect failed, or a draining daemon's door refused it."""
    return any(
        isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout)) or refused_while_draining(current)
        for current in chain(exc)
    )
