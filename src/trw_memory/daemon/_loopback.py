"""The loopback bind boundary -- PRD-CORE-253 FR03 property 1, NFR03.

The daemon's bind host is the module constant :data:`LOOPBACK_HOST`. It is not
a ``MemoryConfig`` field and it is not a CLI flag, because a configurable host
turns one typo into a network-reachable memory store fronted by a bearer token.
The port is tunable; the host is not.

:func:`require_loopback` is the guard that makes that a property rather than a
convention: every address this module binds passes through it, and a
non-loopback address raises *before* a socket is created, so the rejected
address never reaches ``bind``. Containers reach the daemon by sharing the host
network namespace (OQ-2), not by relaxing this.
"""

from __future__ import annotations

import ipaddress
import socket

import structlog

from trw_memory.exceptions import ConfigError

__all__ = ["LOOPBACK_HOST", "bind_loopback_socket", "endpoint_url", "require_loopback"]

logger = structlog.get_logger(__name__)

#: The only address the daemon ever binds.
LOOPBACK_HOST = "127.0.0.1"

#: Path segment of the streamable-HTTP endpoint, matching fastmcp's default.
MCP_PATH = "/mcp"

#: Backlog for the listening socket. One daemon serves a handful of local
#: clients; the queue only has to absorb a simultaneous reconnect burst.
_LISTEN_BACKLOG = 64


def require_loopback(host: str) -> str:
    """Return *host* if it is a loopback address, else raise.

    Args:
        host: Candidate bind address.

    Returns:
        The validated host, unchanged.

    Raises:
        ConfigError: If *host* is not a parseable loopback IP address. The
            message names the rejected address so an operator sees which value
            was refused.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ConfigError(
            f"refusing to bind the memory daemon to {host!r}: not an IP address. The daemon binds {LOOPBACK_HOST} only."
        ) from exc
    if not address.is_loopback:
        raise ConfigError(
            f"refusing to bind the memory daemon to {host!r}: not a loopback address. "
            f"The daemon binds {LOOPBACK_HOST} only; a container reaches it by sharing "
            "the host network namespace."
        )
    return host


def bind_loopback_socket(port: int, *, host: str = LOOPBACK_HOST) -> socket.socket:
    """Bind and listen on a loopback socket, returning it unserved.

    Binding here rather than inside uvicorn is what makes an ephemeral port
    usable: the assigned port is readable from the socket before the server
    starts, so the discovery file can advertise it and a client never has to
    guess.

    Args:
        port: TCP port, or 0 to let the operating system assign one.
        host: Bind address. Defaults to -- and is validated against -- the
            loopback constant.

    Returns:
        A listening socket. The caller owns it and must close it.

    Raises:
        ConfigError: If *host* is not a loopback address; no socket is created.
    """
    require_loopback(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_LISTEN_BACKLOG)
    except BaseException:
        sock.close()
        raise
    logger.info("daemon_socket_bound", host=host, port=sock.getsockname()[1])
    return sock


def endpoint_url(sock: socket.socket) -> str:
    """Return the MCP endpoint URL a bound socket serves."""
    host, port = sock.getsockname()[:2]
    return f"http://{host}:{port}{MCP_PATH}"
