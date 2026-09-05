"""One loopback daemon in front of the user-space memory store.

PRD-CORE-253 FR03 and FR08. ``trw-memory-server serve http`` runs a
streamable-HTTP MCP endpoint on 127.0.0.1 with a per-user bearer token, an
ephemeral port published through a 0600 discovery file, a single-instance
claim, and an idle shutdown. :class:`DaemonClient` attaches to it and fails
closed -- for reads as well as writes -- when it cannot.

The narrow interface is everything below; the modules behind it
(``_loopback``, ``_instance``, ``_token``, ``_discovery``, ``_serve``) are
implementation and may be reshaped freely.
"""

from __future__ import annotations

from trw_memory.daemon._discovery import (
    DaemonInfo,
    DiscoveryAbsent,
    DiscoveryInvalid,
    DiscoveryRead,
    read_discovery,
    read_discovery_result,
    read_live_discovery,
)
from trw_memory.daemon._instance import claim_single_instance, release_single_instance
from trw_memory.daemon._loopback import LOOPBACK_HOST, bind_loopback_socket, require_loopback
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._token import ensure_token, read_token, tokens_match

__all__ = [
    "DAEMON_START_COMMAND",
    "LOOPBACK_HOST",
    "DaemonClient",
    "DaemonInfo",
    "DaemonPaths",
    "DaemonServeOptions",
    "DiscoveryAbsent",
    "DiscoveryInvalid",
    "DiscoveryRead",
    "bind_loopback_socket",
    "claim_single_instance",
    "ensure_token",
    "read_discovery",
    "read_discovery_result",
    "read_live_discovery",
    "read_token",
    "release_single_instance",
    "require_loopback",
    "serve_loopback",
    "start_daemon_detached",
    "tokens_match",
]


def __getattr__(name: str) -> object:
    """Defer the fastmcp-dependent surface until it is actually asked for.

    ``fastmcp`` is the optional ``[mcp]`` extra, and ``uvicorn`` arrives with
    it. Importing this package for a path resolution, a token or a discovery
    read must not require either, so the serving loop and the client are loaded
    on first attribute access instead of at import time.
    """
    if name in ("DaemonServeOptions", "serve_loopback"):
        from trw_memory.daemon import _serve

        return getattr(_serve, name)
    if name in ("DAEMON_START_COMMAND", "DaemonClient", "start_daemon_detached"):
        from trw_memory.daemon import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
