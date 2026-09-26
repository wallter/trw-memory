"""The daemon's bearer-token verifier -- PRD-CORE-253 FR03, PRD-CORE-298 FR02.

It is wired through ``FastMCP.auth``, which the streamable-HTTP app enforces as
transport middleware: a request with a missing or unknown token is rejected
before dispatch, so no tool body runs and no row is read or written. A known
token becomes ``ns:<namespace>`` scopes, which ``require_namespace_permission``
checks on every namespaced call.
"""

from __future__ import annotations

import asyncio

import structlog
from fastmcp.server.auth import AccessToken, TokenVerifier

from trw_memory.daemon._grants import read_grant
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.exceptions import TokenUnreadableError

__all__ = ["LoopbackTokenVerifier"]

logger = structlog.get_logger(__name__)

#: Client identity recorded on an accepted token.
LOOPBACK_CLIENT_ID = "trw-memory-loopback"


class LoopbackTokenVerifier(TokenVerifier):
    """Accept a granted token, scoped to its namespaces."""

    def __init__(self, paths: DaemonPaths) -> None:
        """Args: paths: daemon files; the grants file is re-read per request, so new grants need no restart."""
        super().__init__()
        self._paths = paths

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return the token's grant as scopes, ``None`` (fastmcp's reject) for anything else.

        The token is never logged, on either branch.
        """
        try:
            grant = await asyncio.to_thread(read_grant, self._paths, token) if token else None  # file I/O, off the loop
        except TokenUnreadableError:  # trw-fail-silent-allow: None is fastmcp's logged 401
            logger.warning("daemon_grants_unreadable", path=str(self._paths.grants))
            return None
        if grant is None:
            logger.warning("daemon_token_rejected")
            return None
        scopes = [f"ns:{ns}" for ns in sorted(grant.namespaces)]
        claims = {"root": grant.root} if grant.root else {}
        return AccessToken(token=token, client_id=LOOPBACK_CLIENT_ID, scopes=scopes, claims=claims)
