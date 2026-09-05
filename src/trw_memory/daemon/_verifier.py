"""The daemon's bearer-token verifier -- FR03 property 3, NFR03.

fastmcp 3.2.4 exports the ``TokenVerifier`` base and ships no static
implementation (``fastmcp.server.auth.providers.in_memory`` has an OAuth
provider and nothing simpler), so the per-user token needs this small subclass.

It is wired through ``FastMCP.auth``, which the streamable-HTTP app enforces as
transport middleware: a request with a missing or wrong token is rejected
before dispatch, so no tool body runs and no row is read or written.
"""

from __future__ import annotations

import structlog
from fastmcp.server.auth import AccessToken, TokenVerifier

from trw_memory.daemon._token import tokens_match

__all__ = ["LoopbackTokenVerifier"]

logger = structlog.get_logger(__name__)

#: Client identity recorded on an accepted token. The daemon serves exactly one
#: user account, so there is one principal; RBAC scoping is per NAMESPACE and is
#: enforced by the tool bodies, not by the transport.
LOOPBACK_CLIENT_ID = "trw-memory-loopback"


class LoopbackTokenVerifier(TokenVerifier):
    """Accept exactly one token, compared in constant time."""

    def __init__(self, expected_token: str) -> None:
        """Args: expected_token: the per-user token read from ``daemon-token``."""
        super().__init__()
        self._expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return an access token for a match, ``None`` for anything else.

        ``None`` -- not an exception -- is fastmcp's "reject" signal. The token
        is never logged, on either branch.
        """
        if not token or not tokens_match(token, self._expected_token):
            logger.warning("daemon_token_rejected")
            return None
        return AccessToken(token=token, client_id=LOOPBACK_CLIENT_ID, scopes=[])
