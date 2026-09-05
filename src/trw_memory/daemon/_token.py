"""Per-user bearer token for the loopback daemon -- PRD-CORE-253 FR03/FR08.

One 32-byte URL-safe token per user, at mode 0600 in a 0700 directory. It is
created on first need, by whichever of the daemon or a client gets there first,
under the same advisory lock the single-instance claim uses -- so two processes
racing on a fresh install converge on one token rather than each writing its
own.

It is **never** regenerated on rejection (FR08 clause 3). Automatic rotation on
a rejected token would let any local process force a rotation by corrupting the
file, and would silently paper over a daemon started under a different account.
A rejection is an operator-facing error naming the file to remove.

The same reasoning binds the CREATE path, which is the one that used to leak:
generation happens only when the file is genuinely absent. A token file that
exists but cannot be read -- a planted symlink, a permission change, non-UTF-8
bytes, a truncation -- is a :class:`~trw_memory.exceptions.TokenUnreadableError`,
never an occasion to mint one over it. Replacing it is the forbidden rotation
arrived at from the other direction: the live daemon keeps the old secret and
starts rejecting every client.
"""

from __future__ import annotations

import hmac
import secrets

import structlog

from trw_memory.daemon._paths import DaemonPaths, read_secret_file, write_secret_file
from trw_memory.exceptions import DaemonSecretUnreadableError, TokenUnreadableError
from trw_memory.storage.persistence import lock_for_rmw

__all__ = ["TOKEN_BYTES", "ensure_token", "read_token", "tokens_match"]

logger = structlog.get_logger(__name__)

#: Entropy of the bearer token. ``token_urlsafe(32)`` yields 256 bits in 43
#: URL-safe characters, matching the NFR03 requirement of 32 bytes.
TOKEN_BYTES = 32


def read_token(paths: DaemonPaths) -> str | None:
    """Return the stored token, or ``None`` when the file does not exist.

    Raises:
        TokenUnreadableError: The token file exists but could not be read, or
            holds no token. The caller must NOT mint a replacement.
    """
    try:
        raw = read_secret_file(paths.token)
    except DaemonSecretUnreadableError as exc:
        raise _unreadable(paths, str(exc)) from exc
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        # An empty file is not first run. ``write_secret_file`` is atomic, so
        # emptiness is never a partial write -- it is a truncation someone
        # else performed, and minting over it rotates a live daemon's token.
        raise _unreadable(paths, f"{paths.token} exists but contains no token")
    return token


def _unreadable(paths: DaemonPaths, detail: str) -> TokenUnreadableError:
    """Build the refusal, carrying the remedy the operator has to apply."""
    return TokenUnreadableError(
        f"{detail}. The token was NOT regenerated -- doing so would rotate the secret a running "
        f"daemon is still authenticating against and reject every client. Inspect or remove "
        f"{paths.token} (and {paths.discovery}, which carries a copy), then retry."
    )


def ensure_token(paths: DaemonPaths) -> str:
    """Return the per-user token, generating it at 0600 if it is missing.

    A MISSING token file is first run, not an error (FR08 clause 2). A token
    file that exists but cannot be read is the opposite: minting a replacement
    would ``os.replace`` the secret a running daemon is authenticating against,
    and every client would then be rejected -- exactly the automatic rotation
    clause 3 forbids. Only ``FileNotFoundError`` reaches the generator.

    The read-generate-write cycle runs under the daemon lock so two processes
    starting at once cannot each mint a token and disagree about which is
    current.

    Returns:
        The token string, freshly generated or previously stored.

    Raises:
        TokenUnreadableError: The token file exists but could not be read.
            Nothing was written.
    """
    existing = read_token(paths)
    if existing is not None:
        return existing
    with lock_for_rmw(paths.lock_anchor):
        # Re-read inside the lock: another process may have won the race
        # between the unlocked probe above and the lock being granted.
        existing = read_token(paths)
        if existing is not None:
            return existing
        token = secrets.token_urlsafe(TOKEN_BYTES)
        write_secret_file(paths.token, token)
        logger.info("daemon_token_generated", path=str(paths.token))
        return token


def tokens_match(presented: str, expected: str) -> bool:
    """Compare two tokens in constant time (NFR03).

    ``hmac.compare_digest`` keeps the comparison independent of how many
    leading characters match, so a local attacker cannot recover the token one
    byte at a time from response timing.
    """
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
