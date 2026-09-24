"""Per-checkout namespace grants for the loopback daemon -- PRD-CORE-298 FR02.

A token authorises a fixed set of namespaces, not the whole store. The daemon
keeps ``daemon-grants.json`` (0600, in the 0700 daemon directory) mapping each
token's sha256 digest to its namespaces; the raw token lives only in the
minting checkout's ``.trw/runtime/memory-token`` (0600). The verifier turns a
match into ``ns:<namespace>`` scopes, and ``require_namespace_permission``
refuses any namespace outside them.

The Slice A rules carry over: files are written atomically through the
hardened 0600 path, a grants file that exists but cannot be read is never
minted over (that would revoke every other checkout's grant), and digests are
compared in constant time.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, cast

import structlog

from trw_memory.daemon._paths import DaemonPaths, read_secret_file, write_secret_file
from trw_memory.exceptions import DaemonAuthError, DaemonSecretUnreadableError, TokenUnreadableError
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.persistence import lock_for_rmw

__all__ = [
    "CHECKOUT_TOKEN_RELPATH",
    "Grant",
    "granted_namespaces",
    "mint_grant",
    "read_checkout_grant",
    "read_checkout_pin",
    "read_grant",
    "write_checkout_grant",
]

logger = structlog.get_logger(__name__)

#: ``token_urlsafe(32)``: 256 bits in 43 URL-safe characters (PRD-CORE-253 NFR03).
_TOKEN_BYTES = 32
#: A migrated checkout's config, holding the ``project_namespace`` pin.
CHECKOUT_CONFIG_RELPATH = Path(".trw") / "config.yaml"
#: Where a checkout keeps its raw token, relative to the checkout root.
CHECKOUT_TOKEN_RELPATH = Path(".trw") / "runtime" / "memory-token"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Grant(NamedTuple):
    """One token's grant: its namespaces, and the checkout root it was minted for (``None``: no checkout)."""

    namespaces: frozenset[str]
    root: str | None


def _read_grants(paths: DaemonPaths) -> dict[str, Grant]:
    """The grants map; empty when the file is absent, an error when it is unreadable."""
    try:
        raw = read_secret_file(paths.grants)
    except DaemonSecretUnreadableError as exc:
        raise _unreadable(paths, str(exc)) from exc
    if raw is None:
        return {}
    try:
        grants = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _unreadable(paths, f"{paths.grants} is not JSON") from exc
    if not isinstance(grants, dict) or not all(_well_formed(grant) for grant in grants.values()):
        raise _unreadable(paths, f"{paths.grants} is not a digest-to-grant map")
    return {digest: _grant(g) for digest, g in grants.items()}


def _grant(raw: list[str] | dict[str, object]) -> Grant:
    # A bare list is a grant minted before grants recorded a checkout: rootless, so file tools refuse it.
    if isinstance(raw, list):
        return Grant(frozenset(raw), None)
    return Grant(frozenset(cast("list[str]", raw["namespaces"])), cast("str | None", raw.get("root")))


def _well_formed(grant: object) -> bool:
    if isinstance(grant, list):
        return all(isinstance(n, str) for n in grant)
    if not isinstance(grant, dict):
        return False
    namespaces, root = grant.get("namespaces"), grant.get("root")
    return (
        isinstance(namespaces, list)
        and all(isinstance(n, str) for n in namespaces)
        and (root is None or isinstance(root, str))
    )


def _unreadable(paths: DaemonPaths, detail: str) -> TokenUnreadableError:
    return TokenUnreadableError(
        f"{detail}. No grant was minted and the file was NOT rewritten -- replacing it would revoke "
        f"every checkout's grant. Inspect or remove {paths.grants}, then run `trw-mcp memory token` "
        f"in each checkout."
    )


def mint_grant(paths: DaemonPaths, namespaces: Iterable[str], *, root: Path | None = None) -> str:
    """Record a new token for exactly *namespaces* and return the raw token.

    Callers decide WHICH namespaces; ``trw-mcp memory token`` passes only the
    caller's own pinned project namespace plus ``user:local``, and its checkout
    as *root* -- the only tree a file-reading tool may reach with the token.
    """
    granted = sorted({validate_namespace(namespace) for namespace in namespaces})
    if not granted:
        raise ValueError("a grant needs at least one namespace")
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    with lock_for_rmw(paths.lock_anchor):
        grants = {digest: grant._asdict() for digest, grant in _read_grants(paths).items()}
        grants[_digest(token)] = {"namespaces": granted, "root": str(root.resolve()) if root else None}
        write_secret_file(paths.grants, json.dumps(grants, sort_keys=True, default=sorted))
    logger.info("daemon_grant_minted", namespaces=granted)
    return token


def read_grant(paths: DaemonPaths, token: str) -> Grant | None:
    """*token*'s grant, or ``None`` for an unknown token."""
    digest = _digest(token)
    for known, grant in _read_grants(paths).items():
        if hmac.compare_digest(known, digest):
            return grant
    return None


def granted_namespaces(paths: DaemonPaths, token: str) -> frozenset[str] | None:
    """The namespaces *token* was granted, or ``None`` for an unknown token."""
    grant = read_grant(paths, token)
    return grant.namespaces if grant else None


def write_checkout_grant(trw_dir: Path, token: str) -> Path:
    """Store the raw token in this checkout's ``.trw/runtime`` at 0600."""
    path = trw_dir.parent / CHECKOUT_TOKEN_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    write_secret_file(path, token)
    return path


def read_checkout_grant(start: Path) -> str:
    """The token of the checkout enclosing *start*; a missing one names the verb that mints it."""
    for directory in (start, *start.parents):
        candidate = directory / CHECKOUT_TOKEN_RELPATH
        try:
            raw = read_secret_file(candidate)
        except DaemonSecretUnreadableError as exc:
            raise DaemonAuthError(f"{candidate} cannot be read ({exc}); run `trw-mcp memory token`") from exc
        if raw is not None and raw.strip():
            return raw.strip()
    raise DaemonAuthError(
        f"no memory grant for {start}: run `trw-mcp memory token` in this checkout to mint one "
        f"for its project namespace"
    )


def read_checkout_pin(start: Path) -> str | None:
    """The ``project_namespace`` pinned by the checkout enclosing *start*; ``None`` when unpinned.

    The pin, not the identity derived from the checkout's current path, names
    its rows: a moved checkout keeps the namespace ``memory migrate`` pinned.
    The pin only routes. It never widens a grant, so an edited pin reaches
    nothing its token was not granted.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    for directory in (start, *start.parents):
        candidate = directory / CHECKOUT_CONFIG_RELPATH
        if not candidate.is_file():
            continue
        try:
            config = YAML(typ="safe").load(candidate.read_text(encoding="utf-8"))
        except (OSError, YAMLError) as exc:
            raise DaemonAuthError(f"{candidate} cannot be read ({exc}); fix it or pass --namespace") from exc
        pinned = config.get("project_namespace") if isinstance(config, dict) else None
        return validate_namespace(pinned) if isinstance(pinned, str) and pinned else None
    return None
