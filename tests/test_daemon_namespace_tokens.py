"""PRD-CORE-298 FR02 -- a daemon token grants a fixed namespace set.

Before FR02 one per-user bearer authorised every namespace in the store, so
with RBAC off (the default) any checkout could store into, forget from or
recall out of another checkout's namespace. These tests hold the replacement:
a grants file keyed by token digest, a verifier that turns a match into
``ns:`` scopes, and one grant step inside ``require_namespace_permission``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from trw_memory.daemon import DaemonClient, DaemonPaths
from trw_memory.daemon._grants import (
    CHECKOUT_TOKEN_RELPATH,
    granted_namespaces,
    mint_grant,
    read_checkout_grant,
    write_checkout_grant,
)
from trw_memory.daemon._verifier import LoopbackTokenVerifier
from trw_memory.exceptions import AuthorizationError, DaemonAuthError, TokenUnreadableError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission, require_namespace_permission

from .test_daemon_server import _await_discovery, _call, _daemon_env, _spawn_daemon

pytest.importorskip("fastmcp")

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    return DaemonPaths.resolve()


async def test_the_verifier_returns_exactly_the_granted_namespaces(paths: DaemonPaths) -> None:
    token = mint_grant(paths, [_ALPHA, "user:local"])
    verifier = LoopbackTokenVerifier(paths)

    accepted = await verifier.verify_token(token)

    assert accepted is not None
    assert sorted(accepted.scopes) == [f"ns:{_ALPHA}", "ns:user:local"]
    assert await verifier.verify_token("not-a-granted-token") is None
    assert await verifier.verify_token("") is None
    assert paths.grants.stat().st_mode & 0o777 == 0o600
    assert token not in paths.grants.read_text(encoding="utf-8"), "the grants file holds digests, not tokens"


def test_a_second_grant_keeps_the_first(paths: DaemonPaths) -> None:
    first = mint_grant(paths, [_ALPHA])
    second = mint_grant(paths, [_BETA])

    assert granted_namespaces(paths, first) == frozenset({_ALPHA})
    assert granted_namespaces(paths, second) == frozenset({_BETA})


def test_minting_never_overwrites_an_unreadable_grants_file(paths: DaemonPaths) -> None:
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    paths.grants.write_text("not json{", encoding="utf-8")

    with pytest.raises(TokenUnreadableError, match="NOT"):
        mint_grant(paths, [_ALPHA])
    assert paths.grants.read_text(encoding="utf-8") == "not json{"


def test_a_checkout_grant_round_trips_at_0600_and_its_absence_names_the_verb(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    (checkout / "src").mkdir(parents=True)

    with pytest.raises(DaemonAuthError, match="trw-mcp memory token"):
        read_checkout_grant(checkout / "src")

    write_checkout_grant(checkout / ".trw", "t0ken")

    assert read_checkout_grant(checkout / "src") == "t0ken"
    assert (checkout / CHECKOUT_TOKEN_RELPATH).stat().st_mode & 0o777 == 0o600


def test_the_grant_step_runs_before_the_rbac_switch() -> None:
    """RBAC off is the default, and it must not reopen a namespace the token lacks."""
    from mcp.server.auth.provider import AccessToken

    config = MemoryConfig(rbac_enabled=False)
    granted = AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"])
    reset = auth_context_var.set(AuthenticatedUser(granted))
    try:
        require_namespace_permission(config, _ALPHA, Permission.WRITE, "store")
        with pytest.raises(AuthorizationError, match=_BETA):
            require_namespace_permission(config, _BETA, Permission.WRITE, "store")
    finally:
        auth_context_var.reset(reset)
    # No token -- the in-process SDK -- is unchanged.
    require_namespace_permission(config, _BETA, Permission.WRITE, "store")


async def test_a_token_cannot_reach_a_namespace_outside_its_grant(paths: DaemonPaths, tmp_path: Path) -> None:
    """End to end over the real transport, with RBAC off: store, recall, forget, get and update."""
    token = mint_grant(paths, [_ALPHA, "user:local"])
    proc = _spawn_daemon(tmp_path / "userhome")
    try:
        info = _await_discovery(paths, proc)
        stored = await _call(info, "memory_store", {"content": "alpha row", "namespace": _ALPHA}, token=token)
        assert isinstance(stored, dict) and stored["status"] == "stored"
        client = DaemonClient(token, paths=paths)
        assert (await client.update(stored["memory_id"], _ALPHA, {"impact": 0.9}))["status"] == "updated"
        assert (await client.get(stored["memory_id"], _ALPHA))["entry"]["importance"] == 0.9
        with pytest.raises(ToolError, match=_BETA):  # the refusal itself, never "unreachable" after a retry
            await client.get(stored["memory_id"], _BETA)

        for name, arguments in (
            ("memory_store", {"content": "must not land", "namespace": _BETA}),
            ("memory_recall", {"query": "row", "namespace": _BETA}),
            ("memory_forget", {"memory_id": "L-any", "namespace": _BETA}),
            ("memory_get", {"memory_id": "L-any", "namespace": _BETA}),
            ("memory_update", {"entry_id": "L-any", "namespace": _BETA, "patch": {"impact": 0.1}}),
        ):
            with pytest.raises(Exception, match=_BETA):
                await _call(info, name, arguments, token=token)
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_a_slice_a_token_file_stops_the_daemon_starting(paths: DaemonPaths, tmp_path: Path) -> None:
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    paths.token.write_text("an-all-namespace-bearer", encoding="utf-8")

    done = subprocess.run(
        [sys.executable, "-m", "trw_memory.server", "serve", "http"],
        env=_daemon_env(tmp_path / "userhome"),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode != 0
    assert "trw-mcp memory token --migrate" in done.stderr + done.stdout
    assert not paths.discovery.exists()


def test_no_module_mints_an_all_namespace_bearer() -> None:
    import trw_memory.daemon as daemon

    assert not hasattr(daemon, "ensure_token")
    assert asyncio.iscoroutinefunction(LoopbackTokenVerifier.verify_token)
