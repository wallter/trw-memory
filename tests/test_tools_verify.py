"""memory_verify runs against the granted checkout's own files, never a root the caller names.

A grant records the checkout it was minted for; the verifier carries that root as
a claim, and the tool refuses any other ``project_root`` before it opens a backend,
so a write grant cannot aim the sweep's file checks at arbitrary paths (codex
286bcc090 P1). The sweep's knobs are type- and range-checked before it runs (P2).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from trw_memory.daemon import DaemonPaths, mint_grant
from trw_memory.daemon._verifier import LoopbackTokenVerifier
from trw_memory.lifecycle.verification_pass import MaintainVerifySummary, VerifySettings

_ALPHA = "project:alpha-11111111"


class _Captured:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self) -> object:
        return lambda fn: self.tools.setdefault(fn.__name__, fn)


@pytest.fixture
def swept(monkeypatch: pytest.MonkeyPatch) -> list[Path | None]:
    """The roots the sweep ran against; no real backend is opened."""
    roots: list[Path | None] = []
    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config", lambda _cfg, namespace: nullcontext(object())
    )

    def sweep(_backend: object, *, project_root: Path | None, **_knobs: object) -> MaintainVerifySummary:
        roots.append(project_root)
        return MaintainVerifySummary()

    monkeypatch.setattr("trw_memory.lifecycle.verification_pass.run_maintain_verify", sweep)
    return roots


def _token(root: str | None) -> Iterator[None]:
    claims = {"root": root} if root is not None else {}
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"], claims=claims)
    reset = auth_context_var.set(AuthenticatedUser(token))
    yield
    auth_context_var.reset(reset)


def _verify(**arguments: object) -> dict[str, object]:
    from trw_memory.tools.verify import register_verify_tool

    server = _Captured()
    register_verify_tool(server)  # type: ignore[arg-type]
    return asyncio.run(server.tools["memory_verify"](namespace=_ALPHA, **arguments))  # type: ignore[operator]


def test_a_grant_records_its_checkout_root_as_a_token_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    paths = DaemonPaths.resolve()
    checkout = tmp_path / "repo"
    checkout.mkdir()

    accepted = asyncio.run(LoopbackTokenVerifier(paths).verify_token(mint_grant(paths, [_ALPHA], root=checkout)))

    assert accepted is not None
    assert accepted.claims["root"] == str(checkout.resolve())


def test_a_foreign_root_is_refused_before_any_sweep(tmp_path: Path, swept: list[Path | None]) -> None:
    checkout = tmp_path / "repo"
    for _ in _token(str(checkout)):
        answer = _verify(project_root=str(tmp_path / "elsewhere"))

    assert answer["status"] == "refused"
    assert str(checkout) in str(answer["error"])
    assert swept == []


def test_the_granted_root_is_used_when_the_caller_names_it_or_none(tmp_path: Path, swept: list[Path | None]) -> None:
    checkout = tmp_path / "repo"
    for _ in _token(str(checkout)):
        assert _verify(project_root=str(checkout))["status"] == "ok"
        assert _verify(project_root=None)["status"] == "ok"

    assert swept == [checkout, checkout]


def test_a_grant_without_a_root_cannot_verify_over_the_transport(swept: list[Path | None]) -> None:
    for _ in _token(None):
        answer = _verify(project_root=None)

    assert answer["status"] == "refused"
    assert "trw-mcp memory token" in str(answer["error"])
    assert swept == []


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_limit": "5"},
        {"batch_limit": True},
        {"batch_limit": 0},
        {"batch_limit": 100_001},
        {"assertion_stale_threshold_days": 0},
        {"assertion_stale_threshold_days": 1.5},
        {"assertion_failure_penalty": 1.5},
        {"assertion_failure_penalty": "high"},
        {"anchor_validity_verified_floor": -0.1},
        {"assertion_failure_penalty": 10**1000},
        {"batch_limit": 10**1000},
        {"unknown_knob": 1},
    ],
)
def test_out_of_range_or_mistyped_settings_are_invalid_before_the_sweep(
    settings: dict[str, object], tmp_path: Path, swept: list[Path | None]
) -> None:
    for _ in _token(str(tmp_path)):
        answer = _verify(project_root=None, settings=settings)

    assert answer["status"] == "invalid"
    assert swept == []


def test_integral_floats_and_int_penalties_are_accepted() -> None:
    assert VerifySettings(assertion_failure_penalty=0, anchor_validity_verified_floor=1).batch_limit == 1000


def test_a_pre_root_grants_file_stays_usable_and_a_re_mint_adds_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upgraded install's digest-to-list entries read as rootless grants; minting beside them succeeds."""
    import json

    from trw_memory.daemon._grants import _digest, read_grant

    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    paths = DaemonPaths.resolve()
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    paths.grants.write_text(json.dumps({_digest("old-token"): [_ALPHA]}), encoding="utf-8")

    old = read_grant(paths, "old-token")
    assert old is not None
    assert (old.namespaces, old.root) == (frozenset({_ALPHA}), None), "rootless: memory_verify refuses it"

    checkout = tmp_path / "repo"
    fresh = mint_grant(paths, [_ALPHA], root=checkout)

    assert read_grant(paths, fresh).root == str(checkout.resolve())  # type: ignore[union-attr]
    assert read_grant(paths, "old-token").namespaces == frozenset({_ALPHA})  # type: ignore[union-attr]


def test_a_non_finite_setting_is_invalid() -> None:
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            VerifySettings(assertion_failure_penalty=value)


def test_maintain_verifies_the_granted_checkout_not_the_daemons_root(
    tmp_path: Path, swept: list[Path | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.maintain import register_maintain_tool

    checkout = tmp_path / "repo"
    checkout.mkdir()
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(tmp_path))  # the daemon's own root: never swept for a grant
    backend = SQLiteBackend(tmp_path / "memory.db")
    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config", lambda _cfg, namespace: nullcontext(backend)
    )
    server = _Captured()
    register_maintain_tool(server)  # type: ignore[arg-type]
    try:
        for _ in _token(str(checkout)):
            asyncio.run(server.tools["memory_maintain"](namespace=_ALPHA))  # type: ignore[operator]
        for _ in _token(None):
            asyncio.run(server.tools["memory_maintain"](namespace=_ALPHA))  # type: ignore[operator]
    finally:
        backend.close()

    assert swept == [checkout.resolve(), None]
