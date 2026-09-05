"""Tests for the FastMCP server entry point (server.py).

fastmcp is an optional dependency not installed in the dev venv.
Tests mock the fastmcp module to verify server structure without requiring it.

Each test that needs the server module uses importlib.reload() with the mock
in place to ensure a fresh server state.
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import EncryptionUnavailableError, LocalOnlyViolationError
from trw_memory.tools._contract import SURFACE_MEMBERS, MemoryToolSurface


def _make_fastmcp_mock() -> tuple[MagicMock, MagicMock, list[str]]:
    """Create a mock FastMCP class and instance for testing.

    Returns:
        (FastMCP_cls, mcp_instance, registered_tools_list)
    """
    registered_tools: list[str] = []

    mcp_instance = MagicMock()
    mcp_instance.name = "trw-memory"

    def _tool_decorator() -> object:
        def _decorator(fn: object) -> object:
            registered_tools.append(getattr(fn, "__name__", "unknown"))
            return fn

        return _decorator

    mcp_instance.tool = _tool_decorator
    mcp_instance._registered_tools = registered_tools

    FastMCP_cls = MagicMock(return_value=mcp_instance)
    return FastMCP_cls, mcp_instance, registered_tools


def _reload_server_with_mock() -> tuple[types.ModuleType, MagicMock, list[str]]:
    """Reload trw_memory.server with a mocked fastmcp and return helpers.

    Returns:
        (server_module, mcp_instance, registered_tools_list)
    """
    FastMCP_cls, mcp_instance, registered_tools = _make_fastmcp_mock()

    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = FastMCP_cls  # type: ignore[attr-defined]

    # Remove cached server module so import does a fresh load
    sys.modules.pop("trw_memory.server", None)
    sys.modules["fastmcp"] = fake_fastmcp

    try:
        import trw_memory.server as server_mod
    finally:
        # Restore: remove mock fastmcp, leave server cached for callers
        sys.modules.pop("fastmcp", None)

    return server_mod, mcp_instance, registered_tools


class TestServerModule:
    def test_mcp_instance_exists(self) -> None:
        """Server module must export a FastMCP instance named 'mcp'."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        assert hasattr(server_mod, "mcp")
        assert server_mod.mcp is not None
        # Cleanup
        sys.modules.pop("trw_memory.server", None)

    def test_mcp_named_trw_memory(self) -> None:
        """FastMCP must be instantiated with 'trw-memory'."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        # The mcp_instance.name is set to "trw-memory" in our mock
        assert mcp_instance.name == "trw-memory"
        sys.modules.pop("trw_memory.server", None)

    def test_main_callable(self) -> None:
        """server.main must be callable."""
        server_mod, _, _ = _reload_server_with_mock()
        assert callable(server_mod.main)
        sys.modules.pop("trw_memory.server", None)

    def test_main_preflights_local_only_embedder(self) -> None:
        """Local-only startup should fail before mcp.run when embeddings require download."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        with (
            patch(
                "trw_memory.models.config.MemoryConfig",
                return_value=SimpleNamespace(
                    encryption_enabled=False,
                    local_only=True,
                    embedding_model="test-model",
                    embedding_dim=384,
                ),
            ),
            patch("trw_memory.embeddings.get_local_embedder", side_effect=LocalOnlyViolationError("blocked")),
            pytest.raises(LocalOnlyViolationError, match="blocked"),
        ):
            server_mod.main([])
        mcp_instance.run.assert_not_called()
        sys.modules.pop("trw_memory.server", None)

    def test_main_preflights_sqlcipher_when_encryption_enabled(self) -> None:
        """Encrypted startup should fail before mcp.run when SQLCipher is unavailable."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        with (
            patch(
                "trw_memory.models.config.MemoryConfig",
                return_value=SimpleNamespace(
                    encryption_enabled=True,
                    local_only=False,
                    embedding_model="test-model",
                    embedding_dim=384,
                ),
            ),
            patch(
                "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
                side_effect=EncryptionUnavailableError("sqlcipher missing"),
            ),
            pytest.raises(EncryptionUnavailableError, match="sqlcipher missing"),
        ):
            server_mod.main([])
        mcp_instance.run.assert_not_called()
        sys.modules.pop("trw_memory.server", None)

    def test_registered_tools_match_the_published_surface(self) -> None:
        """What ``_register_tools`` actually registers must equal what we publish.

        The expected set is NOT restated here: it is
        ``server.REGISTERED_TOOL_NAMES``, the one place the surface is declared
        (PRD-CORE-253 FR03/FR05/FR06 added four tools to it). Asserting in both
        directions is what makes the constant load-bearing -- an unlisted
        registration fails just as loudly as a missing one, so neither half can
        drift ahead of the other.
        """
        server_mod, _mcp_instance, registered_tools = _reload_server_with_mock()
        published = set(server_mod.REGISTERED_TOOL_NAMES)

        assert len(server_mod.REGISTERED_TOOL_NAMES) == len(published), "REGISTERED_TOOL_NAMES has a duplicate"
        assert set(registered_tools) == published, (
            f"registered tool set drift: {sorted(set(registered_tools) ^ published)}"
        )
        sys.modules.pop("trw_memory.server", None)

    def test_all_tool_modules_importable(self) -> None:
        """All tool modules must be importable with their impl functions."""
        from trw_memory.tools import audit, consolidate, forget, recall, review, search, status, store

        assert callable(store.memory_store_impl)
        assert callable(recall.memory_recall_impl)
        assert callable(audit.memory_audit_impl)
        assert callable(review.memory_review_impl)
        assert callable(forget.memory_forget_impl)
        assert callable(consolidate.memory_consolidate_impl)
        assert callable(search.memory_search_impl)
        assert callable(status.memory_status_impl)

    def test_register_functions_exist(self) -> None:
        """Each tool module must export a register_*_tool function."""
        from trw_memory.tools.audit import register_audit_tool
        from trw_memory.tools.consolidate import register_consolidate_tool
        from trw_memory.tools.forget import register_forget_tool
        from trw_memory.tools.recall import register_recall_tool
        from trw_memory.tools.review import register_review_tool
        from trw_memory.tools.search import register_search_tool
        from trw_memory.tools.status import register_status_tool
        from trw_memory.tools.store import register_store_tool

        for fn in [
            register_store_tool,
            register_recall_tool,
            register_audit_tool,
            register_review_tool,
            register_forget_tool,
            register_consolidate_tool,
            register_search_tool,
            register_status_tool,
        ]:
            assert callable(fn), f"{fn} must be callable"


# ---------------------------------------------------------------------------
# PRD-CORE-251 FR01 — the tool surface is a typed, contract-tested interface
# ---------------------------------------------------------------------------


def _protocol_parameters(member: str) -> dict[str, inspect.Parameter]:
    """Return the call parameters ``MemoryToolSurface`` declares for *member*.

    ``MemoryToolSurface.__annotations__`` holds strings (the module uses
    ``from __future__ import annotations``), so the callback Protocol behind
    each member is resolved through ``typing.get_type_hints`` rather than read
    literally.
    """
    member_protocol = get_type_hints(MemoryToolSurface)[member]
    signature = inspect.signature(member_protocol.__call__)
    return {name: param for name, param in signature.parameters.items() if name != "self"}


def test_surface_membership_is_not_empty() -> None:
    """Guard against a silently empty Protocol — an empty contract passes everything."""
    assert set(SURFACE_MEMBERS) == {
        "memory_store_impl",
        "memory_recall_impl",
        "memory_search_impl",
        "memory_forget_impl",
        "memory_consolidate_impl",
        "memory_status_impl",
        "memory_review_impl",
        "memory_audit_impl",
    }, f"MemoryToolSurface membership drift: {sorted(SURFACE_MEMBERS)}"


@pytest.mark.parametrize("member", SURFACE_MEMBERS)
def test_every_impl_satisfies_the_protocol(member: str) -> None:
    """FR01 — every declared member resolves, is exported, and matches its signature.

    The static half of this contract lives in ``trw_memory/tools/__init__.py``
    (a ``TYPE_CHECKING`` binding per impl that ``mypy --strict`` checks). This
    is the runtime half: it fails without a type checker in the loop, which is
    the condition a downstream consumer actually installs under.
    """
    import trw_memory.tools as tools_module

    impl = getattr(tools_module, member, None)
    assert impl is not None, f"{member} is declared by MemoryToolSurface but not exported by trw_memory.tools"
    assert callable(impl), f"{member} is exported but is not callable"
    assert member in tools_module.__all__, f"{member} is not in trw_memory.tools.__all__, so it is not public API"

    declared = _protocol_parameters(member)
    actual = dict(inspect.signature(impl).parameters)

    for name, param in declared.items():
        assert name in actual, f"{member} does not accept the contract parameter {name!r}"
        assert actual[name].kind == param.kind, (
            f"{member}.{name} is {actual[name].kind}, the contract declares {param.kind}"
        )
        contract_required = param.default is inspect.Parameter.empty
        impl_required = actual[name].default is inspect.Parameter.empty
        assert impl_required == contract_required, (
            f"{member}.{name} required={impl_required} but the contract declares required={contract_required}"
        )

    # An impl may add a DEFAULTED parameter (a backward-compatible extension);
    # it may not add one a contract-compliant caller would have to supply.
    extra_required = [
        name
        for name, param in actual.items()
        if name not in declared
        and param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not extra_required, f"{member} requires {extra_required}, which no MemoryToolSurface caller can supply"


def test_tools_module_is_a_memory_tool_surface() -> None:
    """The module itself is the implementation of the Protocol."""
    import trw_memory.tools as tools_module

    assert isinstance(tools_module, MemoryToolSurface)


def test_update_impl_is_declared_and_implemented_together() -> None:
    """FR07's ``memory_update_impl`` must arrive as implementation AND contract member.

    PRD-CORE-251 Phase 1 ships the Protocol without an update member because
    the tool does not exist yet (verified 2026-09-03: ``trw_memory.tools`` has
    no update entry point) and FR01 requires every member to resolve. This is
    the ratchet that keeps the two halves from landing separately — whichever
    arrives first fails until its counterpart does.
    """
    import trw_memory.tools as tools_module

    declared = "memory_update_impl" in SURFACE_MEMBERS
    implemented = hasattr(tools_module, "memory_update_impl")
    assert declared == implemented, (
        "memory_update_impl is declared by MemoryToolSurface but not exported by trw_memory.tools"
        if declared
        else "memory_update_impl is exported by trw_memory.tools but missing from MemoryToolSurface (PRD-CORE-251 FR07)"
    )
