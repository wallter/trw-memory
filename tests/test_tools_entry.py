"""PRD-CORE-298 FR01 -- ``memory_get`` / ``memory_update``: one entry by id, inside the grant.

trw-mcp's ``trw_learn_update`` reads an entry and patches fields on it. Over the
daemon that needs a get and an update tool; both are namespace-qualified and go
through the same grant step as every other tool, and an update can never move a
row to another namespace or rename it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests.conftest import make_entry
from trw_memory.exceptions import AuthorizationError
from trw_memory.lifecycle.correction import LearningPatch, parse_patch
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.entry import memory_get_impl
from trw_memory.tools.update import memory_update_impl

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    for namespace in (_ALPHA, _BETA):
        store.store(make_entry(entry_id="M-1", namespace=namespace, content=f"row in {namespace}", importance=0.4))
    yield store
    store.close()


@pytest.fixture
def alpha_token() -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"])))
    yield
    auth_context_var.reset(reset)


def test_get_returns_the_namespace_qualified_row(backend: SQLiteBackend) -> None:
    result = memory_get_impl("M-1", _ALPHA, backend=backend, config=MemoryConfig())

    assert result["status"] == "ok"
    assert result["entry"]["content"] == f"row in {_ALPHA}"  # type: ignore[index]
    assert memory_get_impl("M-none", _ALPHA, backend=backend, config=MemoryConfig())["status"] == "not_found"


@pytest.mark.parametrize("field", ["namespace", "id"])
def test_a_correction_never_moves_or_renames_a_row(field: str) -> None:
    # memory_update takes a correction patch (PRD-CORE-294 FR03); identity is not a patch field.
    assert parse_patch({field: _BETA})["status"] == "invalid"  # type: ignore[index]


def test_both_verbs_refuse_an_ungranted_namespace(backend: SQLiteBackend, alpha_token: None) -> None:
    with pytest.raises(AuthorizationError, match=_BETA):
        memory_get_impl("M-1", _BETA, backend=backend, config=MemoryConfig())
    with pytest.raises(AuthorizationError, match=_BETA):
        memory_update_impl("M-1", LearningPatch(impact=0.1), _BETA, backend=backend, config=MemoryConfig())
    assert backend.get("M-1", namespace=_BETA).importance == 0.4  # type: ignore[union-attr]


def test_both_verbs_are_served_by_the_daemon() -> None:
    from trw_memory.server import REGISTERED_TOOL_NAMES

    assert {"memory_get", "memory_update"} <= set(REGISTERED_TOOL_NAMES)


def test_the_daemon_store_tool_accepts_every_entry_field_the_impl_does() -> None:
    """trw_learn sends type, confidence, anchors and the rest; over the daemon none may be dropped."""
    import asyncio
    import inspect

    from trw_memory.server import mcp
    from trw_memory.tools.store import LearningFields, memory_store_impl

    wiring = {"backend", "config", "trw_dir"}
    tool = asyncio.run(mcp.get_tool("memory_store"))
    # Typed learning fields arrive in the one ``learning`` object (PRD-CORE-294 FR07a).
    accepted = (set(tool.parameters["properties"]) - {"learning"}) | set(LearningFields.model_fields)
    assert set(inspect.signature(memory_store_impl).parameters) - wiring <= accepted
