"""SYNC-FIND-MANY: a pulled page maps to its local rows in ONE daemon call, not one call per row.

``merge_team_learnings`` used to call ``memory_sync_find`` once per pulled learning; under load each round trip took
about 185 ms, so a 200-row page of unchanged learnings took 37 s and crowded recall out of the daemon's four workers.
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
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.sync import memory_sync_find_many_impl

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    store.store(make_entry(entry_id="A-0", namespace=_ALPHA, content="by id"))
    store.store(MemoryEntry(id="A-1", namespace=_ALPHA, content="by remote id", remote_id="R-1"))
    store.store(make_entry(entry_id="A-2", namespace=_ALPHA, content="unrelated"))
    store.store(MemoryEntry(id="B-0", namespace=_BETA, content="other namespace", remote_id="R-1"))
    yield store
    store.close()


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path), embeddings_enabled=False)


def _ids(answer: dict[str, object]) -> list[str]:
    assert answer["status"] == "ok"
    return sorted(str(entry["id"]) for entry in answer["entries"])  # type: ignore[attr-defined,union-attr]


def test_one_call_returns_every_row_a_remote_id_or_an_id_names(backend: SQLiteBackend, config: MemoryConfig) -> None:
    answer = memory_sync_find_many_impl(_ALPHA, ["R-1", "R-none"], ["A-0", "A-nope"], backend=backend, config=config)

    assert _ids(answer) == ["A-0", "A-1"]


def test_a_row_in_another_namespace_is_never_returned(backend: SQLiteBackend, config: MemoryConfig) -> None:
    answer = memory_sync_find_many_impl(_ALPHA, ["R-1"], [], backend=backend, config=config)

    assert _ids(answer) == ["A-1"]  # B-0 carries the same remote id in beta


def test_nothing_asked_answers_nothing(backend: SQLiteBackend, config: MemoryConfig) -> None:
    assert _ids(memory_sync_find_many_impl(_ALPHA, [], [], backend=backend, config=config)) == []


def test_a_page_larger_than_one_sqlite_statement_still_resolves(backend: SQLiteBackend, config: MemoryConfig) -> None:
    remote_ids = [f"R-{n}" for n in range(1000)]
    ids = [f"I-{n}" for n in range(1000)] + ["A-2"]

    answer = memory_sync_find_many_impl(_ALPHA, remote_ids, ids, backend=backend, config=config)

    assert _ids(answer) == ["A-1", "A-2"]


def test_an_ungranted_namespace_is_refused(backend: SQLiteBackend, config: MemoryConfig) -> None:
    token = AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"]))
    reset = auth_context_var.set(token)
    try:
        with pytest.raises(AuthorizationError, match=_BETA):
            memory_sync_find_many_impl(_BETA, ["R-1"], [], backend=backend, config=config)
    finally:
        auth_context_var.reset(reset)


def test_the_daemon_serves_and_forwards_the_batched_find() -> None:
    from trw_memory.daemon.client import DaemonClient
    from trw_memory.server import REGISTERED_TOOL_NAMES

    assert "memory_sync_find_many" in REGISTERED_TOOL_NAMES
    assert callable(DaemonClient.sync_find_many)
