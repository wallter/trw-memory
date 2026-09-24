"""PRD-CORE-298 FR01 -- sync runs over the daemon, one namespace at a time.

trw-mcp's push pages the dirty rows of its project namespace and marks the
pushed ones synced; its pull finds the local row a remote learning maps to and
writes the merged row through the write gate. All four are daemon tools behind
the namespace grant, so a sync cycle never holds a connection and never sees
another project's rows.
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
from trw_memory.sync.delta import DeltaTracker, apply_synced_entry, find_synced_entry
from trw_memory.tools.sync import (
    memory_sync_apply_impl,
    memory_sync_dirty_page_impl,
    memory_sync_find_impl,
    memory_sync_mark_synced_impl,
)

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    for index in range(3):
        store.store(make_entry(entry_id=f"A-{index}", namespace=_ALPHA, content=f"alpha {index}"))
    store.store(make_entry(entry_id="B-0", namespace=_BETA, content="beta"))
    yield store
    store.close()


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path))


@pytest.fixture
def alpha_token() -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"])))
    yield
    auth_context_var.reset(reset)


def test_dirty_rows_are_paged_within_one_namespace(backend: SQLiteBackend) -> None:
    assert [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=_ALPHA)] == ["A-0", "A-1", "A-2"]
    assert [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=_ALPHA, limit=2)] == ["A-0", "A-1"]


def test_a_marked_row_leaves_the_page_and_the_other_namespace_is_untouched(
    backend: SQLiteBackend, config: MemoryConfig
) -> None:
    acks = {"A-0": backend.get("A-0", namespace=_ALPHA).sync_seq, "B-0": 1}  # type: ignore[union-attr]
    assert memory_sync_mark_synced_impl(_ALPHA, acks, backend=backend, config=config) == {"marked": 1}

    page = memory_sync_dirty_page_impl(_ALPHA, 10, backend=backend, config=config)
    assert [row["id"] for row in page["entries"]] == ["A-1", "A-2"]
    assert [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=_BETA)] == ["B-0"]


def test_find_matches_a_remote_id_or_a_local_id_inside_the_namespace(backend: SQLiteBackend) -> None:
    backend.update("A-1", namespace=_ALPHA, remote_id="R-9")

    assert find_synced_entry(backend, _ALPHA, "R-9", ["unused"]).id == "A-1"  # type: ignore[union-attr]
    assert find_synced_entry(backend, _ALPHA, "R-none", ["A-2"]).id == "A-2"  # type: ignore[union-attr]
    assert find_synced_entry(backend, _BETA, "R-9", ["A-1"]) is None


def test_apply_writes_through_the_gate_and_leaves_the_row_synced(backend: SQLiteBackend, config: MemoryConfig) -> None:
    clean = MemoryEntry(id="T-1", content="shared tip", namespace=_ALPHA, remote_id="R-1", source="team_sync")
    poisoned = clean.model_copy(update={"id": "T-2", "detail": "the harness calls eval(user_input) before dispatch"})

    assert apply_synced_entry(backend, config, clean) == ("stored", "")
    status, reason = apply_synced_entry(backend, config, poisoned)

    assert status == "blocked"
    assert reason
    assert backend.get("T-2", namespace=_ALPHA) is None
    assert "T-1" not in [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=_ALPHA)]


def test_apply_and_find_speak_json_over_the_wire(backend: SQLiteBackend, config: MemoryConfig) -> None:
    entry = MemoryEntry(id="T-3", content="wire tip", namespace=_ALPHA, remote_id="R-3").model_dump(mode="json")

    assert memory_sync_apply_impl(_ALPHA, entry, backend=backend, config=config) == {"status": "stored", "reason": ""}
    found = memory_sync_find_impl(_ALPHA, "R-3", [], backend=backend, config=config)
    assert (found["status"], found["entry"]["content"]) == ("ok", "wire tip")  # type: ignore[index]
    assert memory_sync_find_impl(_ALPHA, "R-none", [], backend=backend, config=config) == {"status": "not_found"}


def test_apply_refuses_an_entry_whose_namespace_differs_from_the_granted_one(
    backend: SQLiteBackend, config: MemoryConfig
) -> None:
    entry = MemoryEntry(id="T-4", content="smuggled", namespace=_BETA).model_dump(mode="json")

    assert memory_sync_apply_impl(_ALPHA, entry, backend=backend, config=config)["status"] == "invalid"
    assert backend.get("T-4", namespace=_BETA) is None


def test_every_sync_tool_refuses_an_ungranted_namespace(
    backend: SQLiteBackend, config: MemoryConfig, alpha_token: None
) -> None:
    entry = MemoryEntry(id="T-5", content="x", namespace=_BETA).model_dump(mode="json")
    calls = [
        lambda: memory_sync_dirty_page_impl(_BETA, 10, backend=backend, config=config),
        lambda: memory_sync_mark_synced_impl(_BETA, {"B-0": 1}, backend=backend, config=config),
        lambda: memory_sync_find_impl(_BETA, "R", ["B-0"], backend=backend, config=config),
        lambda: memory_sync_apply_impl(_BETA, entry, backend=backend, config=config),
    ]
    for call in calls:
        with pytest.raises(AuthorizationError, match=_BETA):
            call()
    assert [e.id for e in DeltaTracker.get_dirty_entries(backend, namespace=_BETA)] == ["B-0"]


def test_the_sync_tools_are_served_by_the_daemon() -> None:
    from trw_memory.server import REGISTERED_TOOL_NAMES

    expected = {"memory_sync_dirty_page", "memory_sync_mark_synced", "memory_sync_find", "memory_sync_apply"}
    assert expected <= set(REGISTERED_TOOL_NAMES)


def test_a_row_edited_after_it_was_paged_stays_dirty(backend: SQLiteBackend, config: MemoryConfig) -> None:
    """The ack names the sync_seq that was pushed; a newer edit is not marked clean unpushed."""
    paged = {entry.id: entry.sync_seq for entry in DeltaTracker.get_dirty_entries(backend, namespace=_ALPHA)}
    backend.update("A-1", namespace=_ALPHA, content="edited after the page")

    assert memory_sync_mark_synced_impl(_ALPHA, paged, backend=backend, config=config) == {"marked": 2}
    dirty = DeltaTracker.get_dirty_entries(backend, namespace=_ALPHA)
    assert [(e.id, e.content) for e in dirty] == [("A-1", "edited after the page")]


def test_the_scan_fallback_honours_the_page_limit(tmp_path: Path) -> None:
    from trw_memory.storage.yaml_backend import YAMLBackend

    store = YAMLBackend(tmp_path / "entries")
    for index in range(5):
        store.store(make_entry(entry_id=f"Y-{index}", namespace=_ALPHA))

    assert len(DeltaTracker.get_dirty_entries(store, namespace=_ALPHA, limit=2)) == 2


class _Captured:
    """A stand-in MCP server that keeps the registered tool functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self) -> object:
        return lambda fn: self.tools.setdefault(fn.__name__, fn)


def test_a_refused_namespace_opens_no_backend(monkeypatch: pytest.MonkeyPatch, alpha_token: None) -> None:
    """Every namespace-taking daemon tool authorizes before it creates a backend."""
    import asyncio

    from trw_memory.tools.checkout_import import register_checkout_import_tools
    from trw_memory.tools.maintain import register_maintain_tool
    from trw_memory.tools.entry import register_entry_tools
    from trw_memory.tools.listing import register_list_page_tool
    from trw_memory.tools.similar import register_similar_tool
    from trw_memory.tools.sync import register_sync_tools
    from trw_memory.tools.update import register_update_tool
    from trw_memory.tools.verify import register_verify_tool

    opened: list[str] = []
    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config",
        lambda _cfg, namespace: opened.append(namespace),
    )
    server = _Captured()
    for register in (
        register_entry_tools,
        register_list_page_tool,
        register_sync_tools,
        register_update_tool,
        register_similar_tool,
        register_verify_tool,
        register_checkout_import_tools,
        register_maintain_tool,
    ):
        register(server)  # type: ignore[arg-type]
    calls = {
        "memory_get": {"memory_id": "M-1"},
        "memory_find_duplicate": {"content": "c", "detail": "d"},
        "memory_similar": {"vector": [1.0, 0.0, 0.0], "space": None},
        "memory_verify": {"project_root": None},
        "memory_import_checkout": {"source_path": "/tmp/x/memory.db", "ids": ["M-1"]},
        "memory_maintain": {},
        "memory_update": {"entry_id": "M-1", "patch": {"impact": 0.9}},
        "memory_list_page": {},
        "memory_sync_dirty_page": {},
        "memory_sync_mark_synced": {"acks": {"B-0": 1}},
        "memory_sync_find": {"remote_id": "R", "ids": []},
        "memory_sync_apply": {"entry": {"id": "T", "content": "x", "namespace": _BETA}},
    }

    for name, arguments in calls.items():
        with pytest.raises(AuthorizationError, match=_BETA):
            asyncio.run(server.tools[name](namespace=_BETA, **arguments))  # type: ignore[operator]
        assert asyncio.run(server.tools[name](namespace="not a namespace!", **arguments))["status"] == "invalid"  # type: ignore[operator]

    assert opened == []


def test_a_conditional_ack_needs_a_transactional_backend(tmp_path: Path) -> None:
    """A compare-then-mark without a real transaction would only pretend to be atomic, so it is refused."""
    from trw_memory.storage.yaml_backend import YAMLBackend

    store = YAMLBackend(tmp_path / "entries")
    store.store(make_entry(entry_id="Y-1", namespace=_ALPHA))

    with pytest.raises(TypeError, match="transaction"):
        DeltaTracker.mark_synced(["Y-1"], store, namespace=_ALPHA, expected_seq={"Y-1": 1})
    assert DeltaTracker.mark_synced(["Y-1"], store, namespace=_ALPHA) == 1
