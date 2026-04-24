"""Interop tests for registered MCP tool wrappers and MemoryClient."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from trw_memory.client import MemoryClient
from trw_memory.tools.recall import register_recall_tool
from trw_memory.tools.search import register_search_tool
from trw_memory.tools.store import register_store_tool


class _FakeMCP:
    """Minimal FastMCP-like registry used to capture registered tool callables."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Coroutine[Any, Any, dict[str, object]]]] = {}

    def tool(
        self,
    ) -> Callable[
        [Callable[..., Coroutine[Any, Any, dict[str, object]]]],
        Callable[..., Coroutine[Any, Any, dict[str, object]]],
    ]:
        def _decorator(
            fn: Callable[..., Coroutine[Any, Any, dict[str, object]]],
        ) -> Callable[..., Coroutine[Any, Any, dict[str, object]]]:
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


async def test_registered_store_tool_writes_visible_to_client(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tool wrapper and client must resolve the same namespace-scoped backend."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    mcp = _FakeMCP()
    register_store_tool(mcp)

    await mcp.tools["memory_store"](
        content="stored via tool wrapper",
        namespace="project:interop",
        tags=["interop"],
    )

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        results = await client.search(tags=["interop"])
    finally:
        await client.close()

    assert [result["content"] for result in results] == ["stored via tool wrapper"]


async def test_registered_search_tool_reads_entries_stored_by_client(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tool wrapper must see entries created through the local client."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        await client.store("stored via client", tags=["interop"])
    finally:
        await client.close()

    mcp = _FakeMCP()
    register_search_tool(mcp)
    result = await mcp.tools["memory_search"](
        namespace="project:interop",
        tags=["interop"],
    )

    entries = result["entries"]
    assert isinstance(entries, list)
    assert [entry["content"] for entry in entries] == ["stored via client"]


async def test_registered_recall_tool_accepts_source_aware_args(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Registered recall tool must expose the same source-aware policy args as the client."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        await client.store(
            "durable rule",
            metadata={"source_kind": "instruction_rule"},
            entry_id="M-durable",
        )
        await client.store(
            "ephemeral bulletin",
            metadata={"source_kind": "lifecycle"},
            expires="2020-01-01T00:00:00+00:00",
            entry_id="M-expired",
        )
    finally:
        await client.close()

    mcp = _FakeMCP()
    register_recall_tool(mcp)
    result = await mcp.tools["memory_recall"](
        query="",
        namespace="project:interop",
        include_source_kinds=["instruction_rule", "lifecycle"],
        exclude_expired=True,
    )

    memories = result["memories"]
    assert isinstance(memories, list)
    assert [entry["id"] for entry in memories] == ["M-durable"]
