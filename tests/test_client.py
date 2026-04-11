"""Tests for MemoryClient SDK.

Covers:
- Constructor with different modes (local, auto, mcp stub)
- store() — valid content, empty content, invalid importance
- recall() — sorted by score, limit validation
- forget() — existing entry, non-existent entry
- search() — tags filter, min_importance filter, since filter, limit validation
- Thread safety — concurrent store/recall via asyncio.gather
- Context manager — async with
- register_tools() — mock agent, double registration
- Connection mode auto-detection
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient, MemoryResultDict, StoreResultDict
from trw_memory.exceptions import (
    MemoryNotFoundError,
    ToolAlreadyRegisteredError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Create a MemoryClient with local SQLite backend in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def yaml_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Create a MemoryClient with YAML backend in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "yaml_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    return MemoryClient(namespace="default", mode="local")


# ---------------------------------------------------------------------------
# Constructor tests (FR01 + FR07)
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_local_mode_creates_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        c = MemoryClient(namespace="default", mode="local")
        assert c.resolved_mode == "local"
        assert c.namespace == "default"

    def test_auto_mode_resolves_to_local(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        c = MemoryClient(namespace="default", mode="auto")
        assert c.resolved_mode == "local"

    def test_mcp_mode_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="MCP mode"):
            MemoryClient(namespace="default", mode="mcp")

    def test_invalid_namespace_raises(self) -> None:
        with pytest.raises(Exception):
            MemoryClient(namespace="invalid namespace!")

    def test_project_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        c = MemoryClient(namespace="project:test-proj", mode="local")
        assert c.namespace == "project:test-proj"
        assert c.resolved_mode == "local"

    def test_timeout_stored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        c = MemoryClient(
            namespace="default",
            mode="local",
            timeout=10.0,
        )
        assert c._timeout == 10.0


# ---------------------------------------------------------------------------
# store() tests (FR02)
# ---------------------------------------------------------------------------


class TestStore:
    async def test_store_returns_expected_keys(self, client: MemoryClient) -> None:
        result = await client.store("test content", tags=["tag1"])
        assert "memory_id" in result
        assert result["memory_id"].startswith("M-")
        assert result["namespace"] == "default"
        assert result["status"] == "stored"
        assert "timestamp" in result

    async def test_store_empty_content_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="content must not be empty"):
            await client.store("")

    async def test_store_whitespace_only_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="content must not be empty"):
            await client.store("   ")

    async def test_store_importance_too_low_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="importance"):
            await client.store("content", importance=-0.1)

    async def test_store_importance_too_high_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="importance"):
            await client.store("content", importance=1.1)

    async def test_store_with_metadata(self, client: MemoryClient) -> None:
        result = await client.store(
            "content with meta",
            metadata={"source": "test"},
        )
        assert result["status"] == "stored"

    async def test_store_boundary_importance(self, client: MemoryClient) -> None:
        r0 = await client.store("min importance", importance=0.0)
        r1 = await client.store("max importance", importance=1.0)
        assert r0["status"] == "stored"
        assert r1["status"] == "stored"

    async def test_store_with_detail(self, client: MemoryClient) -> None:
        result = await client.store("summary", detail="extended explanation", tags=["a"])
        assert result["status"] == "stored"

    async def test_store_sync_publish_marks_entry_as_published(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        client = MemoryClient(namespace="default", mode="local")

        with patch("trw_memory.client.publish_memory", return_value=True):
            stored = await client.store("publish this entry", importance=0.9)
            await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        entry = reopened._get_backend().get(stored["memory_id"])
        assert entry is not None
        assert entry.published_to_platform is True
        await reopened.close()

    async def test_store_sync_failure_enqueues_retry_payload(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        client = MemoryClient(namespace="default", mode="local")

        with (
            patch("trw_memory.client.publish_memory", return_value=False),
            patch("trw_memory.client._anonymize_entry", return_value={"summary": "queued"}),
        ):
            stored = await client.store("queue this entry", importance=0.9)
            await client.close()

        queue = client._retry_queue
        assert queue.depth() == 1
        lines = (Path(tmp_path) / "storage" / "sync_queue.jsonl").read_text(encoding="utf-8").splitlines()
        assert stored["memory_id"] in lines[0]


# ---------------------------------------------------------------------------
# recall() tests (FR03)
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_recall_returns_stored_entry(self, client: MemoryClient) -> None:
        await client.store("pydantic validation error", tags=["pydantic"])
        results = await client.recall("pydantic")
        assert len(results) >= 1
        assert results[0]["content"] == "pydantic validation error"

    async def test_recall_sorted_by_score_desc(self, client: MemoryClient) -> None:
        await client.store("low importance match", importance=0.2)
        await client.store("high importance match", importance=0.9)
        results = await client.recall("importance match")
        assert len(results) >= 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_recall_limit_zero_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.recall("query", limit=0)

    async def test_recall_negative_limit_raises(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.recall("query", limit=-5)

    async def test_recall_respects_limit(self, client: MemoryClient) -> None:
        for i in range(5):
            await client.store(f"entry number {i}", importance=0.5)
        results = await client.recall("entry number", limit=3)
        assert len(results) <= 3

    async def test_recall_with_tags_filter(self, client: MemoryClient) -> None:
        await client.store("tagged entry", tags=["python", "testing"])
        await client.store("untagged entry")
        results = await client.recall("entry", tags=["python"])
        assert all("python" in r["tags"] for r in results)

    async def test_recall_empty_result(self, client: MemoryClient) -> None:
        results = await client.recall("nonexistent query xyz")
        assert results == []

    async def test_recall_min_score_filters(self, client: MemoryClient) -> None:
        await client.store("low score entry", importance=0.1)
        await client.store("high score entry", importance=0.8)
        results = await client.recall("score entry", min_score=0.5)
        for r in results:
            assert r["score"] >= 0.5

    async def test_recall_include_shared_appends_remote_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        client = MemoryClient(namespace="default", mode="local")
        await client.store("local entry", importance=0.8)
        shared_result = {
            "memory_id": "remote-1",
            "content": "[shared] remote entry",
            "detail": "remote detail",
            "tags": ["shared"],
            "importance": 0.6,
            "score": 0.55,
            "namespace": "team:shared",
            "created_at": "",
            "updated_at": "",
            "source": "shared",
        }

        with patch("trw_memory.client.fetch_shared_memories", return_value=[shared_result]) as fetch_mock:
            results = await client.recall("entry", include_shared=True)

        assert fetch_mock.called
        assert any(result["source"] == "shared" for result in results)
        assert results[0]["source"] == "local"
        await client.close()


# ---------------------------------------------------------------------------
# forget() tests (FR04)
# ---------------------------------------------------------------------------


class TestForget:
    async def test_forget_existing_entry(self, client: MemoryClient) -> None:
        stored = await client.store("to be forgotten")
        memory_id = stored["memory_id"]
        result = await client.forget(memory_id)
        assert result["memory_id"] == memory_id
        assert result["status"] == "deleted"
        assert result["namespace"] == "default"

    async def test_forget_nonexistent_raises(self, client: MemoryClient) -> None:
        with pytest.raises(MemoryNotFoundError):
            await client.forget("M-nonexistent")

    async def test_forget_then_recall_empty(self, client: MemoryClient) -> None:
        stored = await client.store("unique ephemeral content xyz123")
        await client.forget(stored["memory_id"])
        results = await client.recall("unique ephemeral content xyz123")
        assert len(results) == 0

    async def test_forget_wrong_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Entry in namespace A cannot be forgotten by client in namespace B."""
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ns_test"))
        client_a = MemoryClient(namespace="project:aaa", mode="local")
        stored = await client_a.store("ns-A entry")

        # client_b uses a different namespace, but same storage path
        client_b = MemoryClient(namespace="project:bbb", mode="local")
        with pytest.raises(MemoryNotFoundError):
            await client_b.forget(stored["memory_id"])


# ---------------------------------------------------------------------------
# search() tests (FR05)
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_returns_all_entries(self, client: MemoryClient) -> None:
        await client.store("entry one", importance=0.3)
        await client.store("entry two", importance=0.7)
        results = await client.search()
        assert len(results) >= 2

    async def test_search_min_importance_filter(self, client: MemoryClient) -> None:
        await client.store("low", importance=0.2)
        await client.store("high", importance=0.8)
        results = await client.search(min_importance=0.5)
        for r in results:
            assert r["importance"] >= 0.5

    async def test_search_tags_filter(self, client: MemoryClient) -> None:
        await client.store("tagged", tags=["python"])
        await client.store("untagged")
        results = await client.search(tags=["python"])
        assert all("python" in r["tags"] for r in results)

    async def test_search_since_filter(self, client: MemoryClient) -> None:
        await client.store("old entry", importance=0.5)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = await client.search(since=future)
        assert len(results) == 0

    async def test_search_limit_validation(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.search(limit=0)

    async def test_search_min_importance_validation(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="min_importance"):
            await client.search(min_importance=-0.1)
        with pytest.raises(ValueError, match="min_importance"):
            await client.search(min_importance=1.5)

    async def test_search_sorted_by_importance_desc(self, client: MemoryClient) -> None:
        await client.store("low", importance=0.2)
        await client.store("mid", importance=0.5)
        await client.store("high", importance=0.9)
        results = await client.search()
        importances = [r["importance"] for r in results]
        assert importances == sorted(importances, reverse=True)


# ---------------------------------------------------------------------------
# Thread safety tests (FR08)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    async def test_concurrent_store_and_recall(self, client: MemoryClient) -> None:
        """Concurrent store + recall should not raise or corrupt data."""

        async def store_batch(prefix: str, count: int) -> list[StoreResultDict]:
            results = []
            for i in range(count):
                r = await client.store(f"{prefix} entry {i}", importance=0.5)
                results.append(r)
            return results

        async def recall_batch(count: int) -> list[list[MemoryResultDict]]:
            results = []
            for _ in range(count):
                r = await client.recall("entry", limit=50)
                results.append(r)
            return results

        # Run concurrent stores and recalls
        store_tasks = [store_batch(f"batch-{j}", 5) for j in range(3)]
        recall_tasks = [recall_batch(3) for _ in range(2)]
        all_results = await asyncio.gather(*store_tasks, *recall_tasks, return_exceptions=True)

        # No exceptions should have been raised
        for result in all_results:
            assert not isinstance(result, Exception), f"Got exception: {result}"


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    async def test_async_context_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ctx"))
        async with MemoryClient(namespace="default", mode="local") as c:
            result = await c.store("context managed")
            assert result["status"] == "stored"
        # After exit, backend should be closed (None)
        assert c._backend is None

    async def test_close_idempotent(self, client: MemoryClient) -> None:
        await client.close()
        await client.close()  # Should not raise

    async def test_async_context_manager_drains_retry_queue(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ctx"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        seed_client = MemoryClient(namespace="default", mode="local")
        with (
            patch("trw_memory.client.publish_memory", return_value=False),
            patch("trw_memory.client._anonymize_entry", return_value={"summary": "queued"}),
        ):
            await seed_client.store("queue this entry", importance=0.9)
            await seed_client.close()

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock(status_code=200)
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client_cls.return_value = mock_client

            async with MemoryClient(namespace="default", mode="local"):
                pass

        reopened = MemoryClient(namespace="default", mode="local")
        assert reopened._retry_queue.depth() == 0
        await reopened.close()


# ---------------------------------------------------------------------------
# register_tools() tests (FR09)
# ---------------------------------------------------------------------------


class TestRegisterTools:
    def test_register_via_register_tool_method(self, client: MemoryClient) -> None:
        agent = MagicMock()
        agent.register_tool = MagicMock()
        # Remove tool attribute so it uses register_tool
        del agent.tool

        client.register_tools(agent)
        assert agent.register_tool.call_count == 4
        call_names = [call.args[0] for call in agent.register_tool.call_args_list]
        assert "memory_store" in call_names
        assert "memory_recall" in call_names
        assert "memory_forget" in call_names
        assert "memory_search" in call_names

    def test_register_via_tool_decorator(self, client: MemoryClient) -> None:
        agent = MagicMock()
        # Remove register_tool so it falls through to tool()
        del agent.register_tool
        inner_dec = MagicMock(side_effect=lambda fn: fn)
        agent.tool = MagicMock(return_value=inner_dec)

        client.register_tools(agent)
        # tool() called once to get the decorator, then decorator called 4 times
        assert agent.tool.call_count == 1
        assert inner_dec.call_count == 4

    def test_double_registration_raises(self, client: MemoryClient) -> None:
        agent = MagicMock()
        agent.register_tool = MagicMock()
        del agent.tool

        client.register_tools(agent)
        with pytest.raises(ToolAlreadyRegisteredError):
            client.register_tools(agent)

    def test_incompatible_agent_raises_type_error(self, client: MemoryClient) -> None:
        agent: Any = object()  # No register_tool or tool
        with pytest.raises(TypeError, match=r"register_tool.*tool"):
            client.register_tools(agent)


# ---------------------------------------------------------------------------
# YAML backend mode test
# ---------------------------------------------------------------------------


class TestAutoRecallDecorator:
    """Tests for the auto_recall decorator fail-open behavior."""

    async def test_auto_recall_injects_memories(self, client: MemoryClient) -> None:
        """When backend works, recalled memories are injected."""
        await client.store("test pattern for recall", importance=0.8)

        @client.auto_recall(query_from="topic", limit=5)
        async def my_func(topic: str, recalled_memories: list[Any] | None = None) -> list[Any]:
            return recalled_memories or []

        result = await my_func(topic="test pattern")
        assert isinstance(result, list)

    async def test_auto_recall_fail_open_on_broken_backend(self, client: MemoryClient) -> None:
        """When backend raises, decorator injects empty list (fail-open)."""
        # Store something first so we know recall would normally work
        await client.store("something", importance=0.5)

        @client.auto_recall(query_from="topic", limit=5)
        async def my_func(topic: str, recalled_memories: list[Any] | None = None) -> list[Any]:
            return recalled_memories or []

        # Close the backend to force an error on recall
        await client.close()
        result = await my_func(topic="anything")
        assert result == []  # fail-open: empty list, no exception

    async def test_auto_recall_missing_query_key(self, client: MemoryClient) -> None:
        """When query_from key is absent, injects empty list."""

        @client.auto_recall(query_from="missing_key", limit=5)
        async def my_func(recalled_memories: list[Any] | None = None) -> list[Any]:
            return recalled_memories or []

        result = await my_func()
        assert result == []


class TestYAMLBackend:
    async def test_store_and_recall_yaml(self, yaml_client: MemoryClient) -> None:
        await yaml_client.store("yaml content", tags=["yaml"])
        results = await yaml_client.recall("yaml content")
        assert len(results) >= 1
        assert results[0]["content"] == "yaml content"
