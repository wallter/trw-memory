from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trw_memory.client import MemoryClient, MemoryResultDict, StoreResultDict
from trw_memory.exceptions import ToolAlreadyRegisteredError


class TestThreadSafety:
    async def test_concurrent_store_and_recall(self, client: MemoryClient) -> None:
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

        store_tasks = [store_batch(f"batch-{j}", 5) for j in range(3)]
        recall_tasks = [recall_batch(3) for _ in range(2)]
        all_results = await asyncio.gather(*store_tasks, *recall_tasks, return_exceptions=True)

        for result in all_results:
            assert not isinstance(result, Exception), f"Got exception: {result}"


class TestContextManager:
    async def test_async_context_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ctx"))
        async with MemoryClient(namespace="default", mode="local") as c:
            result = await c.store("context managed")
            assert result["status"] == "stored"
        assert c._backend is None

    async def test_close_idempotent(self, client: MemoryClient) -> None:
        await client.close()
        await client.close()

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
            patch(
                "trw_memory.client.publish_memory_result",
                return_value={"success": False, "remote_id": None, "retryable": True},
            ),
            patch(
                "trw_memory.client._anonymize_entry",
                return_value={"summary": "queued", "source_learning_id": "queued-entry"},
            ),
        ):
            await seed_client.store("queue this entry", importance=0.9, entry_id="queued-entry")
            await seed_client.close()

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock(status_code=200)
            mock_response.json.return_value = {"id": "42"}
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client_cls.return_value = mock_client

            async with MemoryClient(namespace="default", mode="local"):
                pass

        reopened = MemoryClient(namespace="default", mode="local")
        assert reopened._retry_queue.depth() == 0
        drained_entry = reopened._get_backend().list_entries(limit=10)[0]
        assert drained_entry.published_to_platform is True
        assert drained_entry.remote_id == "42"
        await reopened.close()

    async def test_retry_queue_recovery_preserves_remote_id_for_later_retire(
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
            patch(
                "trw_memory.client.publish_memory_result",
                return_value={"success": False, "remote_id": None, "retryable": True},
            ),
            patch(
                "trw_memory.client._anonymize_entry",
                return_value={"summary": "queued", "source_learning_id": "queued-entry"},
            ),
        ):
            await seed_client.store("queue this entry", importance=0.9, entry_id="queued-entry")
            await seed_client.close()

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock(status_code=200)
            mock_response.json.return_value = {"id": "42"}
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client_cls.return_value = mock_client

            async with MemoryClient(namespace="default", mode="local"):
                pass

        reopened = MemoryClient(namespace="default", mode="local")
        with patch("trw_memory.client.retire_remote_memory", return_value=True) as retire_mock:
            await reopened.forget("queued-entry")
            await reopened.close()

        retire_mock.assert_called_once_with("42", reopened._config)


class TestRegisterTools:
    def test_register_via_register_tool_method(self, client: MemoryClient) -> None:
        agent = MagicMock()
        agent.register_tool = MagicMock()
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
        del agent.register_tool
        inner_dec = MagicMock(side_effect=lambda fn: fn)
        agent.tool = MagicMock(return_value=inner_dec)

        client.register_tools(agent)
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
        agent: Any = object()
        with pytest.raises(TypeError, match=r"register_tool.*tool"):
            client.register_tools(agent)

    @pytest.mark.asyncio
    async def test_memory_recall_tool_wrapper_forwards_source_aware_args(self, client: MemoryClient) -> None:
        tools = client._make_tool_functions()
        recall_mock = AsyncMock(return_value=[])
        client.recall = recall_mock  # type: ignore[method-assign]

        await tools["memory_recall"](
            query="source aware",
            limit=7,
            include_org_memories=False,
            include_shared=True,
            include_distilled=False,
            distilled_weight=0.4,
            include_source_kinds=["instruction_rule"],
            exclude_source_kinds=["episodic"],
            source_weights={"instruction_rule": 1.2},
            exclude_expired=False,
        )

        recall_mock.assert_awaited_once_with(
            "source aware",
            limit=7,
            include_org_memories=False,
            include_shared=True,
            include_distilled=False,
            distilled_weight=0.4,
            include_source_kinds=["instruction_rule"],
            exclude_source_kinds=["episodic"],
            source_weights={"instruction_rule": 1.2},
            exclude_expired=False,
        )
