from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import SHARED_EVENT_CACHE_MAX, MemoryClient


class TestRecall:
    async def test_recall_surfaces_cached_sse_publish(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        with patch("trw_memory.client.SSESubscriber"):
            client = MemoryClient(namespace="default", mode="local")

        client._handle_sse_event({"type": "learning_published", "id": 42, "summary": "deployment rollback guide"})

        with patch("trw_memory.client.fetch_shared_memories", return_value=[]):
            results = await client.recall("deployment", include_shared=True)

        assert any(result["memory_id"] == "42" and result["source"] == "shared" for result in results)
        await client.close()

    async def test_recall_dedupes_cached_sse_publish_by_similarity(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")

        with patch("trw_memory.client.SSESubscriber"):
            client = MemoryClient(namespace="default", mode="local")

        await client.store("Local deployment advice", importance=0.5)
        client._handle_sse_event({"type": "learning_published", "id": 42, "summary": "Remote deployment guidance"})

        embedder = MagicMock()
        embedder.available.return_value = True
        embedder.embed_batch.return_value = [[1.0, 0.0], [0.99, 0.01]]

        with (
            patch("trw_memory.client.fetch_shared_memories", return_value=[]),
            patch.object(client, "_get_embedder", return_value=embedder),
        ):
            results = await client.recall("deployment", include_shared=True)

        assert all(result["memory_id"] != "42" for result in results)
        await client.close()

    async def test_cached_sse_events_are_bounded(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        with patch("trw_memory.client.SSESubscriber"):
            client = MemoryClient(namespace="default", mode="local")

        for idx in range(SHARED_EVENT_CACHE_MAX + 20):
            client._handle_sse_event({"type": "learning_published", "id": idx, "summary": f"entry {idx}"})

        assert len(client._shared_event_cache) == SHARED_EVENT_CACHE_MAX
        assert client._shared_event_cache[0]["memory_id"] == "20"
        assert client._shared_event_cache[-1]["memory_id"] == str(SHARED_EVENT_CACHE_MAX + 19)
        await client.close()

    async def test_recall_marks_remote_retirements_pending_delete(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        with patch("trw_memory.client.SSESubscriber"):
            client = MemoryClient(namespace="default", mode="local")

        with patch(
            "trw_memory.client.publish_memory_result",
            return_value={"success": True, "remote_id": "42", "retryable": False},
        ):
            stored = await client.store("published memory", importance=0.9)
            if client._background_tasks:
                await asyncio.gather(*list(client._background_tasks))

        client._handle_sse_event({"type": "learning_retired", "id": 42})

        with patch("trw_memory.client.fetch_shared_memories", return_value=[]):
            await client.recall("published", include_shared=True)

        entry = client._get_backend().get(stored["memory_id"])
        assert entry is not None
        assert entry.pending_delete is True
        await client.close()
