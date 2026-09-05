from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync import SharedFetchResult


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

    async def test_recall_include_org_memories_appends_cross_validated_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="project:default", mode="local")
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "storage"))
        await client.store("deployment lesson", importance=0.7)

        with create_backend_from_config(cfg, "project:other") as storage:
            remote_backend = cast("SQLiteBackend", storage)
            org_entry = MemoryEntry(
                id="M-org",
                content="deployment lesson from another project",
                namespace="project:other",
                importance=0.85,
                cross_validated=True,
            )
            remote_backend.store(org_entry)

            with patch("trw_memory.client.list_org_shared_entries", return_value=[org_entry]):
                results = await client.recall("", include_org_memories=True)

        assert any(result["source"] == "org" for result in results)
        assert results[0]["source"] == "local"
        assert any(result["namespace"] == "project:other" for result in results)
        backend = cast("SQLiteBackend", client._get_backend())
        current_entry = backend.get(results[0]["memory_id"], namespace="project:default")
        with create_backend_from_config(cfg, "project:other") as reopened_storage:
            remote_entry = cast("SQLiteBackend", reopened_storage).get("M-org", namespace="project:other")
        assert current_entry is not None
        assert remote_entry is not None
        assert current_entry.access_count == 1
        assert remote_entry.access_count == 1
        await client.close()

    async def test_recall_include_org_memories_skips_below_threshold_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="project:default", mode="local")
        await client.store("deployment lesson", importance=0.7)

        low_entry = MemoryEntry(
            id="M-org-low",
            content="deployment lesson from another project",
            namespace="project:other",
            importance=0.79,
            cross_validated=True,
        )
        with patch("trw_memory.client.list_org_shared_entries", return_value=[low_entry]):
            results = await client.recall("", include_org_memories=True)

        assert all(result["namespace"] != "project:other" for result in results)
        await client.close()

    async def test_recall_include_org_memories_false_skips_org_lookup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="project:default", mode="local")
        await client.store("deployment lesson", importance=0.7)

        with patch(
            "trw_memory.client.list_org_shared_entries",
            side_effect=AssertionError("org lookup should be skipped"),
        ):
            results = await client.recall("", include_org_memories=False)

        assert all(result["source"] == "local" for result in results)
        await client.close()

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
        monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")
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

        fetched = SharedFetchResult([shared_result], "ok", 1, 0)
        with patch("trw_memory.client.fetch_shared_memories", return_value=fetched) as fetch_mock:
            results = await client.recall("entry", include_shared=True)

        assert fetch_mock.called
        assert any(result["source"] == "shared" for result in results)
        assert results[0]["source"] == "local"
        await client.close()

    async def test_recall_returns_empty_for_expired_team_namespace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="team:sprint-24", mode="local")
        await client.store("team finding", tags=["team"])

        backend = cast("SQLiteBackend", client._get_backend())
        NamespaceManager(backend).mark_team_namespace_completed(
            "team:sprint-24",
            completed_at=datetime.now(timezone.utc) - timedelta(days=2),
        )

        assert await client.recall("team") == []
        await client.close()

    async def test_recall_returns_empty_for_expired_team_namespace_with_yaml(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        client = MemoryClient(namespace="team:sprint-24", mode="local")
        await client.store("team finding", tags=["team"])

        backend = client._get_backend()
        NamespaceManager(backend).mark_team_namespace_completed(
            "team:sprint-24",
            completed_at=datetime.now(timezone.utc) - timedelta(days=2),
        )

        assert await client.recall("team") == []
        await client.close()
