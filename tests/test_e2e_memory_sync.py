"""E2E sync wiring tests for trw-memory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient


class TestSyncE2E:
    """Verify MemoryClient wires the package sync surface end-to-end."""

    @staticmethod
    def _mock_httpx_client(
        mock_client_cls: MagicMock,
        *,
        status_code: int,
        json_data: object | None = None,
    ) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data if json_data is not None else []
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client_cls.return_value = mock_client
        return mock_client

    async def test_store_sync_success_marks_entry_published(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_sync"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        client = MemoryClient(namespace="default", mode="local")
        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            self._mock_httpx_client(mock_client_cls, status_code=200)
            stored = await client.store("syncable entry", importance=0.9)
            await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        entry = reopened._get_backend().get(stored["memory_id"])
        assert entry is not None
        assert entry.published_to_platform is True
        await reopened.close()

    async def test_recall_include_shared_merges_remote_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_sync"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        client = MemoryClient(namespace="default", mode="local")
        await client.store("local entry", importance=0.8)

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            self._mock_httpx_client(
                mock_client_cls,
                status_code=200,
                json_data=[{"summary": "remote shared entry", "impact": 0.6}],
            )
            results = await client.recall("entry", include_shared=True)

        assert any(result["source"] == "shared" for result in results)
        assert results[0]["source"] == "local"
        await client.close()
