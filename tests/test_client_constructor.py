from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests._test_client_support import _RecordingSQLCipherDBAPI
from trw_memory.client import MemoryClient
from trw_memory.exceptions import EncryptionUnavailableError


class TestConstructor:
    async def test_local_mode_creates_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        async with MemoryClient(namespace="default", mode="local") as client:
            assert client.resolved_mode == "local"
            assert client.namespace == "default"

    async def test_auto_mode_resolves_to_local(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        async with MemoryClient(namespace="default", mode="auto") as client:
            assert client.resolved_mode == "local"

    def test_local_mode_raises_when_sqlite_encryption_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")

        with pytest.raises(
            EncryptionUnavailableError,
            match=r"SQLCipher driver not installed",
        ):
            MemoryClient(namespace="default", mode="local")

    async def test_local_mode_uses_sqlcipher_key_first_and_disables_tier_sidecars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statements: list[str] = []
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
        monkeypatch.setenv("MEMORY_MASTER_KEY", "11" * 32)
        monkeypatch.setattr(
            "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
            lambda: _RecordingSQLCipherDBAPI(statements),
        )
        monkeypatch.setattr(MemoryClient, "_get_embedder", lambda self: None)

        async with MemoryClient(namespace="default", mode="local") as client:
            await client.store("encrypted runtime path")
            results = await client.recall("encrypted")

            assert results
            assert results[0]["content"] == "encrypted runtime path"
            assert statements[0].startswith("PRAGMA key = \"x'")
            assert "PRAGMA cipher = 'aes-256-cbc'" in statements
            assert "PRAGMA cipher_page_size = 4096" in statements
            assert "PRAGMA kdf_iter = 256000" in statements
            assert (tmp_path / "storage" / "default" / "memory" / "warm.jsonl").exists() is False

    def test_mcp_mode_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="MCP mode"):
            MemoryClient(namespace="default", mode="mcp")

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported memory client mode"):
            MemoryClient(namespace="default", mode="bogus")  # type: ignore[arg-type]

    def test_invalid_namespace_raises(self) -> None:
        with pytest.raises(Exception):
            MemoryClient(namespace="invalid namespace!")

    async def test_project_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        async with MemoryClient(namespace="project:test-proj", mode="local") as client:
            assert client.namespace == "project:test-proj"
            assert client.resolved_mode == "local"

    async def test_timeout_stored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        async with MemoryClient(namespace="default", mode="local", timeout=10.0) as client:
            assert client._timeout == 10.0

    async def test_explicit_db_path_anchors_ancillary_state_outside_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        db_path = tmp_path / "isolated" / "memory.db"
        monkeypatch.chdir(cwd)

        client = MemoryClient(namespace="project:explicit-path", mode="local", db_path=db_path)
        try:
            assert client._config.storage_path == str(db_path.parent.resolve())
            assert (cwd / ".memory").exists() is False
            assert client._tier_manager is not None
            assert client._tier_manager._base_dir == db_path.parent.resolve() / "project_explicit-path"
        finally:
            await client.close()

    async def test_sync_enabled_starts_sse_subscription(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")

        with patch("trw_memory.client.SSESubscriber") as subscriber_cls:
            subscriber = MagicMock()
            subscriber_cls.return_value = subscriber

            client = MemoryClient(namespace="default", mode="local")

        try:
            subscriber_cls.assert_called_once()
            assert client._sse_subscriber is subscriber
            subscriber.start.assert_called_once()
        finally:
            await client.close()


class TestAutoRecallDecorator:
    async def test_auto_recall_injects_memories(self, client: MemoryClient) -> None:
        await client.store("test pattern for recall", importance=0.8)

        @client.auto_recall(query_from="topic", limit=5)
        async def my_func(topic: str, recalled_memories: list[Any] | None = None) -> list[Any]:
            return recalled_memories or []

        result = await my_func(topic="test pattern")
        assert isinstance(result, list)

    async def test_auto_recall_fail_open_on_broken_backend(self, client: MemoryClient) -> None:
        await client.store("something", importance=0.5)

        @client.auto_recall(query_from="topic", limit=5)
        async def my_func(topic: str, recalled_memories: list[Any] | None = None) -> list[Any]:
            return recalled_memories or []

        await client.close()
        result = await my_func(topic="anything")
        assert result == []

    async def test_auto_recall_missing_query_key(self, client: MemoryClient) -> None:
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
