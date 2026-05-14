# ruff: noqa: F401
from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry


class TestRecall:
    async def test_recall_warms_hot_tier_from_persisted_warm_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

        client = MemoryClient(namespace="default", mode="local")
        with patch.object(client, "_get_embedder", return_value=None):
            await client.store("hot cache seed one", importance=0.9)
            await client.store("hot cache seed two", importance=0.8)
        await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        assert reopened._tier_manager is not None
        assert reopened._tier_manager.hot_size >= 2
        await reopened.close()

    async def test_recall_surfaces_cold_tier_entry_and_promotes_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold",
                content="archived deployment lesson",
                detail="cold tier recovery",
                namespace="default",
                tags=["cold"],
            ).model_dump(mode="json"),
        )

        with patch.object(client, "_get_embedder", return_value=None):
            results = await client.recall("archived deployment", limit=5)

        assert any(result["memory_id"] == "M-cold" for result in results)
        assert not cold_file.exists()
        assert client._get_backend().get("M-cold") is not None
        warm_ids = [str(item["id"]) for item in client._tier_manager.warm_search(["archived"], None)]
        assert "M-cold" in warm_ids
        await client.close()

    async def test_recall_restores_cold_hit_into_warm_vector_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        class _FakeWarmBackend:
            def __init__(self) -> None:
                self.vectors: dict[str, list[float]] = {}

            def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
                self.vectors[entry_id] = embedding

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry-vector.yaml"
        payload = MemoryEntry(
            id="M-cold-vector",
            content="keyword promoted vector lesson",
            namespace="default",
            tags=["cold"],
        ).model_dump(mode="json")
        payload["_warm_embedding"] = [1.0, 0.0]
        write_yaml(cold_file, payload)

        fake_backend = _FakeWarmBackend()
        client._tier_manager._warm_store._get_warm_backend = lambda dim=None: fake_backend  # type: ignore[assignment,return-value]
        with patch.object(client, "_get_embedder", return_value=None):
            results = await client.recall("keyword promoted", limit=5)

        assert any(result["memory_id"] == "M-cold-vector" for result in results)
        assert fake_backend.vectors["M-cold-vector"] == [1.0, 0.0]
        await client.close()

    async def test_recall_does_not_surface_cold_hit_when_promotion_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry-fail.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold-fail",
                content="archived rollback lesson",
                detail="cold tier failure",
                namespace="default",
                tags=["cold"],
            ).model_dump(mode="json"),
        )

        original_warm_add = client._tier_manager._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        client._tier_manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            with patch.object(client, "_get_embedder", return_value=None):
                results = await client.recall("archived rollback", limit=5)
        finally:
            client._tier_manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert not any(result["memory_id"] == "M-cold-fail" for result in results)
        assert cold_file.exists()
        assert client._get_backend().get("M-cold-fail") is None
        await client.close()

    async def test_recall_does_not_leave_sqlite_canonical_copy_when_promotion_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry-sqlite-fail.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold-sqlite-fail",
                content="sqlite rollback lesson",
                detail="cold tier sqlite failure",
                namespace="default",
                tags=["cold"],
            ).model_dump(mode="json"),
        )

        original_warm_add = client._tier_manager._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        client._tier_manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            with patch.object(client, "_get_embedder", return_value=None):
                results = await client.recall("sqlite rollback", limit=5)
        finally:
            client._tier_manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert not any(result["memory_id"] == "M-cold-sqlite-fail" for result in results)
        assert cold_file.exists()
        assert client._get_backend().get("M-cold-sqlite-fail") is None
        await client.close()

    async def test_recall_does_not_leave_sqlite_canonical_copy_when_archive_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry-sqlite-unlink-fail.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold-sqlite-unlink-fail",
                content="sqlite archive delete rollback lesson",
                namespace="default",
                tags=["cold"],
            ).model_dump(mode="json"),
        )

        original_unlink = Path.unlink

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == cold_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
        try:
            with patch.object(client, "_get_embedder", return_value=None):
                results = await client.recall("archive delete rollback", limit=5)
        finally:
            monkeypatch.setattr(Path, "unlink", original_unlink)

        assert not any(result["memory_id"] == "M-cold-sqlite-unlink-fail" for result in results)
        assert cold_file.exists()
        assert client._get_backend().get("M-cold-sqlite-unlink-fail") is None
        await client.close()

    async def test_recall_force_deletes_yaml_canonical_copy_when_primary_rollback_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "archived-entry-yaml-double-fail.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold-yaml-double-fail",
                content="yaml canonical cleanup fallback lesson",
                namespace="default",
                tags=["cold"],
            ).model_dump(mode="json"),
        )

        backend = client._get_backend()
        original_unlink = Path.unlink

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == cold_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        def _fail_primary_delete(_entry_id: str) -> bool:
            raise OSError("primary rollback delete failed")

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
        monkeypatch.setattr(backend, "delete", _fail_primary_delete)
        try:
            with patch.object(client, "_get_embedder", return_value=None):
                results = await client.recall("yaml canonical cleanup fallback", limit=5)
        finally:
            monkeypatch.setattr(Path, "unlink", original_unlink)

        assert not any(result["memory_id"] == "M-cold-yaml-double-fail" for result in results)
        assert cold_file.exists()
        assert client._get_backend().get("M-cold-yaml-double-fail") is None
        await client.close()
