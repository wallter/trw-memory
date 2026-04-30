from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient, MemoryResultDict
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry


class TestRecall:
    async def test_recall_force_deletes_sqlite_canonical_copy_when_primary_rollback_delete_fails(
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
        cold_file = cold_partition / "archived-entry-sqlite-double-fail.yaml"
        write_yaml(
            cold_file,
            MemoryEntry(
                id="M-cold-sqlite-double-fail",
                content="sqlite canonical cleanup fallback lesson",
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
                results = await client.recall("canonical cleanup fallback", limit=5)
        finally:
            monkeypatch.setattr(Path, "unlink", original_unlink)

        assert not any(result["memory_id"] == "M-cold-sqlite-double-fail" for result in results)
        assert cold_file.exists()
        assert client._get_backend().get("M-cold-sqlite-double-fail") is None
        await client.close()

    async def test_recall_surfaces_semantic_warm_tier_hit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")
        manager = get_tier_manager(client._config, "default")
        manager.warm_add(
            "M-semantic",
            MemoryEntry(
                id="M-semantic",
                content="opaque title",
                detail="vector-only match",
                namespace="default",
                importance=0.9,
                q_value=0.95,
                q_observations=5,
            ).model_dump(mode="json"),
            [1.0, 0.0],
        )

        fake_embedder = MagicMock()
        fake_embedder.embed.return_value = [1.0, 0.0]

        with patch.object(client, "_get_embedder", return_value=fake_embedder):
            results = await client.recall("semantic query", limit=5)

        assert [result["memory_id"] for result in results] == ["M-semantic"]
        await client.close()

    def test_merge_tier_results_reranks_by_composite_score(self) -> None:
        cfg = MemoryConfig()
        local_results: list[MemoryResultDict] = [
            MemoryResultDict(
                memory_id="M-local",
                content="deploy lesson",
                detail="stale local result",
                tags=[],
                importance=0.1,
                score=0.9,
                created_at=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                updated_at=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                last_accessed_at=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                namespace="default",
                source="local",
                q_value=0.1,
                q_observations=5,
                recurrence=1,
                access_count=0,
            )
        ]
        tier_results: list[MemoryResultDict] = [
            MemoryResultDict(
                memory_id="M-tier",
                content="deploy playbook",
                detail="fresh high-value tier result",
                tags=[],
                importance=0.9,
                score=0.2,
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                last_accessed_at=datetime.now(timezone.utc).isoformat(),
                namespace="default",
                source="local",
                q_value=0.95,
                q_observations=5,
                recurrence=1,
                access_count=3,
            )
        ]

        merged = MemoryClient._merge_tier_results(local_results, tier_results, 5, ["deploy"], cfg, None)
        assert [result["memory_id"] for result in merged] == ["M-tier", "M-local"]

    async def test_forget_removes_entry_from_warm_tier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")
        assert client._tier_manager is not None

        with patch.object(client, "_get_embedder", return_value=None):
            stored = await client.store("tracked warm tier entry", importance=0.7)

        warm_ids = [str(item["id"]) for item in client._tier_manager.warm_search(["tracked"], None)]
        assert stored["memory_id"] in warm_ids

        await client.forget(stored["memory_id"])
        warm_ids_after = [str(item["id"]) for item in client._tier_manager.warm_search(["tracked"], None)]
        assert stored["memory_id"] not in warm_ids_after
        await client.close()
