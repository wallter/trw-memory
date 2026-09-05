from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import memory_recall_impl


class TestMemoryRecallImpl:
    def test_recall_promotes_cold_tier_hit_through_tool_surface(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold",
                    content="tool cold archive lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            result = memory_recall_impl(
                "tool cold archive",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-cold" for memory in memories)
            assert not cold_file.exists()
            assert backend.get("M-tool-cold", namespace="project:default") is not None
            warm_ids = [str(item["id"]) for item in manager.warm_search(["tool"], None)]
            assert "M-tool-cold" in warm_ids

    def test_recall_restores_cold_hit_into_warm_vector_index(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        class _FakeWarmBackend:
            def __init__(self) -> None:
                self.vectors: dict[str, list[float]] = {}

            def upsert_vector(self, entry_id: str, embedding: list[float], *, namespace: str = "default") -> None:
                self.vectors[entry_id] = embedding

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-vector.yaml"
            payload = MemoryEntry(
                id="M-tool-cold-vector",
                content="tool keyword promoted vector lesson",
                namespace="project:default",
                tags=["cold"],
            ).model_dump(mode="json")
            payload["_warm_embedding"] = [1.0, 0.0]
            write_yaml(cold_file, payload)

            fake_backend = _FakeWarmBackend()
            manager._warm_store._get_warm_backend = lambda dim=None: fake_backend  # type: ignore[assignment,return-value]
            result = memory_recall_impl(
                "keyword promoted",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-cold-vector" for memory in memories)
            assert fake_backend.vectors["M-tool-cold-vector"] == [1.0, 0.0]

    def test_recall_does_not_surface_cold_hit_when_promotion_fails(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-fail",
                    content="tool cold rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_warm_add = manager._cold_store._warm_store.warm_add

            def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
                raise OSError("warm unavailable")

            manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
            try:
                result = memory_recall_impl(
                    "tool cold rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-fail", namespace="default") is None

    def test_recall_does_not_leave_sqlite_canonical_copy_when_promotion_fails(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-fail",
                    content="tool sqlite rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_warm_add = manager._cold_store._warm_store.warm_add

            def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
                raise OSError("warm unavailable")

            manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
            try:
                result = memory_recall_impl(
                    "tool sqlite rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-fail", namespace="default") is None

    def test_recall_does_not_leave_sqlite_canonical_copy_when_archive_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-unlink-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-unlink-fail",
                    content="tool sqlite archive delete rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False, **_kwargs) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            try:
                result = memory_recall_impl(
                    "archive delete rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-unlink-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-unlink-fail", namespace="default") is None

    def test_recall_force_deletes_yaml_canonical_copy_when_primary_rollback_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-yaml-double-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-yaml-double-fail",
                    content="tool yaml canonical cleanup fallback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False, **_kwargs) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            def _fail_primary_delete(_entry_id: str, **_kwargs) -> bool:
                raise OSError("primary rollback delete failed")

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            monkeypatch.setattr(backend, "delete", _fail_primary_delete)
            try:
                result = memory_recall_impl(
                    "yaml canonical cleanup fallback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-yaml-double-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-yaml-double-fail", namespace="default") is None

    def test_recall_force_deletes_sqlite_canonical_copy_when_primary_rollback_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-double-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-double-fail",
                    content="tool sqlite canonical cleanup fallback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False, **_kwargs) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            def _fail_primary_delete(_entry_id: str, **_kwargs) -> bool:
                raise OSError("primary rollback delete failed")

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            monkeypatch.setattr(backend, "delete", _fail_primary_delete)
            try:
                result = memory_recall_impl(
                    "canonical cleanup fallback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-double-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-double-fail", namespace="default") is None
