"""Tests for lifecycle/tiers.py cold archive and promote workflows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers import TierManager

from ._test_tiers_support import mem_dir, mgr  # noqa: F401


class TestColdTier:
    def test_cold_archive_moves_file(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "e1-test.yaml"
        write_yaml(entry_file, {"id": "e1", "content": "test"})

        mgr.cold_archive("e1", entry_file)

        assert not entry_file.exists()
        yaml_files = list(mgr._cold_dir().rglob("*.yaml"))
        assert len(yaml_files) == 1
        assert read_yaml(yaml_files[0])["id"] == "e1"

    def test_cold_archive_uses_entry_created_partition(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "dated-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "e-created",
                "content": "test",
                "created_at": "2024-03-15T12:00:00+00:00",
            },
        )

        mgr.cold_archive("e-created", entry_file)

        assert (mgr._cold_dir() / "2024" / "03" / "dated-entry.yaml").exists()

    def test_cold_archive_raises_on_missing_file(self, mgr: TierManager, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            mgr.cold_archive("bad", tmp_path / "nonexistent.yaml")

    def test_cold_archive_rolls_back_when_warm_cleanup_fails(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        entry_file = entries_dir / "e-warm-fail.yaml"
        write_yaml(entry_file, {"id": "e-warm-fail", "content": "test"})

        original_warm_remove = mgr._warm_store.warm_remove

        def _fail_warm_remove(_entry_id: str) -> bool:
            return False

        mgr._warm_store.warm_remove = _fail_warm_remove  # type: ignore[assignment]
        try:
            with pytest.raises(StorageError):
                mgr.cold_archive("e-warm-fail", entry_file)
        finally:
            mgr._warm_store.warm_remove = original_warm_remove  # type: ignore[method-assign]

        assert entry_file.exists()
        assert not any(path.name == "e-warm-fail.yaml" for path in mgr._cold_dir().rglob("*.yaml"))

    def test_cold_promote_finds_entry(self, mgr: TierManager, mem_dir: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e1-promote.yaml"
        write_yaml(yaml_file, {"id": "e1", "content": "promote me", "tags": []})

        result = mgr.cold_promote("e1")
        assert result is not None
        assert result["id"] == "e1"
        assert not yaml_file.exists()
        assert (mem_dir / "entries" / "e1.yaml").exists()

    def test_cold_promote_returns_none_if_not_found(self, mgr: TierManager) -> None:
        assert mgr.cold_promote("nonexistent-id") is None

    def test_cold_promote_updates_last_accessed(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e2.yaml"
        old_time = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e2", "content": "old entry", "last_accessed_at": old_time})

        result = mgr.cold_promote("e2")
        assert result is not None
        assert result.get("last_accessed_at") != old_time

    def test_cold_promote_keeps_archive_when_restore_fails(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e3.yaml"
        write_yaml(yaml_file, {"id": "e3", "content": "cold entry"})

        def _fail_restore(_entry_data: dict[str, object]) -> None:
            raise OSError("backend unavailable")

        assert mgr.cold_promote("e3", restore_entry_fn=_fail_restore) is None
        assert yaml_file.exists()

    def test_cold_promote_rolls_back_restore_when_warm_add_fails(
        self,
        mgr: TierManager,
        mem_dir: Path,
    ) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e4.yaml"
        original_last_accessed = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e4", "content": "cold entry", "last_accessed_at": original_last_accessed})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote("e4")
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e4.yaml").exists()
        assert read_yaml(yaml_file)["last_accessed_at"] == original_last_accessed

    def test_cold_promote_rolls_back_warm_and_canonical_when_archive_delete_fails(
        self,
        mgr: TierManager,
        mem_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e5.yaml"
        write_yaml(yaml_file, {"id": "e5", "content": "rollback me"})

        original_unlink = Path.unlink

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == yaml_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)

        result = mgr.cold_promote("e5")

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e5.yaml").exists()
        assert mgr.warm_search(["rollback"], None, top_k=5) == []

    def test_cold_promote_suppresses_rollback_delete_failure(
        self,
        mgr: TierManager,
        mem_dir: Path,
    ) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e6.yaml"
        original_last_accessed = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        write_yaml(yaml_file, {"id": "e6", "content": "cold entry", "last_accessed_at": original_last_accessed})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        def _restore(entry_data: dict[str, object]) -> None:
            write_yaml(mem_dir / "entries" / "e6.yaml", entry_data)

        def _fail_rollback(_entry_id: str) -> bool:
            raise RuntimeError("rollback unavailable")

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote(
                "e6",
                restore_entry_fn=_restore,
                delete_restored_entry_fn=_fail_rollback,
            )
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert read_yaml(yaml_file)["last_accessed_at"] == original_last_accessed
        assert not (mem_dir / "entries" / "e6.yaml").exists()

    def test_cold_promote_force_deletes_restored_entry_when_primary_rollback_fails(
        self,
        mgr: TierManager,
        mem_dir: Path,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e6b.yaml"
        write_yaml(yaml_file, {"id": "e6b", "content": "cold entry"})

        original_warm_add = mgr._cold_store._warm_store.warm_add

        def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("warm unavailable")

        def _restore(entry_data: dict[str, object]) -> None:
            write_yaml(mem_dir / "entries" / "e6b.yaml", entry_data)

        def _fail_rollback(_entry_id: str) -> bool:
            raise RuntimeError("rollback unavailable")

        def _force_delete(entry_id: str) -> bool:
            (mem_dir / "entries" / f"{entry_id}.yaml").unlink(missing_ok=True)
            return True

        def _verify_removed(entry_id: str) -> bool:
            return not (mem_dir / "entries" / f"{entry_id}.yaml").exists()

        mgr._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
        try:
            result = mgr.cold_promote(
                "e6b",
                restore_entry_fn=_restore,
                delete_restored_entry_fn=_fail_rollback,
                force_delete_restored_entry_fn=_force_delete,
                verify_restored_entry_removed_fn=_verify_removed,
            )
        finally:
            mgr._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert not (mem_dir / "entries" / "e6b.yaml").exists()
