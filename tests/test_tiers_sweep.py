# ruff: noqa: F811
"""Tests for lifecycle/tiers.py sweep behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers import TierManager, TierSweepResult
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

from ._test_tiers_support import _make_entry, cfg, mem_dir, mgr  # noqa: F401


class TestSweep:
    def test_sweep_returns_tier_sweep_result(self, mgr: TierManager) -> None:
        assert isinstance(mgr.sweep(), TierSweepResult)

    def test_sweep_empty_dirs_returns_zeros(self, mgr: TierManager) -> None:
        result = mgr.sweep()
        assert result.promoted == 0
        assert result.demoted == 0
        assert result.purged == 0
        assert result.errors == 0

    def test_sweep_demotes_stale_hot_entry(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        mgr.hot_put("stale", _make_entry("stale", days_old=cfg.hot_ttl_days + 5))
        assert mgr.hot_size == 1
        result = mgr.sweep()
        assert mgr.hot_get("stale") is None
        assert result.demoted >= 1

    def test_sweep_keeps_fresh_hot_entry(self, mgr: TierManager) -> None:
        mgr.hot_put("fresh", _make_entry("fresh", days_old=1))
        result = mgr.sweep()
        assert mgr.hot_get("fresh") is not None
        assert result.demoted == 0

    def test_sweep_hot_to_warm_failure_keeps_entry_in_hot(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        mgr.hot_put("stale-hot", _make_entry("stale-hot", days_old=cfg.hot_ttl_days + 5))

        original_warm_add = mgr.warm_add

        def _fail_once(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("disk full")

        mgr.warm_add = _fail_once  # type: ignore[method-assign]
        try:
            result = mgr.sweep()
        finally:
            mgr.warm_add = original_warm_add  # type: ignore[method-assign]

        assert result.errors == 1
        assert mgr.hot_get("stale-hot") is not None

    def test_sweep_demotes_warm_to_cold(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        entry_file = entries_dir / "old-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "old-entry",
                "content": "ancient knowledge",
                "importance": 0.1,
                "status": "active",
                "last_accessed_at": (
                    datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
                ).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not entry_file.exists()
        assert result.demoted >= 1

    def test_sweep_demotes_sqlite_warm_entry_to_cold(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        sqlite_cfg = cfg.model_copy(update={"storage_backend": "sqlite"})
        mgr.update_config(sqlite_cfg)

        old_time = datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
        entry = MemoryEntry(
            id="old-sqlite-entry",
            content="ancient sqlite knowledge",
            importance=0.1,
            namespace="default",
            last_accessed_at=old_time,
        )
        with SQLiteBackend(mem_dir / sqlite_cfg.sqlite_db_name, dim=sqlite_cfg.embedding_dim) as backend:
            backend.store(entry)

        mgr.warm_add(entry.id, entry.model_dump(mode="json"), [1.0, 0.0])

        result = mgr.sweep(config=sqlite_cfg)

        cold_file = mgr._cold_dir() / str(entry.created_at.year) / f"{entry.created_at.month:02d}" / f"{entry.id}.yaml"
        assert cold_file.exists()
        assert result.demoted >= 1
        with SQLiteBackend(mem_dir / sqlite_cfg.sqlite_db_name, dim=sqlite_cfg.embedding_dim) as backend:
            assert backend.get(entry.id) is None

    def test_sweep_warm_to_cold_skips_non_active(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        entry_file = entries_dir / "resolved-entry.yaml"
        write_yaml(
            entry_file,
            {
                "id": "resolved-entry",
                "content": "resolved",
                "importance": 0.1,
                "status": "resolved",
                "last_accessed_at": (
                    datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
                ).isoformat(),
                "tags": [],
            },
        )

        mgr.sweep()
        assert entry_file.exists()

    def test_sweep_purges_expired_cold_entry(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "expired.yaml"
        write_yaml(
            cold_file,
            {
                "id": "expired-entry",
                "content": "expired",
                "importance": 0.05,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not cold_file.exists()
        assert result.purged >= 1

    def test_sweep_keeps_high_importance_cold_entry(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "important.yaml"
        write_yaml(
            cold_file,
            {
                "id": "important-entry",
                "content": "important",
                "importance": 0.9,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 10)).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert cold_file.exists()
        assert result.purged == 0

    def test_sweep_error_in_entry_increments_errors(self, mgr: TierManager, mem_dir: Path) -> None:
        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir
        (entries_dir / "corrupt.yaml").write_bytes(b"\xff\xfe invalid yaml !!!")

        result = mgr.sweep()
        assert result.errors >= 1

    def test_sweep_writes_purge_audit_log(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "old.yaml",
            {
                "id": "purge-audit-e",
                "content": "purge me",
                "importance": 0.05,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 50)).isoformat(),
                "tags": [],
            },
        )

        mgr.sweep()

        audit_path = mgr._base_dir / "memory" / "purge_audit.jsonl"
        assert audit_path.exists()
        lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) >= 1
        assert json.loads(lines[0])["entry_id"] == "purge-audit-e"

    def test_sweep_uses_call_time_config_thresholds(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        entry_file = entries_dir / "override-threshold.yaml"
        write_yaml(
            entry_file,
            {
                "id": "override-threshold",
                "content": "aged but still important",
                "importance": 0.8,
                "status": "active",
                "last_accessed_at": (
                    datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 10)
                ).isoformat(),
                "tags": [],
            },
        )

        override_cfg = MemoryConfig(
            hot_max_entries=cfg.hot_max_entries,
            hot_ttl_days=cfg.hot_ttl_days,
            cold_threshold_days=cfg.cold_threshold_days,
            retention_days=cfg.retention_days,
            warm_archive_max_score=0.3,
            cold_purge_max_score=cfg.cold_purge_max_score,
        )
        result = mgr.sweep(config=override_cfg)
        assert result.demoted >= 1
        assert not entry_file.exists()

    def test_sweep_reads_hot_ttl_from_environment(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_HOT_TTL_DAYS", "1")
        mgr.hot_put("env-hot", _make_entry("env-hot", days_old=2))

        result = mgr.sweep(config=MemoryConfig())

        assert result.demoted >= 1
        assert mgr.hot_get("env-hot") is None

    def test_sweep_reads_retention_days_from_environment(
        self,
        mgr: TierManager,
        cfg: MemoryConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        monkeypatch.setenv("MEMORY_RETENTION_DAYS", "30")
        cold_partition = mgr._cold_dir() / "2024" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "env-retention.yaml"
        write_yaml(
            cold_file,
            {
                "id": "env-retention-entry",
                "content": "purge me via env",
                "importance": 0.05,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep(config=MemoryConfig(cold_purge_max_score=cfg.cold_purge_max_score))

        assert result.purged >= 1
        assert not cold_file.exists()
