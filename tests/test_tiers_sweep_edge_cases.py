# ruff: noqa: F811
"""Tests for lifecycle/tiers.py sweep and warm keyword edge cases."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models.config import MemoryConfig

from ._test_tiers_support import cfg, mem_dir, mgr  # noqa: F401


class TestSweepEdgeCases:
    """FR04: sweep() edge cases -- hot->warm, warm->cold, cold->purge."""

    def test_sweep_warm_to_cold_archival(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        entry_file = entries_dir / "archive-me.yaml"
        write_yaml(
            entry_file,
            {
                "id": "archive-me",
                "content": "stale warm",
                "importance": 0.05,
                "status": "active",
                "last_accessed_at": (
                    datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 20)
                ).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not entry_file.exists()
        assert result.demoted >= 1
        assert list(mgr._cold_dir().rglob("*.yaml"))

    def test_sweep_cold_to_purge(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        cold_file = cold_partition / "purge-me.yaml"
        write_yaml(
            cold_file,
            {
                "id": "purge-me",
                "content": "expired entry",
                "importance": 0.01,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 100)).isoformat(),
                "tags": [],
            },
        )

        result = mgr.sweep()
        assert not cold_file.exists()
        assert result.purged >= 1

    def test_sweep_purge_writes_audit_jsonl(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2023" / "05"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "audit-test.yaml",
            {
                "id": "audit-entry",
                "content": "will be purged",
                "importance": 0.02,
                "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=cfg.retention_days + 200)).isoformat(),
                "tags": [],
            },
        )

        mgr.sweep()

        audit_path = mem_dir / "memory" / "purge_audit.jsonl"
        assert audit_path.exists()
        lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["entry_id"] == "audit-entry"
        assert "purged_at" in record
        assert "days_idle" in record

    def test_sweep_error_handling(self, mgr: TierManager, mem_dir: Path) -> None:
        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir
        (entries_dir / "bad-data.yaml").write_bytes(b"\x00\x01\x02 not yaml at all")

        result = mgr.sweep()
        assert result.errors >= 1

    def test_sweep_no_entries_to_process(self, mgr: TierManager) -> None:
        result = mgr.sweep()
        assert result.promoted == 0
        assert result.demoted == 0
        assert result.purged == 0
        assert result.errors == 0
        assert result.total == 0

    def test_sweep_warm_to_cold_writes_yaml(self, mgr: TierManager, mem_dir: Path, cfg: MemoryConfig) -> None:
        from trw_memory.storage.persistence import read_yaml, write_yaml

        entries_dir = mem_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        mgr._entries_dir = entries_dir

        write_yaml(
            entries_dir / "cold-check.yaml",
            {
                "id": "cold-check",
                "content": "verify cold write",
                "importance": 0.05,
                "status": "active",
                "last_accessed_at": (
                    datetime.now(timezone.utc) - timedelta(days=cfg.cold_threshold_days + 30)
                ).isoformat(),
                "tags": [],
            },
        )

        mgr.sweep()

        cold_ids = [read_yaml(path).get("id") for path in mgr._cold_dir().rglob("*.yaml")]
        assert "cold-check" in cold_ids


class TestWarmKeywordSearchEdgeCases:
    """FR04: _warm_keyword_search edge cases."""

    def test_warm_keyword_search_malformed_jsonl(self, mgr: TierManager) -> None:
        sidecar = mgr._warm_sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            "\n".join(
                [
                    '{"id": "valid-1", "summary": "python coding", "tags": []}',
                    "this is not valid json at all",
                    '{"id": "valid-2", "summary": "java programming", "tags": []}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        results = mgr.warm_search(["python"], None)
        ids = [result["id"] for result in results]
        assert "valid-1" in ids
        assert "valid-2" not in ids

    def test_warm_keyword_search_empty_sidecar(self, mgr: TierManager) -> None:
        sidecar = mgr._warm_sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("", encoding="utf-8")

        assert mgr.warm_search(["anything"], None) == []

    def test_warm_keyword_search_zero_match(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "alpha beta", "tags": []}, None)
        assert mgr.warm_search(["zzz_nonexistent_token"], None) == []
