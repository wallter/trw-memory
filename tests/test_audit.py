"""Tests for the PRD-INFRA-020 audit log contract."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.security import runtime as security_runtime
from trw_memory.security.audit import AuditLog, AuditRecord, audit_verify


@pytest.fixture()
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture()
def audit_log(audit_path: Path) -> AuditLog:
    return AuditLog(audit_path)


class TestAuditRecord:
    def test_default_fields_match_prd_contract(self) -> None:
        record = AuditRecord(op="store")
        assert record.op == "store"
        assert record.id == ""
        assert record.actor == ""
        assert record.prev_hash == "0" * 64
        assert record.hash == ""


class TestAuditLogAppend:
    def test_genesis_record_uses_zero_hash(self, audit_log: AuditLog) -> None:
        record = audit_log.append("store", entry_id="M-001")
        assert record.prev_hash == "0" * 64
        assert len(record.hash) == 64

    def test_second_record_chains_from_first(self, audit_log: AuditLog) -> None:
        first = audit_log.append("store", entry_id="M-001")
        second = audit_log.append("forget", entry_id="M-001")
        assert second.prev_hash == first.hash

    def test_append_persists_prd_field_names(self, audit_log: AuditLog, audit_path: Path) -> None:
        audit_log.append("store", entry_id="M-001", actor="alice", namespace="project:default", data={"stored": True})
        payload = json.loads(audit_path.read_text(encoding="utf-8").strip())
        assert sorted(payload) == ["actor", "data", "hash", "id", "namespace", "op", "prev_hash", "ts"]
        assert payload["op"] == "store"
        assert payload["id"] == "M-001"


class TestSecurityMaintenanceQueue:
    def test_security_maintenance_can_defer_to_bounded_queue(self, tmp_path: Path) -> None:
        security_runtime._AUDIT_MAINTENANCE_CACHE.clear()
        security_runtime._AUDIT_MAINTENANCE_QUEUE.clear()
        cfg = MemoryConfig(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            security_maintenance_inline=False,
        )

        security_runtime.ensure_security_maintenance(cfg)

        status = security_runtime.security_maintenance_status()
        assert status["bounded"] is True
        assert status["queued"] == 1

        drained = security_runtime.drain_security_maintenance_queue(cfg)
        assert drained == {"drained": 1, "queued": 0}

    def test_security_maintenance_preserves_other_config_queue_items(self, tmp_path: Path) -> None:
        security_runtime._AUDIT_MAINTENANCE_CACHE.clear()
        security_runtime._AUDIT_MAINTENANCE_QUEUE.clear()
        first = MemoryConfig(
            audit_log_path=str(tmp_path / "first.jsonl"),
            security_maintenance_inline=False,
        )
        second = MemoryConfig(
            audit_log_path=str(tmp_path / "second.jsonl"),
            security_maintenance_inline=False,
        )

        security_runtime.ensure_security_maintenance(first)
        security_runtime.ensure_security_maintenance(second)
        drained = security_runtime.drain_security_maintenance_queue(first)

        assert drained == {"drained": 1, "queued": 1}
        assert security_runtime.drain_security_maintenance_queue(second) == {"drained": 1, "queued": 0}

    def test_security_maintenance_enqueue_is_thread_safe_and_deduplicated(self, tmp_path: Path) -> None:
        security_runtime._AUDIT_MAINTENANCE_CACHE.clear()
        security_runtime._AUDIT_MAINTENANCE_QUEUE.clear()
        cfg = MemoryConfig(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            security_maintenance_inline=False,
        )
        barrier = threading.Barrier(8)

        def _enqueue() -> None:
            barrier.wait()
            security_runtime.ensure_security_maintenance(cfg)

        threads = [threading.Thread(target=_enqueue) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert security_runtime.security_maintenance_status()["queued"] == 1
        assert security_runtime.drain_security_maintenance_queue(cfg) == {"drained": 1, "queued": 0}


class TestAuditLogVerify:
    def test_verify_chain_missing_file_returns_empty_valid_result(self, tmp_path: Path) -> None:
        result = AuditLog(tmp_path / "missing.jsonl").verify_chain()
        assert result == {"valid": True, "entries_checked": 0, "first_broken_at": None, "broken_hash": None}

    def test_verify_chain_returns_structured_result(self, audit_log: AuditLog) -> None:
        audit_log.append("store", entry_id="M-001")
        result = audit_log.verify_chain()
        assert result == {"valid": True, "entries_checked": 1, "first_broken_at": None, "broken_hash": None}

    def test_verify_chain_detects_tampering(self, audit_log: AuditLog, audit_path: Path) -> None:
        audit_log.append("store", entry_id="M-001")
        audit_log.append("forget", entry_id="M-001")

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        tampered = json.loads(lines[1])
        tampered["data"] = {"stored": False}
        lines[1] = json.dumps(tampered)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = AuditLog(audit_path).verify_chain()
        assert result["valid"] is False
        assert result["first_broken_at"] == 2
        assert result["broken_hash"] == tampered["hash"]

    def test_audit_verify_wrapper_matches_prd_contract(self, audit_log: AuditLog, audit_path: Path) -> None:
        audit_log.append("store", entry_id="M-001")
        assert audit_verify(audit_path) == {
            "valid": True,
            "entries_checked": 1,
            "first_broken_at": None,
            "broken_hash": None,
        }

    def test_compact_rechains_retained_suffix(self, audit_log: AuditLog, audit_path: Path) -> None:
        for index in range(3):
            audit_log.append("store", entry_id=f"M-{index}")
        lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        for index in range(2):
            lines[index]["ts"] = stale_ts
        audit_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

        retained = audit_log.compact(retention_days=365)
        records = audit_log.read_all()
        assert retained == 1
        assert [record.id for record in records] == ["M-2"]
        assert records[0].prev_hash == "0" * 64
        assert audit_log.verify_chain()["valid"] is True

    def test_compact_keeps_recent_records_when_nothing_expires(self, audit_log: AuditLog) -> None:
        for index in range(3):
            audit_log.append("store", entry_id=f"M-{index}")

        retained = audit_log.compact(retention_days=365)
        records = audit_log.read_all()

        assert retained == 3
        assert [record.id for record in records] == ["M-0", "M-1", "M-2"]
        assert audit_log.verify_chain()["valid"] is True

    def test_compact_all_records_expired_leaves_valid_empty_state(self, audit_log: AuditLog, audit_path: Path) -> None:
        for index in range(2):
            audit_log.append("store", entry_id=f"M-{index}")

        lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        for line in lines:
            line["ts"] = stale_ts
        audit_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

        retained = audit_log.compact(retention_days=1)

        assert retained == 0
        assert audit_log.verify_chain() == {
            "valid": True,
            "entries_checked": 0,
            "first_broken_at": None,
            "broken_hash": None,
        }

    def test_concurrent_writes_preserve_valid_chain(self, audit_path: Path) -> None:
        audit_log = AuditLog(audit_path)
        barrier = threading.Barrier(8)

        def _append(index: int) -> None:
            barrier.wait()
            audit_log.append("store", entry_id=f"M-{index}")

        threads = [threading.Thread(target=_append, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        result = audit_log.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 8

    def test_verify_chain_empty_file_returns_empty_valid_result(self, audit_path: Path) -> None:
        audit_path.write_text("", encoding="utf-8")
        result = AuditLog(audit_path).verify_chain()
        assert result == {"valid": True, "entries_checked": 0, "first_broken_at": None, "broken_hash": None}


class TestAuditPaths:
    def test_audit_log_path_differs_from_entry_storage(self, tmp_path: Path) -> None:
        from trw_memory.models.config import MemoryConfig

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        assert Path(cfg.audit_log_path).parent != Path(cfg.storage_path)
