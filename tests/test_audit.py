"""Tests for the PRD-INFRA-020 audit log contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trw_memory.security.audit import AuditLog, AuditRecord


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


class TestAuditLogVerify:
    def test_verify_chain_returns_structured_result(self, audit_log: AuditLog) -> None:
        audit_log.append("store", entry_id="M-001")
        result = audit_log.verify_chain()
        assert result == {"valid": True, "record_count": 1, "error": "", "first_bad_line": None}

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
        assert result["first_bad_line"] == 2

    def test_compact_rechains_retained_suffix(self, audit_log: AuditLog) -> None:
        for index in range(3):
            audit_log.append("store", entry_id=f"M-{index}")
        retained = audit_log.compact(retention_days=365)
        records = audit_log.read_all()
        assert retained == 3
        assert records[0].prev_hash == "0" * 64
