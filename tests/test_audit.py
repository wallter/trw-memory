"""Tests for trw_memory.security.audit — immutable hash chain audit log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trw_memory.security.audit import AuditLog, AuditRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_path(tmp_path: Path) -> Path:
    """Return a temporary path for the audit log file."""
    return tmp_path / "audit.jsonl"


@pytest.fixture()
def audit_log(audit_path: Path) -> AuditLog:
    """Return a fresh AuditLog instance."""
    return AuditLog(audit_path)


# ---------------------------------------------------------------------------
# AuditRecord model tests
# ---------------------------------------------------------------------------


class TestAuditRecord:
    """Tests for the AuditRecord Pydantic model."""

    def test_default_fields(self) -> None:
        """Record has sensible defaults."""
        record = AuditRecord(action="store")
        assert record.action == "store"
        assert record.actor == ""
        assert record.target_id == ""
        assert record.namespace == "default"
        assert record.detail == {}
        assert record.prev_hash == ""
        assert record.record_hash == ""
        assert record.timestamp is not None

    def test_custom_fields(self) -> None:
        """All fields can be set explicitly."""
        record = AuditRecord(
            action="delete",
            actor="agent-1",
            target_id="M-abc123",
            namespace="project",
            detail={"reason": "cleanup"},
            prev_hash="abc",
            record_hash="def",
        )
        assert record.action == "delete"
        assert record.actor == "agent-1"
        assert record.target_id == "M-abc123"
        assert record.namespace == "project"
        assert record.detail == {"reason": "cleanup"}


# ---------------------------------------------------------------------------
# AuditLog.append tests
# ---------------------------------------------------------------------------


class TestAuditLogAppend:
    """Tests for appending records to the audit log."""

    def test_genesis_record_has_empty_prev_hash(
        self, audit_log: AuditLog
    ) -> None:
        """First record in the chain has prev_hash = ''."""
        record = audit_log.append(action="store", target_id="M-001")
        assert record.prev_hash == ""
        assert record.record_hash != ""

    def test_append_creates_hash_chain(self, audit_log: AuditLog) -> None:
        """Second record's prev_hash equals first record's record_hash."""
        r1 = audit_log.append(action="store", target_id="M-001")
        r2 = audit_log.append(action="recall", target_id="M-001")
        assert r2.prev_hash == r1.record_hash
        assert r2.record_hash != r1.record_hash

    def test_append_with_all_fields(self, audit_log: AuditLog) -> None:
        """All optional fields are persisted."""
        record = audit_log.append(
            action="update",
            target_id="M-002",
            namespace="project-x",
            actor="agent-alpha",
            detail={"field": "content"},
        )
        assert record.action == "update"
        assert record.target_id == "M-002"
        assert record.namespace == "project-x"
        assert record.actor == "agent-alpha"
        assert record.detail == {"field": "content"}

    def test_append_persists_to_file(
        self, audit_log: AuditLog, audit_path: Path
    ) -> None:
        """Records are written to the JSONL file."""
        audit_log.append(action="store")
        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["action"] == "store"
        assert data["record_hash"] != ""

    def test_multiple_appends_maintain_chain(
        self, audit_log: AuditLog
    ) -> None:
        """A chain of 5 records maintains consistent linkage."""
        records: list[AuditRecord] = []
        for i in range(5):
            r = audit_log.append(action="store", target_id=f"M-{i:03d}")
            records.append(r)

        # Check chain linkage
        assert records[0].prev_hash == ""
        for i in range(1, 5):
            assert records[i].prev_hash == records[i - 1].record_hash


# ---------------------------------------------------------------------------
# AuditLog.verify_chain tests
# ---------------------------------------------------------------------------


class TestAuditLogVerifyChain:
    """Tests for hash chain verification."""

    def test_empty_log_is_valid(self, audit_log: AuditLog) -> None:
        """An empty audit log verifies as valid."""
        valid, count, msg = audit_log.verify_chain()
        assert valid is True
        assert count == 0
        assert msg == ""

    def test_single_record_valid(self, audit_log: AuditLog) -> None:
        """A single-record chain verifies correctly."""
        audit_log.append(action="store")
        valid, count, msg = audit_log.verify_chain()
        assert valid is True
        assert count == 1
        assert msg == ""

    def test_multi_record_valid(self, audit_log: AuditLog) -> None:
        """A multi-record chain verifies correctly."""
        for i in range(10):
            audit_log.append(action="store", target_id=f"M-{i}")
        valid, count, msg = audit_log.verify_chain()
        assert valid is True
        assert count == 10
        assert msg == ""

    def test_tampering_detected_in_action_field(
        self, audit_log: AuditLog, audit_path: Path
    ) -> None:
        """Modifying a record's action field breaks the chain."""
        audit_log.append(action="store")
        audit_log.append(action="recall")
        audit_log.append(action="delete")

        # Tamper: change the second record's action
        lines = audit_path.read_text().strip().split("\n")
        tampered = json.loads(lines[1])
        tampered["action"] = "TAMPERED"
        lines[1] = json.dumps(tampered, sort_keys=True)
        audit_path.write_text("\n".join(lines) + "\n")

        # Re-create audit log to read tampered data
        tampered_log = AuditLog(audit_path)
        valid, count, msg = tampered_log.verify_chain()
        assert valid is False
        assert count == 3
        assert "record_hash mismatch" in msg

    def test_tampering_detected_in_prev_hash(
        self, audit_log: AuditLog, audit_path: Path
    ) -> None:
        """Modifying a record's prev_hash breaks the chain."""
        audit_log.append(action="store")
        audit_log.append(action="recall")

        lines = audit_path.read_text().strip().split("\n")
        tampered = json.loads(lines[1])
        tampered["prev_hash"] = "bogus_hash"
        lines[1] = json.dumps(tampered, sort_keys=True)
        audit_path.write_text("\n".join(lines) + "\n")

        tampered_log = AuditLog(audit_path)
        valid, count, msg = tampered_log.verify_chain()
        assert valid is False
        assert "prev_hash mismatch" in msg


# ---------------------------------------------------------------------------
# AuditLog.read_all tests
# ---------------------------------------------------------------------------


class TestAuditLogReadAll:
    """Tests for reading all records from the log."""

    def test_read_empty_nonexistent(self, audit_log: AuditLog) -> None:
        """Reading a nonexistent log returns empty list."""
        records = audit_log.read_all()
        assert records == []

    def test_read_all_returns_all_records(
        self, audit_log: AuditLog
    ) -> None:
        """All appended records are returned in order."""
        audit_log.append(action="store", target_id="M-001")
        audit_log.append(action="recall", target_id="M-002")
        audit_log.append(action="delete", target_id="M-003")

        records = audit_log.read_all()
        assert len(records) == 3
        assert records[0].action == "store"
        assert records[0].target_id == "M-001"
        assert records[1].action == "recall"
        assert records[2].action == "delete"

    def test_read_all_preserves_hash_chain(
        self, audit_log: AuditLog
    ) -> None:
        """Records read from disk maintain their hash chain fields."""
        r1 = audit_log.append(action="store")
        r2 = audit_log.append(action="recall")

        records = audit_log.read_all()
        assert records[0].record_hash == r1.record_hash
        assert records[1].prev_hash == r1.record_hash
        assert records[1].record_hash == r2.record_hash


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    """Verify that hash computation is deterministic."""

    def test_same_data_produces_same_hash(
        self, audit_log: AuditLog
    ) -> None:
        """Internal _compute_hash is deterministic."""
        data = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "action": "store",
            "target_id": "M-001",
            "prev_hash": "",
        }
        h1 = audit_log._compute_hash(data)
        h2 = audit_log._compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_data_produces_different_hash(
        self, audit_log: AuditLog
    ) -> None:
        """Even a small change produces a different hash."""
        data1 = {"action": "store", "target_id": "M-001"}
        data2 = {"action": "store", "target_id": "M-002"}
        assert audit_log._compute_hash(data1) != audit_log._compute_hash(
            data2
        )


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


class TestAuditLogPersistence:
    """Test that a new AuditLog instance resumes from existing data."""

    def test_new_instance_continues_chain(
        self, audit_path: Path
    ) -> None:
        """A new AuditLog created from an existing file continues the chain."""
        log1 = AuditLog(audit_path)
        r1 = log1.append(action="store")
        r2 = log1.append(action="recall")

        # Create a new instance from the same path
        log2 = AuditLog(audit_path)
        r3 = log2.append(action="delete")

        assert r3.prev_hash == r2.record_hash

        # Full chain should still verify
        valid, count, msg = log2.verify_chain()
        assert valid is True
        assert count == 3
