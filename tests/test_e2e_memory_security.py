"""E2E security tests for trw-memory."""

from __future__ import annotations

from pathlib import Path


class TestSecurity:
    """Section 7 of E2E plan: PII detection and audit."""

    def test_audit_logging_records_operations(self, tmp_path: Path) -> None:
        """7.11 — Audit log records store/recall/delete events with hash chain."""
        from trw_memory.security.audit import AuditLog

        log_path = tmp_path / "audit.jsonl"
        audit = AuditLog(log_path)

        audit.append(action="store", target_id="M-001", namespace="test")
        audit.append(action="recall", target_id="", namespace="test")
        audit.append(action="delete", target_id="M-001", namespace="test")

        records = audit.read_all()
        assert len(records) == 3
        assert records[0].op == "store"
        assert records[1].op == "recall"
        assert records[2].op == "delete"

        result = audit.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 3
        assert result["first_broken_at"] is None

        assert records[1].prev_hash == records[0].hash
        assert records[2].prev_hash == records[1].hash
