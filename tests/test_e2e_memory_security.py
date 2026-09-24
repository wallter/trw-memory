"""E2E security tests for trw-memory."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_entry


class TestSecurity:
    """Section 7 of E2E plan: PII detection, encryption, audit."""

    def test_field_encryption_roundtrip(self) -> None:
        """7.9 — Encrypt then decrypt entry fields preserves content."""
        from trw_memory.security.encryption import (
            decrypt_entry_fields,
            derive_namespace_key,
            derive_namespace_key_bytes,
            encrypt_entry_fields,
            generate_master_key,
        )

        master_key = generate_master_key()
        assert len(derive_namespace_key(master_key, "test-ns")) == 64
        namespace_key = derive_namespace_key_bytes(master_key, "test-ns")

        entry = make_entry(
            entry_id="enc-test-1",
            content="sensitive data",
            detail="very secret details",
        )
        encrypted = encrypt_entry_fields(entry, namespace_key)
        assert encrypted.content != "sensitive data"
        assert encrypted.detail != "very secret details"

        decrypted = decrypt_entry_fields(encrypted, namespace_key)
        assert decrypted.content == "sensitive data"
        assert decrypted.detail == "very secret details"

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
