"""E2E security tests for trw-memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_entry


class TestSecurity:
    """Section 7 of E2E plan: PII detection, encryption, audit."""

    def test_pii_detection_block_mode(self) -> None:
        """7.4 — PII detector in block mode raises on email address."""
        from trw_memory.exceptions import MemoryError as TrwMemoryError
        from trw_memory.security.pii import PIIAction, check_entry_pii

        entry = make_entry(
            content="Contact john@example.com for details",
            entry_id="pii-test-1",
        )
        with pytest.raises(TrwMemoryError, match="PII detected"):
            check_entry_pii(entry, action=PIIAction.BLOCK)

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


class TestPIIRedactMode:
    """Additional PII scenarios from section 7.5."""

    def test_pii_redact_mode_masks_content(self) -> None:
        """7.5 — PII detector in redact mode masks email in content."""
        from trw_memory.security.pii import PIIAction, check_entry_pii

        entry = make_entry(
            content="Contact john@example.com for support",
            entry_id="pii-redact-1",
        )
        updated, matches = check_entry_pii(entry, action=PIIAction.REDACT)
        assert len(matches) > 0
        assert "john@example.com" not in updated.content
        assert "[REDACTED:" in updated.content
