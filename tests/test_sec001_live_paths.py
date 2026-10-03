from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.startup import resolve_security_path, verify_defaults


@pytest.fixture()
def secure_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("TRW_DIR", str(tmp_path / ".trw"))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_ENABLE_RECALL_FILTER", "true")
    monkeypatch.setenv("MEMORY_RECALL_FILTER_MODE", "strict")
    monkeypatch.setenv("MEMORY_PROVENANCE_REQUIRED", "true")
    monkeypatch.setenv("MEMORY_CANARY_PROBE_INTERVAL", "1")
    monkeypatch.setenv("MEMORY_CANARY_FAIL_MODE", "halt")
    return MemoryClient(namespace="default", mode="local")


async def test_audit_marks_legacy_unsigned_rows(secure_client: MemoryClient) -> None:
    backend = secure_client._get_backend()
    backend.store(
        MemoryEntry(
            id="M-legacy-001",
            content="legacy unsigned row",
            namespace="default",
            metadata={},
        )
    )
    audit = await secure_client.audit_learning("M-legacy-001")
    assert audit["status"] == "legacy_unsigned"


async def test_audit_uses_real_verification_key_semantics(secure_client: MemoryClient) -> None:
    stored = await secure_client.store(
        "safe content",
        detail="safe detail",
        source_identity="sec-audit-agent",
        session_id="sess-real-key",
    )
    assert stored["status"] == "stored"

    key_path = resolve_security_path(secure_client._config, "provenance_signing_key_path")
    key_path.write_bytes(b"x" * 32)

    audit = await secure_client.audit_learning(stored["memory_id"])
    assert audit["verified"] is False


async def test_audit_and_review_hide_other_namespace_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    owner = MemoryClient("project:owner", db_path=db_path)
    other = MemoryClient("project:other", db_path=db_path)
    try:
        owner._get_backend().store(
            MemoryEntry(id="M-foreign", content="owner only", namespace="project:owner", metadata={})
        )

        assert (await other.audit_learning("M-foreign"))["status"] == "not_found"
        assert (await other.review_quarantined("M-foreign", decision="approve", reviewer_id="other-admin"))[
            "status"
        ] == "not_found"
    finally:
        await owner.close()
        await other.close()


def test_security_path_resolution_anchors_to_trw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir()
    monkeypatch.setenv("TRW_DIR", str(trw_dir))

    config = MemoryConfig(storage_path=str(tmp_path / "storage"), quarantine_db_path="memory/security/quarantine.db")
    resolved = resolve_security_path(config, "quarantine_db_path")
    assert resolved == (trw_dir / "memory/security/quarantine.db").resolve()

    verify_defaults(config)
