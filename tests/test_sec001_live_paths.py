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
    monkeypatch.setenv("MEMORY_ENABLE_TRUST_SCORING", "true")
    monkeypatch.setenv("MEMORY_TRUST_SCORING_MODE", "enforce")
    monkeypatch.setenv("MEMORY_TRUST_SCORE_THRESHOLD", "0.8")
    monkeypatch.setenv("MEMORY_ENABLE_RECALL_FILTER", "true")
    monkeypatch.setenv("MEMORY_RECALL_FILTER_MODE", "strict")
    monkeypatch.setenv("MEMORY_PROVENANCE_REQUIRED", "true")
    monkeypatch.setenv("MEMORY_CANARY_PROBE_INTERVAL", "1")
    monkeypatch.setenv("MEMORY_CANARY_FAIL_MODE", "halt")
    return MemoryClient(namespace="default", mode="local")


async def test_store_quarantine_review_and_audit_live_path(secure_client: MemoryClient) -> None:
    stored = await secure_client.store(
        "benign oversized note " + ("x" * 200000),
        source_identity="sec-audit-agent",
        session_id="sess-1",
    )

    assert stored["status"] == "quarantined"

    quarantined = await secure_client.search(status="quarantined")
    assert [entry["memory_id"] for entry in quarantined] == [stored["memory_id"]]

    assert await secure_client.recall("benign oversized", limit=10) == []

    audit_before = await secure_client.audit_learning(stored["memory_id"])
    assert audit_before["status"] == "quarantined"
    assert audit_before["content_hash"]
    assert audit_before["signature"]
    assert audit_before["verified"] is True
    assert audit_before["status_history"][0]["status"] == "quarantined"

    review = await secure_client.review_quarantined(
        stored["memory_id"],
        decision="approve",
        reviewer_id="maintainer-1",
    )
    assert review["status"] == "approved"

    recalled = await secure_client.recall("benign oversized", limit=10)
    assert [entry["memory_id"] for entry in recalled] == [stored["memory_id"]]

    audit_after = await secure_client.audit_learning(stored["memory_id"])
    assert [item["status"] for item in audit_after["status_history"]] == ["quarantined", "active"]

    second_review = await secure_client.review_quarantined(
        stored["memory_id"],
        decision="reject",
        reviewer_id="maintainer-2",
    )
    assert second_review["status"] == "already_resolved"


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


async def test_observe_mode_store_starts_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trw_dir = tmp_path / ".trw"
    monkeypatch.setenv("TRW_DIR", str(trw_dir))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_ENABLE_TRUST_SCORING", "true")
    monkeypatch.setenv("MEMORY_TRUST_SCORING_MODE", "observe")
    monkeypatch.setenv("MEMORY_PROVENANCE_REQUIRED", "true")

    client = MemoryClient(namespace="default", mode="local")
    await client.store("observe content", detail="observe detail", source_identity="observer", session_id="observe-sess")

    assert (trw_dir / "memory" / "security" / "observe_start.yaml").exists()


def test_security_path_resolution_anchors_to_trw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir()
    monkeypatch.setenv("TRW_DIR", str(trw_dir))

    config = MemoryConfig(storage_path=str(tmp_path / "storage"), quarantine_db_path="memory/security/quarantine.db")
    resolved = resolve_security_path(config, "quarantine_db_path")
    assert resolved == (trw_dir / "memory/security/quarantine.db").resolve()

    verify_defaults(config)
