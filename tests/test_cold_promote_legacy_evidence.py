"""Recall promoting a LEGACY cold-archived entry (pre-PRD-CORE-312, no evidence_level).

Real store in tmp_path, real cold archive YAML on disk, public ``MemoryClient.recall``;
nothing under test is mocked (only the embedder is disabled, as the sibling tier tests do).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from trw_memory.client import MemoryClient
from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import MAX_ENTRY_ID_CHARS, Confidence, MemoryEntry
from trw_memory.storage.persistence import write_yaml


def _legacy_record(entry_id: str, content: str) -> dict[str, object]:
    # What an older trw-memory left in the cold archive: confidence="verified",
    # no evidence_level key at all, and no primary row anywhere.
    return {
        "id": entry_id,
        "namespace": "default",
        "content": content,
        "confidence": "verified",
        "created_at": "2020-01-01T00:00:00+00:00",
    }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


def _cold_file(client: MemoryClient, name: str) -> Path:
    assert client._tier_manager is not None
    partition = client._tier_manager._cold_dir() / "2020" / "01"
    partition.mkdir(parents=True, exist_ok=True)
    return partition / name


async def test_legacy_verified_cold_entry_restores_demoted(client: MemoryClient) -> None:
    cold = _cold_file(client, "legacy-needle-001.yaml")
    write_yaml(cold, _legacy_record("legacy-needle-001", "legacy needle content for recall matching"))
    backend = client._get_backend()
    assert backend.get("legacy-needle-001", namespace="default") is None

    with patch.object(client, "_get_embedder", return_value=None):
        results = await client.recall("legacy needle content", limit=5)

    assert [row["memory_id"] for row in results] == ["legacy-needle-001"]
    restored = backend.get("legacy-needle-001", namespace="default")
    assert restored is not None
    # Restored through the CORE-312 served_view demotion, never as a verified claim.
    assert restored.confidence == Confidence.UNVERIFIED
    assert restored.metadata.get("served_confidence_demoted") == "true"
    assert not cold.exists()
    await client.close()


async def test_unrestorable_legacy_entry_is_skipped_loudly_and_archive_kept(client: MemoryClient) -> None:
    good = _cold_file(client, "good.yaml")
    bad = _cold_file(client, "bad.yaml")
    write_yaml(good, _legacy_record("legacy-good", "shared beacon phrase good"))
    bad_id = "x" * (MAX_ENTRY_ID_CHARS + 1)  # passes the model, refused by validate_entry_for_write
    write_yaml(bad, _legacy_record(bad_id, "shared beacon phrase bad"))

    with capture_logs() as logs, patch.object(client, "_get_embedder", return_value=None):
        results = await client.recall("shared beacon phrase", limit=5)

    assert [row["memory_id"] for row in results] == ["legacy-good"]
    skipped = [log for log in logs if log["event"] == "cold_promote_legacy_entry_skipped"]
    assert [(log["entry_id"], log["reason"]) for log in skipped] == [(bad_id, "entry_id_too_long")]
    assert bad.exists()
    assert client._get_backend().get(bad_id, namespace="default") is None
    await client.close()


def test_new_write_violating_evidence_invariant_is_still_refused(client: MemoryClient) -> None:
    entry = MemoryEntry(id="brand-new", content="new unsubstantiated claim", confidence=Confidence.VERIFIED)
    with pytest.raises(SchemaValidationError, match="requires evidence_level"):
        client._get_backend().store(entry)
    assert client._get_backend().get("brand-new", namespace="default") is None
