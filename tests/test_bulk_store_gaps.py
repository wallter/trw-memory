"""Wave 15: coverage gap-fill for _client_bulk_store.py.

Target lines: 148-149, 163-170, 194, 229-239, 250-260, 270, 281-283, 287-288, 297, 319.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import BulkStoreRequest, MemoryClient
from trw_memory._client_bulk_store import BulkStoreItemResult, bulk_store_impl


@pytest.fixture
async def isolated_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "mem.db"
    monkeypatch.setenv("TRW_DIR", str(trw_dir))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_STORAGE_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "0")
    monkeypatch.chdir(tmp_path)
    ns = f"project:gap-test-{uuid.uuid4().hex[:8]}"
    client = MemoryClient(namespace=ns, mode="local")
    yield client
    await client.close()


@pytest.fixture
async def team_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "mem.db"
    monkeypatch.setenv("TRW_DIR", str(trw_dir))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_STORAGE_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "0")
    monkeypatch.chdir(tmp_path)
    client = MemoryClient(namespace="team:gap-test", mode="local")
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# lines 148-149, 163-170: SchemaValidationError path → rejected item
# ---------------------------------------------------------------------------

class TestBulkStoreSchemaValidationRejection:
    async def test_whitespace_only_content_is_rejected(self, isolated_client: MemoryClient) -> None:
        """BulkStoreRequest with whitespace content fails validate_store_inputs → rejected (lines 148-149, 163-170)."""
        valid = BulkStoreRequest(content="valid content", detail="ok")
        invalid = BulkStoreRequest(content="   ", detail="empty content")  # whitespace only
        summary = await isolated_client.bulk_store([valid, invalid])
        assert summary.total == 2
        assert summary.rejected == 1
        assert summary.stored == 1
        rejected_item = next(it for it in summary.items if it.status == "rejected")
        assert "content" in rejected_item.skipped_reason


# ---------------------------------------------------------------------------
# line 194: update path (existing entry)
# ---------------------------------------------------------------------------

class TestBulkStoreUpdatePath:
    async def test_update_existing_entry(self, isolated_client: MemoryClient) -> None:
        """entry_id matches existing entry → model_copy update path (line 194)."""
        store_result = await isolated_client.store("original content", detail="original")
        memory_id = store_result["memory_id"]
        req = BulkStoreRequest(content="updated content", detail="new detail", entry_id=memory_id)
        summary = await isolated_client.bulk_store([req])
        assert summary.updated == 1


# ---------------------------------------------------------------------------
# lines 229-239: quarantine path
# ---------------------------------------------------------------------------

class TestBulkStoreQuarantinePath:
    async def test_quarantined_decision_lands_in_quarantine(self, isolated_client: MemoryClient) -> None:
        """When security decision quarantines an entry → quarantined result (lines 229-239)."""
        from trw_memory.security.runtime import PreparedStoreEntry
        from trw_memory.models.memory import MemoryEntry

        fake_entry = MemoryEntry(id="M-quarantined", content="test", namespace="project:default")
        quarantined_decision = PreparedStoreEntry(
            entry=fake_entry,
            op="store",
            pii_matches=(),
            quarantined=True,
            anomaly_dimension="injection",
            anomaly_z_score=5.0,
        )

        with patch("trw_memory._client_bulk_store.prepare_entry_for_store", return_value=quarantined_decision):
            with patch("trw_memory._client_bulk_store.store_quarantined_entry"):
                summary = await isolated_client.bulk_store([BulkStoreRequest(content="test", detail="d")])

        assert summary.quarantined == 1
        quarantine_item = next(it for it in summary.items if it.quarantined)
        assert quarantine_item.anomaly_dimension == "injection"


# ---------------------------------------------------------------------------
# lines 250-260: embed_batch exception
# ---------------------------------------------------------------------------

class TestBulkStoreEmbedBatchFailure:
    async def test_embed_batch_exception_uses_none_embeddings(self, isolated_client: MemoryClient) -> None:
        """embed_batch raises → warning logged, embeddings=[None]*n (lines 250-260)."""
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = RuntimeError("CUDA OOM")

        with patch("trw_memory._client_bulk_store.embedding_has_consumer", return_value=True):
            with patch.object(isolated_client, "_get_embedder", return_value=mock_embedder):
                summary = await isolated_client.bulk_store([BulkStoreRequest(content="test embed", detail="d")])

        assert summary.stored == 1  # entry still stored despite embed failure


# ---------------------------------------------------------------------------
# line 270: team namespace ensure_team_namespace
# ---------------------------------------------------------------------------

class TestBulkStoreTeamNamespace:
    async def test_team_namespace_creates_team_ns(self, team_client: MemoryClient) -> None:
        """team: namespace → NamespaceManager.ensure_team_namespace called (line 270)."""
        summary = await team_client.bulk_store([BulkStoreRequest(content="team memory", detail="d")])
        assert summary.stored == 1


# ---------------------------------------------------------------------------
# lines 281-283: embedding upsert in transaction
# ---------------------------------------------------------------------------

class TestBulkStoreVectorUpsert:
    async def test_vector_upserted_when_embedding_available(self, isolated_client: MemoryClient) -> None:
        """When embedding provided → backend.upsert_vector called (lines 281-283)."""
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]

        with patch("trw_memory._client_bulk_store.embedding_has_consumer", return_value=True):
            with patch.object(isolated_client, "_get_embedder", return_value=mock_embedder):
                summary = await isolated_client.bulk_store([BulkStoreRequest(content="vec content", detail="d")])

        assert summary.stored == 1


# ---------------------------------------------------------------------------
# lines 287-288: schedule_graph_update RuntimeError
# ---------------------------------------------------------------------------

class TestBulkStoreGraphScheduleFailure:
    async def test_graph_update_runtime_error_is_logged(self, isolated_client: MemoryClient) -> None:
        """schedule_graph_update raises RuntimeError → warning logged, continues (lines 287-288)."""
        with patch("trw_memory._client_bulk_store.schedule_graph_update", side_effect=RuntimeError("graph locked")):
            summary = await isolated_client.bulk_store([BulkStoreRequest(content="graph test", detail="d")])

        assert summary.stored == 1  # entry still stored despite graph failure


# ---------------------------------------------------------------------------
# line 297: skip_audit_per_item=False path
# ---------------------------------------------------------------------------

class TestBulkStoreTransactionFailure:
    async def test_transaction_exception_raises_storage_error(self, isolated_client: MemoryClient) -> None:
        """backend.store raises inside transaction → StorageError (lines 282-283)."""
        from trw_memory.exceptions import StorageError as SE
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        with patch.object(SQLiteBackend, "store", side_effect=RuntimeError("io error")):
            with pytest.raises(SE, match="transaction rolled back"):
                await bulk_store_impl(
                    isolated_client,
                    [BulkStoreRequest(content="txn test", detail="d")],
                    skip_remote_publish=True,
                )


class TestBulkStorePerItemAudit:
    async def test_per_item_audit_appended_when_not_skipped(self, isolated_client: MemoryClient) -> None:
        """skip_audit_per_item=False → append_audit_event called per item (line 297)."""
        with patch("trw_memory._client_bulk_store.append_audit_event") as mock_audit:
            summary = await bulk_store_impl(
                isolated_client,
                [BulkStoreRequest(content="audit item", detail="d")],
                skip_audit_per_item=False,
                skip_remote_publish=True,
            )
        assert summary.stored == 1
        mock_audit.assert_called()


class TestBulkStoreRemotePublish:
    async def test_remote_publish_scheduled_when_not_skipped(self, isolated_client: MemoryClient) -> None:
        """skip_remote_publish=False + _should_attempt_remote_publish=True → scheduled (line 319)."""
        with patch.object(isolated_client, "_should_attempt_remote_publish", return_value=True):
            with patch.object(isolated_client, "_schedule_background_task") as mock_schedule:
                with patch.object(isolated_client, "_publish_entry", return_value=None):
                    summary = await bulk_store_impl(
                        isolated_client,
                        [BulkStoreRequest(content="remote pub", detail="d")],
                        skip_remote_publish=False,
                    )
        assert summary.stored == 1
        mock_schedule.assert_called()
