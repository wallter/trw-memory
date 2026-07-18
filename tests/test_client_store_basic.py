from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import AuthorizationError, MemoryNotFoundError, PIIBlockError, SchemaValidationError
from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.security.audit import AuditLog


class TestStore:
    async def test_store_returns_expected_keys(self, client: MemoryClient) -> None:
        result = await client.store("test content", tags=["tag1"])
        assert "memory_id" in result
        assert result["memory_id"].startswith("M-")
        assert result["namespace"] == "default"
        assert result["status"] == "stored"
        assert "timestamp" in result

    async def test_store_empty_content_raises(self, client: MemoryClient) -> None:
        with pytest.raises(SchemaValidationError, match="content"):
            await client.store("")

    async def test_store_whitespace_only_raises(self, client: MemoryClient) -> None:
        with pytest.raises(SchemaValidationError, match="content"):
            await client.store("   ")

    async def test_store_embedding_does_not_block_event_loop(self, client: MemoryClient) -> None:
        started = threading.Event()
        release = threading.Event()
        embedder = MagicMock()

        def slow_embed(_text: str) -> list[float]:
            started.set()
            release.wait(timeout=1.0)
            return [0.0] * client._config.embedding_dim

        embedder.embed.side_effect = slow_embed
        safety_release = threading.Timer(0.5, release.set)
        start = time.monotonic()
        safety_release.start()
        try:
            with patch("trw_memory._client_store.embedding_has_consumer", return_value=True):
                with patch.object(client, "_get_embedder", return_value=embedder):
                    task = asyncio.create_task(client.store("non-blocking embedding"))
                    assert await asyncio.to_thread(started.wait, 0.3)
                    assert time.monotonic() - start < 0.3
                    release.set()
                    await task
        finally:
            release.set()
            safety_release.cancel()


class TestRbacEnforcement:
    async def test_store_denied_for_reader_namespace_role(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.namespace_roles = {"default": "reader"}

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'reader' does not have store permission on namespace 'default'\.",
        ):
            await client.store("blocked write")

    async def test_recall_denied_for_writer_namespace_role(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.default_role = "admin"
        client._config.namespace_roles = {"default": "writer"}

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'writer' does not have recall permission on namespace 'default'\.",
        ):
            await client.recall("blocked read")

    async def test_audit_denied_without_read_permission(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.namespace_roles = {"default": "writer"}

        with pytest.raises(AuthorizationError, match="audit_learning permission"):
            await client.audit_learning("M-blocked")

    async def test_quarantine_review_requires_admin_permission(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.namespace_roles = {"default": "reader"}

        with pytest.raises(AuthorizationError, match="review_quarantined permission"):
            await client.review_quarantined("M-blocked", decision="approve", reviewer_id="reader")

    async def test_admin_role_allows_store_recall_and_forget(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.default_role = "admin"

        stored = await client.store("allowed entry", tags=["rbac"])
        results = await client.recall("allowed", limit=5)

        assert [result["memory_id"] for result in results] == [stored["memory_id"]]

        deleted = await client.forget(stored["memory_id"])
        assert deleted["status"] == "deleted"

    async def test_store_importance_too_low_raises(self, client: MemoryClient) -> None:
        with pytest.raises(SchemaValidationError, match="importance"):
            await client.store("content", importance=-0.1)

    async def test_store_importance_too_high_raises(self, client: MemoryClient) -> None:
        with pytest.raises(SchemaValidationError, match="importance"):
            await client.store("content", importance=1.1)

    async def test_store_with_metadata(self, client: MemoryClient) -> None:
        result = await client.store("content with meta", metadata={"source": "test"})
        assert result["status"] == "stored"

    async def test_store_boundary_importance(self, client: MemoryClient) -> None:
        r0 = await client.store("min importance", importance=0.0)
        r1 = await client.store("max importance", importance=1.0)
        assert r0["status"] == "stored"
        assert r1["status"] == "stored"

    async def test_store_with_detail(self, client: MemoryClient) -> None:
        result = await client.store("summary", detail="extended explanation", tags=["a"])
        assert result["status"] == "stored"

    async def test_store_persists_and_preserves_assertions_on_update(self, client: MemoryClient) -> None:
        assertion = Assertion(type=AssertionType.GLOB_EXISTS, target="src/**/*.py")
        await client.store(
            "grounded",
            evidence=["src/example.py:10-20"],
            assertions=[assertion],
            entry_id="M-grounded",
        )
        await client.store("grounded update", entry_id="M-grounded")

        stored = client._get_backend().get("M-grounded")

        assert stored is not None
        assert stored.evidence == ["src/example.py:10-20"]
        assert stored.assertions == [assertion]

    async def test_store_blocks_api_keys_and_audits_rejection(self, client: MemoryClient) -> None:
        with pytest.raises(PIIBlockError, match="blocked by PII policy"):
            await client.store("sk-abcdefghijklmnopqrstuvwxyz")

        audit_records = AuditLog(Path(client._config.audit_log_path)).read_all()
        assert audit_records[-1].op == "store_rejected"
        assert audit_records[-1].data["reason"] == "pii_detected"

    async def test_store_same_id_updates_existing_entry(self, client: MemoryClient) -> None:
        created = await client.store("original", source_identity="alice", entry_id="M-fixed")
        updated = await client.store("replacement", source_identity="alice", entry_id="M-fixed")
        results = await client.recall("replacement", limit=5)
        audit_records = AuditLog(Path(client._config.audit_log_path)).read_all()

        assert created["memory_id"] == "M-fixed"
        assert updated["status"] == "updated"
        assert [result["memory_id"] for result in results] == ["M-fixed"]
        assert any(record.op == "update" and record.id == "M-fixed" for record in audit_records)

    async def test_store_cannot_update_entry_in_another_namespace(self, tmp_path: Path) -> None:
        db_path = tmp_path / "shared.db"
        owner = MemoryClient("project:owner", db_path=db_path)
        other = MemoryClient("project:other", db_path=db_path)
        try:
            with patch("trw_memory._client_store.embedding_has_consumer", return_value=False):
                await owner.store("owner content", entry_id="M-shared")
                with pytest.raises(MemoryNotFoundError, match="not found in namespace"):
                    await other.store("other content", entry_id="M-shared")

            stored = owner._get_backend().get("M-shared")
            assert stored is not None
            assert stored.content == "owner content"
            assert stored.namespace == "project:owner"
        finally:
            await owner.close()
            await other.close()

    async def test_forget_actor_deletes_matching_entries(self, client: MemoryClient) -> None:
        await client.store("alice one", source_identity="alice")
        await client.store("alice two", source_identity="alice")
        await client.store("bob entry", source_identity="bob")

        result = await client.forget(actor="alice")
        remaining = await client.search(actor="alice")

        assert result["entries_deleted"] == 2
        assert remaining == []

    async def test_forget_actor_with_zero_entries_returns_zero_and_audits(self, client: MemoryClient) -> None:
        result = await client.forget(actor="ghost")
        audit_records = AuditLog(Path(client._config.audit_log_path)).read_all()

        assert result["entries_deleted"] == 0
        assert audit_records[-1].op == "forget"
        assert audit_records[-1].data["entries_deleted"] == 0

    async def test_search_actor_scans_full_namespace_before_filtering(
        self,
        client: MemoryClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entries = [MemoryEntry(id=f"M-seed-{index}", content="seed", namespace="default") for index in range(600)] + [
            MemoryEntry(
                id=f"M-alice-{index}",
                content="alice memory",
                namespace="default",
                source_identity="alice",
                importance=0.8,
            )
            for index in range(100)
        ]
        backend = MagicMock()
        backend.count.return_value = len(entries)
        backend.list_entries.side_effect = lambda **kwargs: entries[: int(kwargs["limit"])]
        monkeypatch.setattr(client, "_get_backend", lambda: backend)

        result = await client.search(actor="alice", limit=50)

        assert len(result) == 50
        assert all(str(item["memory_id"]).startswith("M-alice-") for item in result)

    async def test_forget_actor_scans_full_namespace_before_deleting(
        self,
        client: MemoryClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entries = [
            MemoryEntry(id=f"M-seed-{index}", content="seed", namespace="default") for index in range(10_050)
        ] + [
            MemoryEntry(
                id=f"M-alice-{index}",
                content="alice memory",
                namespace="default",
                source_identity="alice",
            )
            for index in range(20)
        ]
        deleted_ids: set[str] = set()
        backend = MagicMock()
        backend.count.return_value = len(entries)
        backend.list_entries.side_effect = lambda **kwargs: [entry for entry in entries if entry.id not in deleted_ids][
            : int(kwargs["limit"])
        ]
        backend.delete.side_effect = lambda entry_id: deleted_ids.add(entry_id) is None
        monkeypatch.setattr(client, "_get_backend", lambda: backend)
        monkeypatch.setattr("trw_memory.client.remove_entry_from_tiers", lambda *args, **kwargs: None)
        monkeypatch.setattr("trw_memory.client.delete_quarantined_entries", lambda *args, **kwargs: 0)

        result = await client.forget(actor="alice")

        assert result["entries_deleted"] == 20
        assert all(entry.id in deleted_ids for entry in entries[10_050:])
