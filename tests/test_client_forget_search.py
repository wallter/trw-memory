from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryNotFoundError
from trw_memory.security.audit import AuditLog


class TestForget:
    async def test_forget_existing_entry(self, client: MemoryClient) -> None:
        stored = await client.store("to be forgotten")
        memory_id = stored["memory_id"]
        result = await client.forget(memory_id)
        assert result["memory_id"] == memory_id
        assert result["status"] == "deleted"
        assert result["namespace"] == "default"

    async def test_forget_nonexistent_raises(self, client: MemoryClient) -> None:
        with pytest.raises(MemoryNotFoundError):
            await client.forget("M-nonexistent")

    async def test_forget_then_recall_empty(self, client: MemoryClient) -> None:
        stored = await client.store("unique ephemeral content xyz123")
        await client.forget(stored["memory_id"])
        results = await client.recall("unique ephemeral content xyz123")
        assert len(results) == 0

    async def test_forget_wrong_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ns_test"))
        client_a = MemoryClient(namespace="project:aaa", mode="local")
        stored = await client_a.store("ns-A entry")

        client_b = MemoryClient(namespace="project:bbb", mode="local")
        with pytest.raises(MemoryNotFoundError):
            await client_b.forget(stored["memory_id"])

    async def test_forget_retired_remote_entry_when_remote_id_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        with patch("trw_memory.client.SSESubscriber"):
            client = MemoryClient(namespace="default", mode="local")

        with patch(
            "trw_memory.client.publish_memory_result",
            return_value={"success": True, "remote_id": "42", "retryable": False},
        ):
            stored = await client.store("retire this memory", importance=0.9)
            if client._background_tasks:
                await asyncio.gather(*list(client._background_tasks))

        with patch("trw_memory.client.retire_remote_memory", return_value=True) as retire_mock:
            await client.forget(stored["memory_id"])
            if client._background_tasks:
                await asyncio.gather(*list(client._background_tasks))

        retire_mock.assert_called_once_with("42", client._config)
        await client.close()


class TestSearch:
    async def test_search_returns_all_entries(self, client: MemoryClient) -> None:
        await client.store("entry one", importance=0.3)
        await client.store("entry two", importance=0.7)
        results = await client.search()
        assert len(results) >= 2

    async def test_search_min_importance_filter(self, client: MemoryClient) -> None:
        await client.store("low", importance=0.2)
        await client.store("high", importance=0.8)
        results = await client.search(min_importance=0.5)
        for r in results:
            assert r["importance"] >= 0.5

    async def test_search_tags_filter(self, client: MemoryClient) -> None:
        await client.store("tagged", tags=["python"])
        await client.store("untagged")
        results = await client.search(tags=["python"])
        assert all("python" in r["tags"] for r in results)

    async def test_search_actor_and_quarantined_filters(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Quarantine only fires under enforce mode (observe mode, the default,
        # records the anomaly but stores the entry normally — by design).
        # This test exercises the full quarantine path so it must opt in to
        # enforce mode explicitly, matching the pattern used in
        # test_sec001_live_paths.py::secure_client.
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_POISONING_DETECTION_MODE", "enforce")
        client = MemoryClient(namespace="default", mode="local")

        for index in range(20):
            await client.store(f"baseline {index}", source_identity="seed")

        result = await client.store("A" * 5000, source_identity="alice")
        quarantined = await client.search(actor="alice", status="quarantined")
        audit_records = AuditLog(Path(client._config.audit_log_path)).read_all()

        assert result["status"] == "quarantined"
        assert len(quarantined) == 1
        assert quarantined[0]["memory_id"] == result["memory_id"]
        assert quarantined[0]["anomaly_dimension"] == result["anomaly_dimension"]
        assert quarantined[0]["z_score"] == result["z_score"]
        assert audit_records[-1].op == "access"
        assert audit_records[-1].actor == "alice"

        await client.close()

    async def test_search_since_filter(self, client: MemoryClient) -> None:
        await client.store("old entry", importance=0.5)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = await client.search(since=future)
        assert len(results) == 0

    async def test_search_limit_validation(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.search(limit=0)

    async def test_search_min_importance_validation(self, client: MemoryClient) -> None:
        with pytest.raises(ValueError, match="min_importance"):
            await client.search(min_importance=-0.1)
        with pytest.raises(ValueError, match="min_importance"):
            await client.search(min_importance=1.5)

    async def test_search_sorted_by_importance_desc(self, client: MemoryClient) -> None:
        await client.store("low", importance=0.2)
        await client.store("mid", importance=0.5)
        await client.store("high", importance=0.9)
        results = await client.search()
        importances = [r["importance"] for r in results]
        assert importances == sorted(importances, reverse=True)

    async def test_search_status_filter_passes_enum_to_list_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1 regression: search(status='active') must push the enum filter to list_entries.

        Without the fix, list_entries is called without a status argument and returns
        ALL entries (including obsolete ones), which the Python post-filter then prunes.
        When there are many obsolete entries and fetch_limit = limit * 5, the truncated
        result set can return fewer results than the actual active population.
        """
        from trw_memory.models.memory import MemoryStatus

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

        client = MemoryClient(namespace="default", mode="local")

        # Store 10 active entries
        for index in range(10):
            await client.store(f"active content {index}", importance=0.6)

        # Track what status list_entries receives
        backend = client._get_backend()
        original = backend.list_entries
        received_statuses: list[MemoryStatus | None] = []

        def tracked(
            *,
            status: MemoryStatus | None = None,
            namespace: str | None = None,
            limit: int = 100,
        ) -> list:
            received_statuses.append(status)
            return original(status=status, namespace=namespace, limit=limit)

        from unittest.mock import patch

        with patch.object(backend, "list_entries", side_effect=tracked):
            results = await client.search(status="active", limit=5)

        assert len(results) == 5
        # list_entries must have been called with the ACTIVE enum, not None
        assert any(s == MemoryStatus.ACTIVE for s in received_statuses), (
            f"list_entries was never called with ACTIVE status; got: {received_statuses}"
        )

        await client.close()

    async def test_search_status_filter_does_not_raise_on_status_value_access(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1 regression: entry.status.value raises AttributeError when use_enum_values=True.

        MemoryEntry uses use_enum_values=True (Pydantic v2), so entry.status is a
        plain str ("active"), not a MemoryStatus enum. Calling .value on a str raises
        AttributeError. The post-filter must use str(entry.status) instead.
        """
        isolated = tmp_path / "isolated_status_value"
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(isolated))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

        client = MemoryClient(namespace="project:status-value-test", mode="local")
        await client.store("test entry", importance=0.7)

        # Must not raise AttributeError regardless of status parameter
        active_results = await client.search(status="active")
        assert len(active_results) == 1

        obsolete_results = await client.search(status="obsolete")
        assert len(obsolete_results) == 0

        await client.close()
