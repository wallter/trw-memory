"""Single-key metadata round-trip investigation.

A downstream smoke test surfaced that records stored with single-key
``metadata={"utility_grade": "R3"}`` come back from ``MemoryClient.recall``
with ``metadata={}``. Records with multi-key metadata round-trip cleanly.

This test reproduces the bug and pins the expected behavior. Bisects
across the four hypothesis paths for this investigation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


class TestSingleKeyMetadataRoundTrip:
    """Each test isolates one layer of the storage path."""

    def test_storage_layer_round_trip_preserves_single_key(self, tmp_path: Path) -> None:
        """SQLite backend round-trips single-key metadata via JSON encoding.

        This is the cheapest test — a direct backend.store + backend.get
        cycle. Uses MemoryEntry (Pydantic v2) directly. If THIS fails,
        the bug is in row mapping or json parsing.
        """
        backend = SQLiteBackend(tmp_path / "test.db")
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="M-single-key",
            content="single-key metadata test",
            detail="",
            tags=[],
            importance=0.5,
            status=MemoryStatus.ACTIVE,
            namespace="default",
            metadata={"utility_grade": "R3"},
            created_at=now,
            updated_at=now,
            source="agent",
        )
        backend.store(entry)
        result = backend.get("M-single-key", namespace="default")

        assert result is not None
        # The entry's metadata field should round-trip the single key.
        assert result.metadata == {"utility_grade": "R3"}, (
            f"Storage layer dropped single-key metadata: got {result.metadata!r}"
        )

    @pytest.mark.asyncio
    async def test_client_store_then_get_preserves_user_keys(self, memory_client: MemoryClient) -> None:
        """MemoryClient.store + backend.get: user metadata + installation_id co-exist.

        The store path adds ``installation_id`` automatically. So the stored
        entry should have at least 2 keys; the user's key must be present.
        """
        result = await memory_client.store(
            content="single-key client store",
            metadata={"utility_grade": "R3"},
        )
        memory_id = result["memory_id"]

        backend = memory_client._get_backend()
        entry = backend.get(memory_id, namespace="default")
        assert entry is not None
        assert entry.metadata.get("utility_grade") == "R3", f"client.store dropped utility_grade: {entry.metadata!r}"
        # installation_id is always added by the store path.
        assert "installation_id" in entry.metadata

    @pytest.mark.asyncio
    async def test_client_recall_returns_user_metadata(self, memory_client: MemoryClient) -> None:
        """MemoryClient.recall: user metadata flows through into result dicts.

        This is the original bug surface. ``recall`` should
        return a result dict where ``metadata['utility_grade'] == 'R3'``.
        """
        await memory_client.store(
            content="The single-key recall round-trip probe content",
            metadata={"utility_grade": "R3"},
        )
        results = await memory_client.recall("single-key recall round-trip probe")
        assert results, "recall returned no results — query may not match"

        # Find our record.
        match = next(
            (r for r in results if r.get("content") == "The single-key recall round-trip probe content"),
            None,
        )
        assert match is not None, f"recall results: {results}"
        meta = match.get("metadata", {})
        assert meta.get("utility_grade") == "R3", f"recall path dropped utility_grade: metadata={meta!r}"

    @pytest.mark.asyncio
    async def test_bulk_store_then_recall_preserves_user_keys(self, memory_client: MemoryClient) -> None:
        """bulk_store + recall: same path the original fixture used.

        This is the exact reproducer for the original audit's bug
        observation: BulkStoreRequest with single-key metadata comes
        back through recall with metadata={}.
        """
        from trw_memory._client_bulk_store import BulkStoreRequest

        req = BulkStoreRequest(
            entry_id="CANARY-R3-probe",
            content="The single-key bulk_store recall probe content",
            detail="probe detail",
            tags=["probe"],
            importance=0.5,
            metadata={"utility_grade": "R3"},
            source="agent",
        )
        await memory_client.bulk_store([req])

        results = await memory_client.recall("single-key bulk_store recall probe")
        assert results, "bulk_store + recall returned no results"

        match = next(
            (r for r in results if r.get("memory_id") == "CANARY-R3-probe"),
            None,
        )
        assert match is not None, f"recall results: {[r.get('memory_id') for r in results]}"
        meta = match.get("metadata", {})
        assert meta.get("utility_grade") == "R3", f"bulk_store + recall path dropped utility_grade: metadata={meta!r}"

    @pytest.mark.asyncio
    async def test_recall_returns_metadata_field_at_all(self, memory_client: MemoryClient) -> None:
        """Diagnostic: recall result should always include a metadata field.

        Even if the bug were intermittent, the metadata key should always
        be present in the result dict (at minimum carrying installation_id).
        """
        from trw_memory._client_bulk_store import BulkStoreRequest

        req = BulkStoreRequest(
            entry_id="CANARY-meta-presence-probe",
            content="metadata presence diagnostic content",
            detail="",
            tags=[],
            importance=0.5,
            metadata={"utility_grade": "R3"},
            source="agent",
        )
        await memory_client.bulk_store([req])
        results = await memory_client.recall("metadata presence diagnostic")
        match = next(
            (r for r in results if r.get("memory_id") == "CANARY-meta-presence-probe"),
            None,
        )
        assert match is not None
        # The metadata key should be present (NotRequired in TypedDict, but
        # _client_recall sets it unconditionally on line 279).
        assert "metadata" in match
        meta = match["metadata"]
        # At minimum, installation_id should be present (set by store path).
        assert "installation_id" in meta or "utility_grade" in meta, f"recall returned empty metadata dict: {meta!r}"


class TestUserKeyDropDiagnostic:
    """Drill into where a single user key gets dropped, if it does."""

    def test_pydantic_model_dump_preserves_single_key_metadata(self) -> None:
        """Pydantic v2 model_dump on MemoryEntry preserves single-key metadata."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="M-dump-test",
            content="dump test",
            detail="",
            namespace="default",
            metadata={"utility_grade": "R3"},
            created_at=now,
            updated_at=now,
            source="agent",
        )
        dumped = entry.model_dump()
        assert dumped["metadata"] == {"utility_grade": "R3"}

    def test_json_round_trip_preserves_single_key_metadata(self) -> None:
        """json.dumps + json.loads preserves single-key dicts."""
        original = {"utility_grade": "R3"}
        round_tripped = json.loads(json.dumps(original))
        assert round_tripped == original
