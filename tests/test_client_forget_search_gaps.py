"""Wave 15: coverage gap-fill for _client_forget_search.py.

Target lines: 67, 107-121, 123, 170, 192-193, 211.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch
from typing import AsyncGenerator

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryNotFoundError
from trw_memory.models.memory import MemoryEntry


@pytest.fixture
async def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[MemoryClient, None]:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "mem.db"
    monkeypatch.setenv("TRW_DIR", str(trw_dir))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_STORAGE_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "0")
    monkeypatch.chdir(tmp_path)
    ns = f"project:fs-test-{uuid.uuid4().hex[:8]}"
    c = MemoryClient(namespace=ns, mode="local")
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# line 67: no memory_id and no actor → ValueError
# ---------------------------------------------------------------------------

class TestForgetNoArgs:
    async def test_forget_no_memory_id_no_actor_raises(self, client: MemoryClient) -> None:
        """forget() with neither memory_id nor actor → ValueError (line 67)."""
        with pytest.raises(ValueError, match="memory_id or actor must be provided"):
            await client.forget()


# ---------------------------------------------------------------------------
# lines 107-121: entry not in backend but IS in quarantine → success delete
# ---------------------------------------------------------------------------

class TestForgetQuarantinePath:
    async def test_forget_quarantined_entry_succeeds(self, client: MemoryClient) -> None:
        """forget(memory_id) where backend.get=None but quarantine has it → deleted (lines 107-121)."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        with patch.object(SQLiteBackend, "get", return_value=None):
            with patch("trw_memory.client.delete_quarantined_entries", return_value=1):
                with patch("trw_memory._client_forget_search.append_audit_event"):
                    result = await client.forget(memory_id="M-quarantined")

        assert result["status"] == "deleted"
        assert result["memory_id"] == "M-quarantined"
        assert result.get("entries_deleted") == 1

    async def test_forget_not_found_in_backend_or_quarantine_raises(self, client: MemoryClient) -> None:
        """forget(memory_id) where backend.get=None and quarantine=0 → MemoryNotFoundError."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        with patch.object(SQLiteBackend, "get", return_value=None):
            with patch("trw_memory.client.delete_quarantined_entries", return_value=0):
                with pytest.raises(MemoryNotFoundError):
                    await client.forget(memory_id="M-missing")


# ---------------------------------------------------------------------------
# line 123: entry found but in different namespace → MemoryNotFoundError
# ---------------------------------------------------------------------------

class TestForgetWrongNamespace:
    async def test_forget_wrong_namespace_raises(self, client: MemoryClient) -> None:
        """forget(memory_id) where entry.namespace != client._namespace → MemoryNotFoundError (line 123)."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        wrong_ns_entry = MemoryEntry(
            id="M-wrong-ns",
            content="belongs to other ns",
            namespace="project:other-namespace",
        )
        with patch.object(SQLiteBackend, "get", return_value=wrong_ns_entry):
            with pytest.raises(MemoryNotFoundError, match="not found in namespace"):
                await client.forget(memory_id="M-wrong-ns")


# ---------------------------------------------------------------------------
# line 170: invalid status → ValueError
# ---------------------------------------------------------------------------

class TestSearchInvalidStatus:
    async def test_invalid_status_raises(self, client: MemoryClient) -> None:
        """search(status='bad') → ValueError (line 170)."""
        with pytest.raises(ValueError, match="status must be one of"):
            await client.search(status="bad_value")


# ---------------------------------------------------------------------------
# lines 192-193: MemoryStatus(status) raises ValueError → _status_enum = None
# ---------------------------------------------------------------------------

class TestSearchStatusEnumFallback:
    async def test_status_enum_value_error_falls_back_to_none(self, client: MemoryClient) -> None:
        """When MemoryStatus(status) raises ValueError → _status_enum stays None (lines 192-193)."""
        await client.store("some content", detail="d")

        # Patch MemoryStatus in the _client_forget_search module so the try/except fires.
        # status="active" passes the allowlist check at line 169 but MemoryStatus("active")
        # then raises → hits except ValueError: _status_enum = None at line 193.
        with patch("trw_memory._client_forget_search.MemoryStatus", side_effect=ValueError("bad enum")):
            results = await client.search(status="active")

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# line 211: status filter skips entries whose status doesn't match
# ---------------------------------------------------------------------------

class TestSearchStatusFilterSkipsNonMatching:
    async def test_status_filter_excludes_non_matching_entries(self, client: MemoryClient) -> None:
        """search(status='resolved') skips non-resolved entries → continue at line 211.

        Patch list_entries to return an active entry so the status != filter fires.
        """
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        active_entry = MemoryEntry(
            id="M-active-skip",
            content="active content",
            namespace=client._namespace,
        )
        # status is ACTIVE by default; search(status='resolved') should skip it
        with patch.object(SQLiteBackend, "list_entries", return_value=[active_entry]):
            results = await client.search(status="resolved")

        assert isinstance(results, list)
        assert len(results) == 0  # active entry skipped by line 211

    async def test_status_active_filter_returns_only_active(self, client: MemoryClient) -> None:
        """search(status='active') returns active entries."""
        await client.store("content for active", detail="d")

        results = await client.search(status="active")

        assert isinstance(results, list)
        assert len(results) >= 1
