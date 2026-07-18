"""Tests for superseded-entry hard filter at SQL candidate-generation level.

Frontier-003: exclude_superseded parameter added to:
- StorageBackend.list_entries() (interface)
- SQLiteBackend.list_entries() (_query_ops.list_entries + sqlite_backend delegator)
- YAMLBackend.list_entries()
- _client_recall_hybrid.try_hybrid_recall() (passes exclude_superseded=True
  when include_superseded=False and as_of is None)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
_T2 = datetime(2026, 1, 3, tzinfo=timezone.utc)


def _make_entry(entry_id: str, *, namespace: str = "default", superseded: bool = False) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        namespace=namespace,
        status=MemoryStatus.ACTIVE,
        importance=0.5,
        created_at=_T0,
        updated_at=_T1,
        valid_from=_T0,
        invalid_from=_T2 if superseded else None,
        invalidated_by=f"newer-{entry_id}" if superseded else None,
    )


def _mem_backend() -> SQLiteBackend:
    return SQLiteBackend(Path(":memory:"))


# ---------------------------------------------------------------------------
# SQLiteBackend
# ---------------------------------------------------------------------------


class TestSQLiteExcludeSuperseded:
    def test_default_includes_superseded(self) -> None:
        backend = _mem_backend()
        backend.store(_make_entry("a", superseded=False))
        backend.store(_make_entry("b", superseded=True))
        entries = backend.list_entries()
        ids = {e.id for e in entries}
        assert "a" in ids
        assert "b" in ids

    def test_exclude_superseded_true_removes_invalid_from(self) -> None:
        backend = _mem_backend()
        backend.store(_make_entry("a", superseded=False))
        backend.store(_make_entry("b", superseded=True))
        entries = backend.list_entries(exclude_superseded=True)
        ids = {e.id for e in entries}
        assert "a" in ids
        assert "b" not in ids

    def test_exclude_superseded_false_keeps_all(self) -> None:
        backend = _mem_backend()
        backend.store(_make_entry("x", superseded=False))
        backend.store(_make_entry("y", superseded=True))
        entries = backend.list_entries(exclude_superseded=False)
        ids = {e.id for e in entries}
        assert "x" in ids
        assert "y" in ids

    def test_exclude_superseded_respects_namespace(self) -> None:
        backend = _mem_backend()
        backend.store(_make_entry("ns1-active", namespace="ns1", superseded=False))
        backend.store(_make_entry("ns1-super", namespace="ns1", superseded=True))
        backend.store(_make_entry("ns2-active", namespace="ns2", superseded=False))
        entries = backend.list_entries(namespace="ns1", exclude_superseded=True)
        ids = {e.id for e in entries}
        assert "ns1-active" in ids
        assert "ns1-super" not in ids
        assert "ns2-active" not in ids

    def test_only_superseded_entries_returns_empty(self) -> None:
        backend = _mem_backend()
        backend.store(_make_entry("s1", superseded=True))
        backend.store(_make_entry("s2", superseded=True))
        entries = backend.list_entries(exclude_superseded=True)
        assert entries == []

    def test_no_entries_returns_empty(self) -> None:
        backend = _mem_backend()
        entries = backend.list_entries(exclude_superseded=True)
        assert entries == []

    def test_limit_applied_after_superseded_filter(self) -> None:
        backend = _mem_backend()
        for i in range(5):
            backend.store(_make_entry(f"active-{i}", superseded=False))
            backend.store(_make_entry(f"super-{i}", superseded=True))
        # limit=3 must apply to the filtered set (only active entries)
        entries = backend.list_entries(exclude_superseded=True, limit=3)
        assert len(entries) <= 3
        assert all(e.invalid_from is None for e in entries)


# ---------------------------------------------------------------------------
# YAMLBackend
# ---------------------------------------------------------------------------


class TestYAMLExcludeSuperseded:
    def test_default_includes_superseded(self, tmp_path: Path) -> None:
        backend = YAMLBackend(tmp_path / "mem.yaml")
        backend.store(_make_entry("a", superseded=False))
        backend.store(_make_entry("b", superseded=True))
        entries = backend.list_entries()
        ids = {e.id for e in entries}
        assert "a" in ids
        assert "b" in ids

    def test_exclude_superseded_true_removes_invalid_from(self, tmp_path: Path) -> None:
        backend = YAMLBackend(tmp_path / "mem.yaml")
        backend.store(_make_entry("a", superseded=False))
        backend.store(_make_entry("b", superseded=True))
        entries = backend.list_entries(exclude_superseded=True)
        ids = {e.id for e in entries}
        assert "a" in ids
        assert "b" not in ids

    def test_only_superseded_returns_empty(self, tmp_path: Path) -> None:
        backend = YAMLBackend(tmp_path / "mem.yaml")
        backend.store(_make_entry("s", superseded=True))
        entries = backend.list_entries(exclude_superseded=True)
        assert entries == []


# ---------------------------------------------------------------------------
# Hybrid recall integration: exclude_superseded wired correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_recall_excludes_superseded_at_candidate_level() -> None:
    """Superseded entries must not consume candidate slots in BM25+dense pool."""
    pytest.importorskip("rank_bm25")
    import uuid

    from trw_memory.client import MemoryClient

    ns = f"project:sup{uuid.uuid4().hex[:8]}"
    client = MemoryClient(namespace=ns, mode="local")
    await client.store(
        content="active memory about dogs",
        detail="dogs are great pets",
        importance=0.8,
    )
    # Store a superseded entry with same content footprint
    superseded = _make_entry("superseded-dogs", namespace=ns, superseded=True)
    superseded.content = "superseded memory about dogs"
    async with client._lock:
        client._get_backend().store(superseded)

    results = await client.recall("dogs", limit=10)
    result_ids = [r["memory_id"] for r in results]
    # Superseded entry must not appear in default recall (include_superseded=False)
    assert "superseded-dogs" not in result_ids
    await client.close()


@pytest.mark.asyncio
async def test_hybrid_recall_include_superseded_surfaces_them() -> None:
    """include_superseded=True must surface superseded entries in results."""
    pytest.importorskip("rank_bm25")
    import uuid

    from trw_memory.client import MemoryClient

    ns = f"project:sup{uuid.uuid4().hex[:8]}"
    client = MemoryClient(namespace=ns, mode="local")
    # Store superseded entry directly (avoid client.store() which generates M-* IDs)
    superseded = _make_entry("super-cats", namespace=ns, superseded=True)
    superseded.content = "superseded memory about cats"
    async with client._lock:
        client._get_backend().store(superseded)
    await client.store(
        content="active memory about cats",
        importance=0.9,
    )

    results = await client.recall("cats", limit=10, include_superseded=True)
    result_ids = [r["memory_id"] for r in results]
    assert "super-cats" in result_ids
    await client.close()
