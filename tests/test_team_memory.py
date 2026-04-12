"""Tests for team namespace promotion in tools/consolidate.py.

Tests the _promote_team_memories function: copying high-impact entries
from team namespaces to the project namespace.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.consolidate import _promote_team_memories, memory_consolidate_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.store import memory_store_impl

# ---------------------------------------------------------------------------
# In-memory backend for team promotion tests
# ---------------------------------------------------------------------------


class _InMemoryBackend(StorageBackend):
    """Simple in-memory StorageBackend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}

    def store(self, entry: MemoryEntry) -> None:
        self._data[entry.id] = entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._data.get(entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        existing = self._data.get(entry_id)
        if existing is None:
            return None
        data = existing.model_dump()
        for k, v in fields.items():
            if k == "status":
                if isinstance(v, MemoryStatus):
                    data[k] = v
                else:
                    data[k] = MemoryStatus(str(v))
            elif isinstance(v, datetime):
                data[k] = v
            else:
                data[k] = v
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MemoryStatus(data["status"])
        self._data[entry_id] = MemoryEntry(**data)
        return self._data[entry_id]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            return True
        return False

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                e for e in results if (e.status.value if isinstance(e.status, MemoryStatus) else str(e.status)) == sv
            ]
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        if namespace is not None:
            return sum(1 for e in self._data.values() if e.namespace == namespace)
        return len(self._data)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [
                e for e in results if (e.status.value if isinstance(e.status, MemoryStatus) else str(e.status)) == sv
            ]
        if namespace is not None:
            results = [e for e in results if e.namespace == namespace]
        return results[:limit]

    def close(self) -> None:
        pass


def _make_entry(
    entry_id: str,
    importance: float = 0.5,
    namespace: str = "team:sprint-37",
    tags: list[str] | None = None,
    outcome_history: list[str] | None = None,
    source_identity: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        importance=importance,
        namespace=namespace,
        tags=tags or [],
        outcome_history=outcome_history or [],
        source_identity=source_identity,
    )


# ===========================================================================
# Team Promotion Tests
# ===========================================================================


class TestPromoteTeamMemories:
    def test_copies_high_impact_entries_to_project_namespace(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))
        backend.store(_make_entry("e2", importance=0.9))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 2
        # Promoted entries should exist in project:default
        promoted_e1 = backend.get("promoted-e1")
        promoted_e2 = backend.get("promoted-e2")
        assert promoted_e1 is not None
        assert promoted_e2 is not None
        assert promoted_e1.namespace == "project:default"
        assert promoted_e2.namespace == "project:default"

    def test_skips_low_impact_entries(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.3))
        backend.store(_make_entry("e2", importance=0.5))
        backend.store(_make_entry("e3", importance=0.8))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 1
        assert result["discarded_count"] == 2
        # Only e3 should be promoted
        assert backend.get("promoted-e3") is not None
        assert backend.get("promoted-e1") is None
        assert backend.get("promoted-e2") is None

    def test_preserves_source_identity(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))

        _promote_team_memories("team:sprint-37", backend)

        promoted = backend.get("promoted-e1")
        assert promoted is not None
        assert promoted.source_identity == "team:sprint-37"

    def test_records_promoted_from_in_outcome_history(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8, outcome_history=["previous_event"]))

        _promote_team_memories("team:sprint-37", backend)

        promoted = backend.get("promoted-e1")
        assert promoted is not None
        assert len(promoted.outcome_history) == 2
        assert promoted.outcome_history[0] == "previous_event"
        assert "promoted_from:team:sprint-37" in promoted.outcome_history[1]

    def test_returns_correct_counts(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.9))
        backend.store(_make_entry("e2", importance=0.3))
        backend.store(_make_entry("e3", importance=0.8))
        backend.store(_make_entry("e4", importance=0.1))

        result = _promote_team_memories("team:sprint-37", backend)

        assert result["promoted_count"] == 2  # e1 (0.9) + e3 (0.8)
        assert result["discarded_count"] == 2  # e2 (0.3) + e4 (0.1)
        assert result["namespace_id"] == "team:sprint-37"
        assert "completed_at" in result


class TestConsolidateImplTeamDispatch:
    def test_team_namespace_dispatches_to_promotion(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8))

        result = memory_consolidate_impl("team:sprint-37", backend=backend)

        assert "promoted_count" in result
        assert result["promoted_count"] == 1

    def test_non_team_namespace_uses_normal_consolidation(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.8, namespace="project:default"))

        result = memory_consolidate_impl("project:default", backend=backend)

        # Normal consolidation path (no embedder = no clusters)
        assert "clusters_found" in result or "entries_consolidated" in result

    def test_custom_promotion_threshold(self) -> None:
        backend = _InMemoryBackend()
        backend.store(_make_entry("e1", importance=0.5))
        backend.store(_make_entry("e2", importance=0.9))

        result = _promote_team_memories("team:sprint-37", backend, promotion_threshold=0.6)

        assert result["promoted_count"] == 1  # only e2 at 0.9
        assert result["discarded_count"] == 1  # e1 at 0.5

    def test_team_namespace_promotion_writes_to_project_backend(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        team_backend = create_backend_from_config(cfg, "team:sprint-37")
        try:
            team_backend.store(_make_entry("e1", importance=0.8))

            result = memory_consolidate_impl(
                "team:sprint-37",
                backend=team_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )

            assert result["promoted_count"] == 1
            assert team_backend.get("promoted-e1") is None
        finally:
            team_backend.close()

        project_backend = create_backend_from_config(cfg, "project:default")
        try:
            promoted = project_backend.get("promoted-e1")
            assert promoted is not None
            assert promoted.namespace == "project:default"
            assert promoted.source_identity == "team:sprint-37"
        finally:
            project_backend.close()

    def test_team_promotion_marks_namespace_expiry_in_sqlite_metadata(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "team:sprint-37") as storage:
            team_backend = cast(SQLiteBackend, storage)
            memory_store_impl(
                "team discovery",
                "team:sprint-37",
                backend=team_backend,
                importance=0.8,
                config=cfg,
            )

            result = memory_consolidate_impl(
                "team:sprint-37",
                backend=team_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )

            assert result["promoted_count"] == 1
            row = team_backend._conn.execute(
                "SELECT expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
                ("team:sprint-37",),
            ).fetchone()
            assert row is not None
            assert row[0] is not None
            assert row[1] == "active"

    def test_team_wildcard_promotes_all_discovered_team_namespaces(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        for namespace, entry_id, importance in (
            ("team:sprint-37-impl", "e1", 0.8),
            ("team:sprint-37-test", "e2", 0.9),
            ("team:sprint-37-test", "e3", 0.2),
        ):
            backend = create_backend_from_config(cfg, namespace)
            try:
                backend.store(_make_entry(entry_id, importance=importance, namespace=namespace))
            finally:
                backend.close()

        default_backend = create_backend_from_config(cfg, "default")
        try:
            result = memory_consolidate_impl(
                "team:*",
                backend=default_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )
        finally:
            default_backend.close()

        assert result["promoted_count"] == 2
        assert result["discarded_count"] == 1
        assert result["namespace_id"] == "team:*"
        namespaces = cast(list[dict[str, object]], result["namespaces"])
        per_namespace = {str(item["namespace_id"]): item for item in namespaces}
        assert per_namespace["team:sprint-37-impl"]["promoted_count"] == 1
        assert per_namespace["team:sprint-37-test"]["promoted_count"] == 1
        assert per_namespace["team:sprint-37-test"]["discarded_count"] == 1

        project_backend = create_backend_from_config(cfg, "project:default")
        try:
            assert project_backend.get("promoted-e1") is not None
            assert project_backend.get("promoted-e2") is not None
            assert project_backend.get("promoted-e3") is None
        finally:
            project_backend.close()

    def test_team_wildcard_skips_when_no_team_namespaces_exist(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        default_backend = create_backend_from_config(cfg, "default")
        try:
            result = memory_consolidate_impl(
                "team:*",
                backend=default_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )
        finally:
            default_backend.close()

        assert result["status"] == "skipped"
        assert result["skipped_reason"] == "no_team_namespaces"
        assert result["promoted_count"] == 0
        assert result["discarded_count"] == 0
        assert result["namespaces"] == []

    def test_team_wildcard_continues_when_one_namespace_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        for namespace, entry_id in (
            ("team:sprint-37-impl", "e1"),
            ("team:sprint-37-test", "e2"),
        ):
            backend = create_backend_from_config(cfg, namespace)
            try:
                backend.store(_make_entry(entry_id, importance=0.8, namespace=namespace))
            finally:
                backend.close()

        def _mock_promote(
            namespace: str,
            source_backend: StorageBackend,
            *,
            target_backend: StorageBackend | None = None,
            promotion_threshold: float = 0.7,
        ) -> dict[str, object]:
            del source_backend, target_backend, promotion_threshold
            if namespace == "team:sprint-37-impl":
                raise StorageError("promotion failed")
            return {
                "promoted_count": 1,
                "discarded_count": 0,
                "namespace_id": namespace,
                "completed_at": datetime.now().isoformat(),
            }

        monkeypatch.setattr("trw_memory.tools.consolidate._promote_team_memories", _mock_promote)

        default_backend = create_backend_from_config(cfg, "default")
        try:
            result = memory_consolidate_impl(
                "team:*",
                backend=default_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )
        finally:
            default_backend.close()

        assert result["status"] == "partial"
        assert result["promoted_count"] == 1
        assert len(cast(list[dict[str, str]], result["errors"])) == 1

    def test_team_namespace_consolidation_completes_under_five_seconds_for_200_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        team_backend = create_backend_from_config(cfg, "team:sprint-37")
        try:
            for idx in range(200):
                team_backend.store(_make_entry(f"e{idx}", importance=0.8))

            started = time.perf_counter()
            result = memory_consolidate_impl(
                "team:sprint-37",
                backend=team_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )
            elapsed = time.perf_counter() - started
        finally:
            team_backend.close()

        assert result["promoted_count"] == 200
        assert elapsed < 5.0

    def test_namespace_isolation_holds_across_store_recall_delete_and_consolidate(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        team_backend = create_backend_from_config(cfg, "team:sprint-37")
        project_backend = create_backend_from_config(cfg, "project:default")
        try:
            team_backend.store(_make_entry("team-entry", importance=0.8))
            project_backend.store(_make_entry("project-entry", importance=0.8, namespace="project:default"))

            project_results = memory_recall_impl("", "project:default", backend=project_backend, config=cfg)
            project_ids = {str(item["id"]) for item in cast(list[dict[str, object]], project_results["memories"])}
            assert "project-entry" in project_ids
            assert "team-entry" not in project_ids

            assert team_backend.delete("team-entry") is True
            post_delete = memory_recall_impl("", "project:default", backend=project_backend, config=cfg)
            post_delete_ids = {str(item["id"]) for item in cast(list[dict[str, object]], post_delete["memories"])}
            assert "project-entry" in post_delete_ids
            assert "team-entry" not in post_delete_ids

            team_backend.store(_make_entry("team-entry-2", importance=0.8))
            result = memory_consolidate_impl(
                "team:sprint-37",
                backend=team_backend,
                config=cfg,
                namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            )

            assert result["promoted_count"] == 1
            final_results = memory_recall_impl("", "project:default", backend=project_backend, config=cfg)
            final_ids = {str(item["id"]) for item in cast(list[dict[str, object]], final_results["memories"])}
            assert "project-entry" in final_ids
            assert "promoted-team-entry-2" in final_ids
            assert "team-entry-2" not in final_ids
        finally:
            team_backend.close()
            project_backend.close()
