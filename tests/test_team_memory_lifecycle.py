"""Lifecycle and isolation tests for team memory consolidation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.persistence import read_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.store import memory_store_impl

from ._test_team_memory_support import _make_entry


def test_team_promotion_marks_namespace_expiry_in_sqlite_metadata(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

    with create_backend_from_config(cfg, "team:sprint-37") as storage:
        team_backend = cast("SQLiteBackend", storage)
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
        assert row[1] == "completed"


def test_team_promotion_marks_namespace_completion_in_yaml_metadata(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

    team_backend = create_backend_from_config(cfg, "team:sprint-37")
    try:
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
    finally:
        team_backend.close()

    metadata = cast("dict[str, object]", read_yaml(tmp_path / "team_sprint-37" / "namespace_lifecycle.yaml"))
    assert metadata["team_id"] == "sprint-37"
    assert metadata["expires_at"] is not None
    assert metadata["status"] == "completed"


def test_team_namespace_repeat_consolidation_skips_after_completion(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

    team_backend = create_backend_from_config(cfg, "team:sprint-37")
    try:
        memory_store_impl(
            "team discovery",
            "team:sprint-37",
            backend=team_backend,
            importance=0.8,
            config=cfg,
        )

        first = memory_consolidate_impl(
            "team:sprint-37",
            backend=team_backend,
            config=cfg,
            namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
        )
        second = memory_consolidate_impl(
            "team:sprint-37",
            backend=team_backend,
            config=cfg,
            namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
        )
    finally:
        team_backend.close()

    assert first["promoted_count"] == 1
    assert second["status"] == "skipped"
    assert second["skipped_reason"] == "already_completed"
    assert second["promoted_count"] == 0


def test_team_namespace_consolidation_completes_under_five_seconds_for_200_entries(tmp_path: Path) -> None:
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


def test_namespace_isolation_holds_across_store_recall_delete_and_consolidate(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

    team_backend = create_backend_from_config(cfg, "team:sprint-37")
    project_backend = create_backend_from_config(cfg, "project:default")
    try:
        team_backend.store(_make_entry("team-entry", importance=0.8))
        project_backend.store(_make_entry("project-entry", importance=0.8, namespace="project:default"))

        project_results = memory_recall_impl("", "project:default", backend=project_backend, config=cfg)
        project_ids = {str(item["id"]) for item in cast("list[dict[str, object]]", project_results["memories"])}
        assert "project-entry" in project_ids
        assert "team-entry" not in project_ids

        assert team_backend.delete("team-entry") is True
        post_delete = memory_recall_impl("", "project:default", backend=project_backend, config=cfg)
        post_delete_ids = {str(item["id"]) for item in cast("list[dict[str, object]]", post_delete["memories"])}
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
        final_ids = {str(item["id"]) for item in cast("list[dict[str, object]]", final_results["memories"])}
        assert "project-entry" in final_ids
        assert "promoted-team-entry-2" in final_ids
        assert "team-entry-2" not in final_ids
    finally:
        team_backend.close()
        project_backend.close()
