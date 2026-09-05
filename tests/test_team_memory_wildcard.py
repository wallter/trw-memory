"""Wildcard team namespace consolidation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.store import memory_store_impl

from ._test_team_memory_support import _make_entry


def test_team_wildcard_promotes_all_discovered_team_namespaces(tmp_path: Path) -> None:
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
    namespaces = cast("list[dict[str, object]]", result["namespaces"])
    per_namespace = {str(item["namespace_id"]): item for item in namespaces}
    assert per_namespace["team:sprint-37-impl"]["promoted_count"] == 1
    assert per_namespace["team:sprint-37-test"]["promoted_count"] == 1
    assert per_namespace["team:sprint-37-test"]["discarded_count"] == 1

    project_backend = create_backend_from_config(cfg, "project:default")
    try:
        assert project_backend.get("promoted-e1", namespace="project:default") is not None
        assert project_backend.get("promoted-e2", namespace="project:default") is not None
        assert project_backend.get("promoted-e3", namespace="project:default") is None
    finally:
        project_backend.close()


def test_team_wildcard_skips_completed_namespaces(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

    completed_backend = create_backend_from_config(cfg, "team:sprint-37-done")
    try:
        memory_store_impl(
            "completed discovery",
            "team:sprint-37-done",
            backend=completed_backend,
            importance=0.8,
            config=cfg,
        )
        NamespaceManager(completed_backend).mark_team_namespace_completed(
            "team:sprint-37-done",
            completed_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
    finally:
        completed_backend.close()

    active_backend = create_backend_from_config(cfg, "team:sprint-37-active")
    try:
        memory_store_impl(
            "active discovery",
            "team:sprint-37-active",
            backend=active_backend,
            importance=0.8,
            config=cfg,
        )
    finally:
        active_backend.close()

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

    assert result["promoted_count"] == 1
    namespaces = cast("list[dict[str, object]]", result["namespaces"])
    assert [str(item["namespace_id"]) for item in namespaces] == ["team:sprint-37-active"]


def test_team_wildcard_skips_when_no_team_namespaces_exist(tmp_path: Path) -> None:
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
            "completed_at": datetime.now(timezone.utc).isoformat(),
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
    assert len(cast("list[dict[str, str]]", result["errors"])) == 1
