"""Promotion-path tests for team memory consolidation."""

from __future__ import annotations

from pathlib import Path

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.tools.consolidate import _promote_team_memories, memory_consolidate_impl

from ._test_team_memory_support import _InMemoryBackend, _make_entry


def test_copies_high_impact_entries_to_project_namespace() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.8))
    backend.store(_make_entry("e2", importance=0.9))

    result = _promote_team_memories("team:sprint-37", backend)

    assert result["promoted_count"] == 2
    promoted_e1 = backend.get("promoted-e1", namespace="default")
    promoted_e2 = backend.get("promoted-e2", namespace="default")
    assert promoted_e1 is not None
    assert promoted_e2 is not None
    assert promoted_e1.namespace == "project:default"
    assert promoted_e2.namespace == "project:default"


def test_skips_low_impact_entries() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.3))
    backend.store(_make_entry("e2", importance=0.5))
    backend.store(_make_entry("e3", importance=0.8))

    result = _promote_team_memories("team:sprint-37", backend)

    assert result["promoted_count"] == 1
    assert result["discarded_count"] == 2
    assert backend.get("promoted-e3", namespace="default") is not None
    assert backend.get("promoted-e1", namespace="default") is None
    assert backend.get("promoted-e2", namespace="default") is None


def test_preserves_source_identity() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.8))

    _promote_team_memories("team:sprint-37", backend)

    promoted = backend.get("promoted-e1", namespace="default")
    assert promoted is not None
    assert promoted.source_identity == "team:sprint-37"


def test_records_promoted_from_in_outcome_history() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.8, outcome_history=["previous_event"]))

    _promote_team_memories("team:sprint-37", backend)

    promoted = backend.get("promoted-e1", namespace="default")
    assert promoted is not None
    assert len(promoted.outcome_history) == 2
    assert promoted.outcome_history[0] == "previous_event"
    assert "promoted_from:team:sprint-37" in promoted.outcome_history[1]


def test_returns_correct_counts() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.9))
    backend.store(_make_entry("e2", importance=0.3))
    backend.store(_make_entry("e3", importance=0.8))
    backend.store(_make_entry("e4", importance=0.1))

    result = _promote_team_memories("team:sprint-37", backend)

    assert result["promoted_count"] == 2
    assert result["discarded_count"] == 2
    assert result["namespace_id"] == "team:sprint-37"
    assert "completed_at" in result


def test_team_namespace_dispatches_to_promotion() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.8))

    result = memory_consolidate_impl("team:sprint-37", backend=backend)

    assert "promoted_count" in result
    assert result["promoted_count"] == 1


def test_non_team_namespace_uses_normal_consolidation() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.8, namespace="project:default"))

    result = memory_consolidate_impl("project:default", backend=backend)

    assert "clusters_found" in result or "entries_consolidated" in result


def test_custom_promotion_threshold() -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.5))
    backend.store(_make_entry("e2", importance=0.9))

    result = _promote_team_memories("team:sprint-37", backend, promotion_threshold=0.6)

    assert result["promoted_count"] == 1
    assert result["discarded_count"] == 1


def test_team_namespace_promotion_writes_to_project_backend(tmp_path: Path) -> None:
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
        assert team_backend.get("promoted-e1", namespace="default") is None
    finally:
        team_backend.close()

    project_backend = create_backend_from_config(cfg, "project:default")
    try:
        promoted = project_backend.get("promoted-e1", namespace="project:default")
        assert promoted is not None
        assert promoted.namespace == "project:default"
        assert promoted.source_identity == "team:sprint-37"
    finally:
        project_backend.close()
