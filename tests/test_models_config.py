"""Tests for trw_memory MemoryConfig settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trw_memory.models.config import MemoryConfig


def _write_trw_config(tmp_path: Path, lines: list[str]) -> None:
    trw_dir = tmp_path / ".trw"
    trw_dir.mkdir()
    (trw_dir / "config.yaml").write_text("\n".join(lines), encoding="utf-8")


def test_memory_config_defaults() -> None:
    cfg = MemoryConfig()
    assert cfg.storage_backend == "sqlite"
    assert cfg.storage_path == ".memory"
    assert cfg.sqlite_db_name == "memory.db"
    assert cfg.embedding_dim == 384
    assert cfg.auto_generate_key is True
    assert cfg.bm25_candidates == 50
    assert cfg.vector_candidates == 50
    assert cfg.rrf_k == 60
    assert cfg.dedup_enabled is True
    assert cfg.hot_max_entries == 50
    assert cfg.warm_archive_max_score == 0.22
    assert cfg.cold_purge_max_score == 0.1
    assert cfg.decay_half_life_days == 14.0
    assert cfg.q_learning_rate == 0.15
    assert cfg.consolidation_enabled is True
    assert cfg.consolidation_max_per_cycle == 50
    assert cfg.consolidation_interval_days == 7
    assert cfg.key_rotation_backup is True
    assert cfg.local_only is False
    assert cfg.rbac_mode == "local"


def test_memory_config_consolidation_min_cluster_requires_two() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(consolidation_min_cluster=1)


def test_memory_config_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    cfg = MemoryConfig()
    assert cfg.storage_backend == "yaml"


def test_memory_config_storage_backend_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "postgres")
    with pytest.raises(ValidationError):
        MemoryConfig()


def test_memory_config_env_var_numeric_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_BM25_CANDIDATES", "100")
    cfg = MemoryConfig()
    assert cfg.bm25_candidates == 100


def test_memory_config_reads_trw_config_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trw_config(
        tmp_path,
        [
            "platform_telemetry_enabled: true",
            "platform_urls:",
            "  - https://platform.example.com",
            'platform_api_key: "yaml-key"',
            "sync_namespace: org:test",
        ],
    )
    monkeypatch.chdir(tmp_path)

    cfg = MemoryConfig()

    assert cfg.sync_enabled is True
    assert cfg.platform_url == "https://platform.example.com"
    assert cfg.platform_api_key == "yaml-key"
    assert cfg.sync_namespace == "org:test"


def test_memory_config_reads_tier_fields_from_trw_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_trw_config(
        tmp_path,
        [
            "memory_hot_max_entries: 12",
            "memory_hot_ttl_days: 3",
            "memory_cold_threshold_days: 45",
            "memory_retention_days: 180",
            "memory_score_w1: 0.5",
            "memory_score_w2: 0.2",
            "memory_score_w3: 0.3",
        ],
    )
    monkeypatch.chdir(tmp_path)

    cfg = MemoryConfig()

    assert cfg.hot_max_entries == 12
    assert cfg.hot_ttl_days == 3
    assert cfg.cold_threshold_days == 45
    assert cfg.retention_days == 180
    assert cfg.score_relevance_weight == 0.5
    assert cfg.score_recency_weight == 0.2
    assert cfg.score_importance_weight == 0.3


def test_memory_config_reads_security_fields_from_trw_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_trw_config(
        tmp_path,
        [
            "memory_encryption_enabled: true",
            "memory_auto_generate_key: false",
            "memory_rbac_enabled: true",
            "memory_rbac_mode: remote",
            "memory_namespace_roles:",
            "  project:default: reader",
            "memory_key_rotation_backup: false",
            "memory_local_only: false",
        ],
    )
    monkeypatch.chdir(tmp_path)

    cfg = MemoryConfig()

    assert cfg.encryption_enabled is True
    assert cfg.auto_generate_key is False
    assert cfg.rbac_enabled is True
    assert cfg.rbac_mode == "remote"
    assert cfg.namespace_roles == {"project:default": "reader"}
    assert cfg.key_rotation_backup is False


def test_memory_config_accepts_memory_prefixed_init_fields() -> None:
    cfg = MemoryConfig(
        memory_encryption_enabled=True,
        memory_auto_generate_key=False,
        memory_local_only=True,
        memory_rbac_enabled=True,
        memory_rbac_mode="remote",
        memory_namespace_roles={"project:default": "reader"},
        memory_key_rotation_backup=False,
    )

    assert cfg.encryption_enabled is True
    assert cfg.auto_generate_key is False
    assert cfg.local_only is True
    assert cfg.rbac_enabled is True
    assert cfg.rbac_mode == "local"
    assert cfg.namespace_roles == {"project:default": "reader"}
    assert cfg.key_rotation_backup is False


def test_memory_config_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trw_config(
        tmp_path,
        [
            "platform_telemetry_enabled: false",
            "platform_api_key: yaml-key",
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "env-key")

    cfg = MemoryConfig()

    assert cfg.sync_enabled is True
    assert cfg.platform_api_key == "env-key"


def test_memory_config_sync_enabled_keeps_local_only_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")

    cfg = MemoryConfig()

    assert cfg.sync_enabled is True
    assert cfg.local_only is False


def test_memory_config_explicit_local_only_is_preserved_with_sync_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_LOCAL_ONLY", "true")

    cfg = MemoryConfig()

    assert cfg.sync_enabled is False
    assert cfg.local_only is True


def test_memory_config_local_only_forces_local_rbac_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_LOCAL_ONLY", "true")
    monkeypatch.setenv("MEMORY_RBAC_MODE", "remote")
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SYNC_NAMESPACE", "org:test")
    monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://platform.example.com")
    monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "secret")

    cfg = MemoryConfig()

    assert cfg.local_only is True
    assert cfg.rbac_mode == "local"
    assert cfg.sync_enabled is False
    assert cfg.sync_namespace == ""
    assert cfg.platform_url == ""
    assert cfg.platform_api_key == ""


def test_memory_config_consolidation_enabled_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    cfg = MemoryConfig()
    assert cfg.consolidation_enabled is False


def test_memory_config_consolidation_similarity_threshold_invalid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_CONSOLIDATION_SIMILARITY_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        MemoryConfig()
