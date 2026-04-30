"""Internal settings source helpers for :mod:`trw_memory.models.config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings
from pydantic_settings.sources import InitSettingsSource
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

__all__ = ["_TRWConfigYamlSource"]


def _read_trw_config_yaml() -> dict[str, object]:
    """Best-effort read of the current project's `.trw/config.yaml`."""
    config_path = Path.cwd() / ".trw" / "config.yaml"
    if not config_path.exists():
        return {}

    yaml = YAML(typ="safe")
    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.load(handle)
    except (OSError, YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_non_none(raw: dict[str, object], *aliases: str) -> object | None:
    for alias in aliases:
        if raw.get(alias) is not None:
            return raw[alias]
    return None


def _first_truthy_item(values: object) -> object | None:
    if isinstance(values, list):
        return next((candidate for candidate in values if candidate), None)
    return None


def _map_trw_config_yaml_to_memory_settings(raw: dict[str, object]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}

    if (sync_enabled := _first_non_none(raw, "sync_enabled", "platform_telemetry_enabled")) is not None:
        mapped["sync_enabled"] = sync_enabled
    for target, aliases in (
        ("sync_min_importance", ("sync_min_importance",)),
        ("sync_namespace", ("sync_namespace",)),
        ("platform_api_key", ("platform_api_key",)),
        ("local_only", ("local_only", "memory_local_only")),
        ("hot_max_entries", ("hot_max_entries", "memory_hot_max_entries")),
        ("hot_ttl_days", ("hot_ttl_days", "memory_hot_ttl_days")),
        ("cold_threshold_days", ("cold_threshold_days", "memory_cold_threshold_days")),
        ("retention_days", ("retention_days", "memory_retention_days")),
        ("score_relevance_weight", ("score_relevance_weight", "memory_score_w1")),
        ("score_recency_weight", ("score_recency_weight", "memory_score_w2")),
        ("score_importance_weight", ("score_importance_weight", "memory_score_w3")),
        ("warm_archive_max_score", ("warm_archive_max_score",)),
        ("cold_purge_max_score", ("cold_purge_max_score",)),
        ("encryption_enabled", ("encryption_enabled", "memory_encryption_enabled")),
        ("auto_generate_key", ("auto_generate_key", "memory_auto_generate_key")),
        ("rbac_enabled", ("rbac_enabled", "memory_rbac_enabled")),
        ("rbac_mode", ("rbac_mode", "memory_rbac_mode")),
        ("namespace_roles", ("namespace_roles", "memory_namespace_roles")),
        ("key_rotation_backup", ("key_rotation_backup", "memory_key_rotation_backup")),
        ("memory_recovery_policy", ("memory_recovery_policy", "recovery_policy")),
        ("memory_corrupt_backup_keep", ("memory_corrupt_backup_keep", "corrupt_backup_keep")),
        (
            "memory_recovery_rebuild_from_cold",
            ("memory_recovery_rebuild_from_cold", "recovery_rebuild_from_cold"),
        ),
        (
            "memory_integrity_check_interval_minutes",
            ("memory_integrity_check_interval_minutes", "integrity_check_interval_minutes"),
        ),
        (
            "memory_concurrent_writer_warn_threshold",
            ("memory_concurrent_writer_warn_threshold", "concurrent_writer_warn_threshold"),
        ),
        ("memory_snapshot_enabled", ("memory_snapshot_enabled", "snapshot_enabled")),
        ("memory_snapshot_daily_keep", ("memory_snapshot_daily_keep", "snapshot_daily_keep")),
        ("memory_snapshot_weekly_keep", ("memory_snapshot_weekly_keep", "snapshot_weekly_keep")),
        ("memory_snapshot_publish_hash", ("memory_snapshot_publish_hash", "snapshot_publish_hash")),
    ):
        if (value := _first_non_none(raw, *aliases)) is not None:
            mapped[target] = value

    direct_url = raw.get("platform_url")
    if direct_url is not None:
        mapped["platform_url"] = direct_url
    else:
        first_url = _first_truthy_item(raw.get("platform_urls"))
        if first_url is not None:
            mapped["platform_url"] = first_url

    return mapped


class _TRWConfigYamlSource(InitSettingsSource):
    """Map framework config keys onto the subset owned by trw-memory."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls, _map_trw_config_yaml_to_memory_settings(_read_trw_config_yaml()))
