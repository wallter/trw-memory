"""Internal settings source helpers for :mod:`trw_memory.models.config`."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import structlog
from dotenv import dotenv_values
from pydantic_settings import BaseSettings
from pydantic_settings.sources import DotEnvSettingsSource, InitSettingsSource, PydanticBaseSettingsSource
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from trw_memory.exceptions import ConfigError

__all__ = ["_TRWConfigYamlSource"]


# Temporary input tombstones only; remove at the next breaking API release.
_RETIRED_HYPE_DEFAULTS = {
    "hype_enabled": False,
    "hype_questions_per_entry": 3,
    "hype_min_question_chars": 8,
}


def _check_retired_hype_settings(raw: dict[str, object], *, source: str, textual: bool = False) -> None:
    """Check each source before precedence/filtering can hide retired activation."""
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = key.lower().removeprefix("memory_")
        if name not in _RETIRED_HYPE_DEFAULTS:
            continue
        default = _RETIRED_HYPE_DEFAULTS[name]
        if textual:
            neutral = isinstance(value, str) and (
                value.lower() in {"false", "0"} if default is False else value == str(default)
            )
        else:
            neutral = type(value) is type(default) and value == default
        if not neutral:
            raise ConfigError(f"{key} in {source}: HyPE is retired; remove this setting")
        warnings.warn(
            f"{key} in {source}: HyPE is retired; remove this neutral legacy setting",
            UserWarning,
            stacklevel=3,
        )


def _check_retired_hype_environment(dotenv_source: PydanticBaseSettingsSource) -> None:
    """Validate raw sources before ignore-empty/precedence discards evidence."""
    _check_retired_hype_settings(dict(os.environ), source="environment", textual=True)
    for path in _dotenv_files(dotenv_source):
        _check_retired_hype_settings(_dotenv_raw(dotenv_source, path), source=f"dotenv {path}", textual=True)


# F5 (2026-09-24, trw-memory 4.0.0 security fix): 'local_only' used to be
# routed through the warn-only _RETIRED_SETTINGS/_warn_retired_settings path
# below. That was a privacy flip: `local_only: true` in 3.1.0 and earlier
# forced sync_enabled=False (clearing platform_url/platform_api_key/
# sync_namespace); 4.0.0 dropped that override entirely, so a config that
# ALSO set sync_enabled/learning_sharing_enabled started syncing after the
# upgrade with only a log line as evidence. Unlike the other retired settings,
# 'local_only' fails closed -- any value, any source -- so an upgrader relying
# on it to keep sync off gets a refusal instead of a silent behavior change.
def _check_retired_local_only_settings(raw: dict[str, object], *, source: str) -> None:
    """Refuse a source that still sets 'local_only'. Checked before precedence/filtering can hide it."""
    for key in raw:
        if not isinstance(key, str):
            continue
        if key.lower().removeprefix("memory_") != "local_only":
            continue
        raise ConfigError(
            f"{key} in {source}: 'local_only' was removed in trw-memory 4.0.0 (trw-mcp 7.0.0); "
            "remove it. To keep sync off, set sync_enabled: false "
            "(and learning_sharing_enabled: false) explicitly."
        )


def _check_retired_local_only_environment(dotenv_source: PydanticBaseSettingsSource) -> None:
    """Validate raw env/dotenv sources before ignore-empty/precedence discards evidence."""
    _check_retired_local_only_settings(dict(os.environ), source="environment")
    for path in _dotenv_files(dotenv_source):
        _check_retired_local_only_settings(_dotenv_raw(dotenv_source, path), source=f"dotenv {path}")


# Removed settings, mapped to (PRD, what replaced them). Unlike HyPE's retirement
# this never raises (operator decision): any legacy value, neutral or not, logs
# one structured warning per (setting, source) per process, because
# ``extra="ignore"`` would otherwise drop it in silence.
_RERANK_REPLACEMENT = (
    "reranking always runs; the confidence floor is adaptive_rerank_floor(limit): "
    "score >= -8.0, top max(5, ceil(limit * 0.5)) rows always kept"
)
_RETIRED_SETTINGS: dict[str, tuple[str, str]] = {
    "recall_rerank": ("PRD-CORE-284", _RERANK_REPLACEMENT),
    "recall_rerank_min_score": ("PRD-CORE-284", _RERANK_REPLACEMENT),
    "recall_rerank_min_keep": ("PRD-CORE-284", _RERANK_REPLACEMENT),
    "lifecycle_use_fsrs": (
        "PRD-CORE-293",
        "none; FSRS scoring was removed and base_impact decays by recall frequency",
    ),
    "key_rotation_backup": (
        "PRD-CORE-293",
        "none; the whole SQLCipher key-rotation surface (rotate_key and its "
        "backup/checkpoint/rekey helpers) was retired — the operator does not use key "
        "rotation. Encryption at rest itself was removed later (trw-memory 4.1).",
    ),
    **dict.fromkeys(
        ("encryption_algorithm", "key_source", "key_file_path", "auto_generate_key", "master_key"),
        (
            "B71-50",
            "none; encryption at rest and its master key were removed in trw-memory 4.1 — "
            "use full-disk encryption (FileVault, BitLocker or LUKS)",
        ),
    ),
    "q_learning_rate": (
        "PRD-CORE-293",
        "none; nothing has read it since the Q-learning reward loop was removed in trw-memory 3.0.0",
    ),
}
_warned_retired_settings: set[tuple[str, str]] = set()
_logger = structlog.get_logger(__name__)


def _warn_retired_settings(raw: dict[str, object], *, source: str) -> None:
    """Warn (never raise) about a leftover removed setting in one source."""
    for key in raw:
        if not isinstance(key, str):
            continue
        name = key.lower().removeprefix("memory_")
        if name not in _RETIRED_SETTINGS or (name, source) in _warned_retired_settings:
            continue
        _warned_retired_settings.add((name, source))
        prd, replacement = _RETIRED_SETTINGS[name]
        _logger.warning(
            "retired_setting_ignored",
            setting=key,
            source=source,
            prd=prd,
            replacement=replacement,
        )


def _dotenv_files(dotenv_source: PydanticBaseSettingsSource) -> list[Path]:
    """Existing dotenv files a settings source would load."""
    if not isinstance(dotenv_source, DotEnvSettingsSource) or dotenv_source.env_file is None:
        return []
    files = dotenv_source.env_file
    paths = [files] if isinstance(files, (str, os.PathLike)) else files
    return [p for p in (Path(path).expanduser() for path in paths) if p.is_file()]


def _dotenv_raw(dotenv_source: PydanticBaseSettingsSource, path: Path) -> dict[str, object]:
    encoding = getattr(dotenv_source, "env_file_encoding", None) or "utf-8"
    return dict(dotenv_values(path, encoding=encoding))


def _warn_retired_environment(dotenv_source: PydanticBaseSettingsSource) -> None:
    """Inspect the environment and dotenv files before Pydantic drops the aliases."""
    _warn_retired_settings(dict(os.environ), source="environment")
    for path in _dotenv_files(dotenv_source):
        _warn_retired_settings(_dotenv_raw(dotenv_source, path), source=f"dotenv {path}")


def _read_trw_config_yaml() -> dict[str, object]:
    """The current project's `.trw/config.yaml`, over the machine file's platform contact switch.

    trw-mcp resolves ``platform_contact_enabled`` from ``~/.trw/config.yaml`` and then the project file
    (``TRW_PLATFORM_CONTACT_ENABLED`` above both), so the machine-wide switch stops sync here too (rc11).
    """
    machine = _read_yaml_file(Path.home() / ".trw" / "config.yaml")
    key = "platform_contact_enabled"
    return {**({key: machine[key]} if key in machine else {}), **_read_yaml_file(Path.cwd() / ".trw" / "config.yaml")}


def _read_yaml_file(config_path: Path) -> dict[str, object]:
    """Best-effort read of one TRW `config.yaml`: ``{}`` when absent or unreadable."""
    if not config_path.exists():
        return {}

    yaml = YAML(typ="safe")
    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.load(handle)
    except (OSError, YAMLError):  # trw-fail-silent-allow: an unreadable config file sets nothing (best-effort)
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
    _check_retired_hype_settings(raw, source=".trw/config.yaml")
    _check_retired_local_only_settings(raw, source=".trw/config.yaml")
    _warn_retired_settings(raw, source=".trw/config.yaml")
    mapped: dict[str, Any] = {}

    # PRD-SEC-004-FR06: derive sync_enabled (learning-content sync) from the
    # NEW learning_sharing_enabled consent flag, NOT from platform_telemetry_enabled
    # (anonymous usage telemetry). The explicit ``sync_enabled`` alias is retained
    # and takes precedence. The legacy platform_telemetry_enabled key MUST NOT gate
    # learning-content sync — that conflated two independent consents.
    if (sync_enabled := _first_non_none(raw, "sync_enabled", "learning_sharing_enabled")) is not None:
        mapped["sync_enabled"] = sync_enabled
    for target, aliases in (
        ("sync_min_importance", ("sync_min_importance",)),
        ("sync_namespace", ("sync_namespace",)),
        ("platform_contact_enabled", ("platform_contact_enabled",)),
        ("platform_api_key", ("platform_api_key",)),
        (
            "embedding_trust_remote_code",
            ("embedding_trust_remote_code", "memory_embedding_trust_remote_code"),
        ),
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
        ("rbac_enabled", ("rbac_enabled", "memory_rbac_enabled")),
        ("rbac_mode", ("rbac_mode", "memory_rbac_mode")),
        ("namespace_roles", ("namespace_roles", "memory_namespace_roles")),
        ("memory_recovery_policy", ("memory_recovery_policy", "recovery_policy")),
        ("memory_corrupt_backup_keep", ("memory_corrupt_backup_keep", "corrupt_backup_keep")),
        (
            "memory_recovery_rebuild_from_cold",
            ("memory_recovery_rebuild_from_cold", "recovery_rebuild_from_cold"),
        ),
        (
            "memory_recovery_inline_max_bytes",
            ("memory_recovery_inline_max_bytes", "recovery_inline_max_bytes"),
        ),
        ("security_maintenance_inline", ("security_maintenance_inline", "memory_security_maintenance_inline")),
        (
            "memory_integrity_check_interval_minutes",
            ("memory_integrity_check_interval_minutes", "integrity_check_interval_minutes"),
        ),
        ("memory_snapshot_daily_keep", ("memory_snapshot_daily_keep", "snapshot_daily_keep")),
        ("memory_snapshot_weekly_keep", ("memory_snapshot_weekly_keep", "snapshot_weekly_keep")),
        # PRD-CORE-253 FR03 loopback daemon. There is deliberately no bind-host
        # key: the host is a module constant, so it cannot be mistyped into a
        # network-reachable listener.
        ("memory_daemon_port", ("memory_daemon_port", "daemon_port")),
        (
            "memory_daemon_idle_shutdown_seconds",
            ("memory_daemon_idle_shutdown_seconds", "daemon_idle_shutdown_seconds"),
        ),
        (
            "memory_daemon_startup_timeout_seconds",
            ("memory_daemon_startup_timeout_seconds", "daemon_startup_timeout_seconds"),
        ),
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
