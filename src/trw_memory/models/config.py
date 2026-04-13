"""Standalone memory configuration.

Uses pydantic-settings with MEMORY_* environment variable prefix.
Defaults match the TRWConfig memory-related values for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import InitSettingsSource, PydanticBaseSettingsSource
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

__all__ = ["MemoryConfig"]


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


class _TRWConfigYamlSource(InitSettingsSource):
    """Map framework config keys onto the subset owned by trw-memory."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        raw = _read_trw_config_yaml()
        mapped: dict[str, Any] = {}

        def _map_first(target: str, *aliases: str) -> None:
            for alias in aliases:
                if raw.get(alias) is not None:
                    mapped[target] = raw[alias]
                    return

        sync_enabled = raw.get("sync_enabled", raw.get("platform_telemetry_enabled"))
        if sync_enabled is not None:
            mapped["sync_enabled"] = sync_enabled
        if raw.get("sync_min_importance") is not None:
            mapped["sync_min_importance"] = raw["sync_min_importance"]
        if raw.get("sync_namespace") is not None:
            mapped["sync_namespace"] = raw["sync_namespace"]
        if raw.get("platform_api_key") is not None:
            mapped["platform_api_key"] = raw["platform_api_key"]
        _map_first("local_only", "local_only", "memory_local_only")

        direct_url = raw.get("platform_url")
        platform_urls = raw.get("platform_urls")
        if direct_url is not None:
            mapped["platform_url"] = direct_url
        elif isinstance(platform_urls, list):
            first_url = next((candidate for candidate in platform_urls if candidate), None)
            if first_url is not None:
                mapped["platform_url"] = first_url

        # Keep the standalone package configurable from the framework config
        # file so tier policies can be changed without a package-local env file.
        _map_first("hot_max_entries", "hot_max_entries", "memory_hot_max_entries")
        _map_first("hot_ttl_days", "hot_ttl_days", "memory_hot_ttl_days")
        _map_first("cold_threshold_days", "cold_threshold_days", "memory_cold_threshold_days")
        _map_first("retention_days", "retention_days", "memory_retention_days")
        _map_first("score_relevance_weight", "score_relevance_weight", "memory_score_w1")
        _map_first("score_recency_weight", "score_recency_weight", "memory_score_w2")
        _map_first("score_importance_weight", "score_importance_weight", "memory_score_w3")
        _map_first("warm_archive_max_score", "warm_archive_max_score")
        _map_first("cold_purge_max_score", "cold_purge_max_score")
        _map_first("encryption_enabled", "encryption_enabled", "memory_encryption_enabled")
        _map_first("auto_generate_key", "auto_generate_key", "memory_auto_generate_key")
        _map_first("rbac_enabled", "rbac_enabled", "memory_rbac_enabled")
        _map_first("rbac_mode", "rbac_mode", "memory_rbac_mode")
        _map_first("namespace_roles", "namespace_roles", "memory_namespace_roles")
        _map_first("key_rotation_backup", "key_rotation_backup", "memory_key_rotation_backup")
        _map_first("memory_recovery_policy", "memory_recovery_policy", "recovery_policy")

        super().__init__(settings_cls, mapped)


class MemoryConfig(BaseSettings):
    """Configuration for the trw-memory package.

    All settings can be overridden via ``MEMORY_*`` environment variables.
    Example: ``MEMORY_STORAGE_BACKEND=yaml`` selects YAML-only storage.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        case_sensitive=False,
        extra="ignore",
    )

    # Storage
    storage_backend: Literal["sqlite", "yaml"] = Field(default="sqlite", description="Storage backend type")
    storage_path: str = Field(default=".memory", description="Root directory for memory storage files")
    sqlite_db_name: str = Field(default="memory.db", description="SQLite database filename within namespace dir")
    embedding_dim: int = Field(default=384, gt=0, description="Dimensionality of dense embedding vectors")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformer model for embeddings")

    # Encryption
    encryption_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("encryption_enabled", "memory_encryption_enabled"),
        description="Enable field-level encryption",
    )
    encryption_algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm for field-level encryption")
    key_source: Literal["keyring", "env", "file"] = Field(default="env", description="Source for encryption master key")
    key_file_path: str = Field(default="~/.trw-memory/master.key", description="Path to master key file when key_source='file'")
    auto_generate_key: bool = Field(
        default=True,
        validation_alias=AliasChoices("auto_generate_key", "memory_auto_generate_key"),
        description="Generate and persist a master key if none exists",
    )
    key_rotation_backup: bool = Field(
        default=True,
        validation_alias=AliasChoices("key_rotation_backup", "memory_key_rotation_backup"),
        description="Create a backup before key rotation work",
    )

    # Local-only mode
    local_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("local_only", "memory_local_only"),
        description="Restrict to local storage only (no remote sync)",
    )

    # RBAC
    rbac_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("rbac_enabled", "memory_rbac_enabled"),
        description="Enable role-based access control",
    )
    rbac_mode: Literal["local", "remote"] = Field(
        default="local",
        validation_alias=AliasChoices("rbac_mode", "memory_rbac_mode"),
        description="RBAC enforcement layer",
    )
    default_role: Literal["admin", "reader", "writer", "none"] = "admin"
    namespace_roles: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("namespace_roles", "memory_namespace_roles"),
        description="Per-namespace role overrides used when RBAC is enabled",
    )

    # Retrieval
    bm25_candidates: int = Field(default=50, gt=0, description="Number of BM25 candidates to consider")
    vector_candidates: int = Field(default=50, gt=0, description="Number of dense vector candidates to consider")
    rrf_k: int = Field(default=60, gt=0, description="RRF constant k for reciprocal rank fusion")

    # Dedup
    dedup_enabled: bool = Field(default=True, description="Enable semantic deduplication")
    dedup_skip_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Similarity threshold for skipping duplicate entries")
    dedup_merge_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity threshold for merging similar entries")

    # Tiers
    hot_max_entries: int = Field(default=50, gt=0, description="Maximum entries in the hot tier")
    hot_ttl_days: int = Field(default=7, gt=0, description="TTL in days for hot tier entries")
    cold_threshold_days: int = Field(default=90, gt=0, description="Days after which entries move to cold tier")
    retention_days: int = Field(default=365, gt=0, description="Days before entries are purged from cold tier")
    warm_archive_max_score: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Maximum composite tier score allowed before a warm entry is archived to cold storage",
    )
    cold_purge_max_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum composite tier score allowed before a cold entry is purged",
    )

    # Scoring
    decay_half_life_days: float = Field(default=14.0, gt=0.0, description="Half-life in days for recency decay")
    decay_use_exponent: float = Field(default=0.6, ge=0.0, le=1.0, description="Exponent for utility-based decay")
    q_learning_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Q-learning update rate")
    score_relevance_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for relevance in composite score")
    score_recency_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for recency in composite score")
    score_importance_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for importance in composite score")

    # Consolidation
    consolidation_enabled: bool = Field(default=True, description="Enable periodic consolidation of similar entries")
    consolidation_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Cosine similarity threshold for consolidation clustering")
    consolidation_min_cluster: int = Field(default=3, ge=2, description="Minimum cluster size for consolidation")
    consolidation_max_per_cycle: int = Field(default=50, gt=0, description="Maximum entries to evaluate in one consolidation cycle")
    consolidation_interval_days: int = Field(default=7, gt=0, description="Days between consolidation sweeps")

    # Audit
    audit_enabled: bool = Field(default=True, description="Enable audit logging of all memory operations")
    audit_log_path: str = Field(default="", description="Path to JSONL audit log file")
    audit_retention_days: int = Field(default=365, gt=0, description="Days of audit history to retain before compaction")
    fsync_on_append: bool = Field(default=False, description="Call os.fsync() after each audit log write for crash safety")

    # PII
    pii_enabled: bool = Field(default=True, description="Enable PII detection in memory content")
    pii_action: Literal["block", "redact", "warn"] = "warn"
    pii_entropy_threshold: float = Field(default=4.5, gt=0.0, description="Shannon entropy threshold for PII detection")
    pii_custom_patterns: list[str] = Field(default_factory=list, description="Additional regex patterns treated as custom PII")

    # Poisoning defense
    poisoning_detection_enabled: bool = Field(default=True, description="Enable statistical poisoning detection")
    poisoning_z_threshold: float = Field(default=3.0, gt=0.0, description="Z-score threshold for anomaly detection")
    quarantine_path: str = Field(default="", description="Directory where quarantined entries are written")
    max_entry_chars: int = Field(default=10_240, gt=0, description="Maximum combined content/detail character count")
    max_memory_writes_per_minute: int = Field(default=10, ge=0, description="Per-session write limit enforced over a rolling minute")
    rate_limit_state_path: str = Field(default="", description="Path to the persisted write-rate limiter state file")

    # Recovery policy (PRD-CORE-138)
    memory_recovery_policy: Literal["strict", "empty_ok"] = Field(
        default="strict",
        validation_alias=AliasChoices("memory_recovery_policy", "recovery_policy"),
        description=(
            "Behavior when DB corruption salvage yields 0 rows on a non-empty backup: "
            "'strict' raises CorruptDatabaseUnsalvageableError (default); "
            "'empty_ok' preserves legacy silent-empty fallback."
        ),
    )

    # Sync configuration (PRD-CORE-047)
    sync_enabled: bool = Field(default=False, description="Enable remote platform sync")
    sync_min_importance: float = Field(default=0.7, ge=0.0, le=1.0, description="Min importance to publish remotely")
    sync_namespace: str = Field(default="", description="Remote namespace for sync operations")
    platform_url: str = Field(default="", description="TRW platform API URL for remote sync")
    platform_api_key: str = Field(default="", description="API key for platform authentication")

    def __repr__(self) -> str:
        """Concise repr showing only key operational settings."""
        return (
            f"MemoryConfig("
            f"backend={self.storage_backend!r}, "
            f"path={self.storage_path!r}, "
            f"encryption={self.encryption_enabled}, "
            f"rbac={self.rbac_enabled})"
        )

    @model_validator(mode="after")
    def _check_weight_sum(self) -> MemoryConfig:
        total = self.score_relevance_weight + self.score_recency_weight + self.score_importance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.3f}")
        return self

    @model_validator(mode="after")
    def _enable_sync_turns_off_default_local_only(self) -> MemoryConfig:
        """Apply local-only overrides to every shipped remote-capable config surface."""
        if self.local_only and self.rbac_mode != "local":
            self.rbac_mode = "local"
        if self.local_only:
            self.sync_enabled = False
            self.sync_namespace = ""
            self.platform_url = ""
            self.platform_api_key = ""
        return self

    @model_validator(mode="after")
    def _derive_security_paths(self) -> MemoryConfig:
        """Keep audit/quarantine/rate-limit files outside the active storage root."""
        security_root = Path(self.storage_path).parent / ".trw" / "security"
        if not self.audit_log_path:
            self.audit_log_path = str(security_root / "audit.jsonl")
        if not self.quarantine_path:
            self.quarantine_path = str(security_root / "quarantine")
        if not self.rate_limit_state_path:
            self.rate_limit_state_path = str(security_root / "rate_limits.yaml")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load `.trw/config.yaml` after env vars, preserving env precedence."""
        return (
            init_settings,
            env_settings,
            _TRWConfigYamlSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
