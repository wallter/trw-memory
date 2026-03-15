"""Standalone memory configuration.

Uses pydantic-settings with MEMORY_* environment variable prefix.
Defaults match the TRWConfig memory-related values for backward compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    storage_backend: Literal["sqlite", "yaml"] = "sqlite"
    storage_path: str = ".memory"
    sqlite_db_name: str = "memory.db"
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"

    # Encryption
    encryption_enabled: bool = False
    encryption_algorithm: str = "AES-256-GCM"
    key_source: Literal["keyring", "env", "file"] = "env"
    key_file_path: str = "~/.trw-memory/master.key"

    # Local-only mode
    local_only: bool = True

    # RBAC
    rbac_enabled: bool = False
    default_role: str = "writer"

    # Retrieval
    bm25_candidates: int = 50
    vector_candidates: int = 50
    rrf_k: int = 60

    # Dedup
    dedup_enabled: bool = True
    dedup_skip_threshold: float = 0.95
    dedup_merge_threshold: float = 0.85

    # Tiers
    hot_max_entries: int = 50
    hot_ttl_days: int = 7
    cold_threshold_days: int = 90
    retention_days: int = 365

    # Scoring
    decay_half_life_days: float = 14.0
    decay_use_exponent: float = 0.6
    q_learning_rate: float = 0.15
    score_relevance_weight: float = 0.4
    score_recency_weight: float = 0.3
    score_importance_weight: float = 0.3

    # Consolidation
    consolidation_enabled: bool = True
    consolidation_similarity_threshold: float = 0.75
    consolidation_min_cluster: int = 3
    consolidation_interval_days: int = 7

    # Audit
    audit_enabled: bool = True
    audit_log_path: str = ".memory/audit.jsonl"

    # PII
    pii_enabled: bool = True
    pii_action: str = "warn"  # block, redact, warn
    pii_entropy_threshold: float = 4.5

    # Poisoning defense
    poisoning_detection_enabled: bool = True
    poisoning_z_threshold: float = 3.0

    # Sync configuration (PRD-CORE-047)
    sync_enabled: bool = False
    sync_min_importance: float = Field(default=0.7, ge=0.0, le=1.0, description="Min importance to publish remotely")
    sync_namespace: str = ""
    platform_url: str = ""
    platform_api_key: str = ""

    @model_validator(mode="after")
    def _check_weight_sum(self) -> MemoryConfig:
        total = self.score_relevance_weight + self.score_recency_weight + self.score_importance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.3f}")
        return self
