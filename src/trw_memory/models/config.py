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
    storage_backend: Literal["sqlite", "yaml"] = Field(default="sqlite", description="Storage backend type")
    storage_path: str = Field(default=".memory", description="Root directory for memory storage files")
    sqlite_db_name: str = Field(default="memory.db", description="SQLite database filename within namespace dir")
    embedding_dim: int = Field(default=384, gt=0, description="Dimensionality of dense embedding vectors")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformer model for embeddings")

    # Encryption
    encryption_enabled: bool = Field(default=False, description="Enable field-level encryption")
    encryption_algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm for field-level encryption")
    key_source: Literal["keyring", "env", "file"] = Field(default="env", description="Source for encryption master key")
    key_file_path: str = Field(default="~/.trw-memory/master.key", description="Path to master key file when key_source='file'")

    # Local-only mode
    local_only: bool = Field(default=True, description="Restrict to local storage only (no remote sync)")

    # RBAC
    rbac_enabled: bool = Field(default=False, description="Enable role-based access control")
    default_role: Literal["admin", "editor", "viewer", "writer"] = "writer"

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
    consolidation_min_cluster: int = Field(default=3, gt=0, description="Minimum cluster size for consolidation")
    consolidation_interval_days: int = Field(default=7, gt=0, description="Days between consolidation sweeps")

    # Audit
    audit_enabled: bool = Field(default=True, description="Enable audit logging of all memory operations")
    audit_log_path: str = Field(default=".memory/audit.jsonl", description="Path to JSONL audit log file")

    # PII
    pii_enabled: bool = Field(default=True, description="Enable PII detection in memory content")
    pii_action: Literal["block", "redact", "warn"] = "warn"
    pii_entropy_threshold: float = Field(default=4.5, gt=0.0, description="Shannon entropy threshold for PII detection")

    # Poisoning defense
    poisoning_detection_enabled: bool = Field(default=True, description="Enable statistical poisoning detection")
    poisoning_z_threshold: float = Field(default=3.0, gt=0.0, description="Z-score threshold for anomaly detection")

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
