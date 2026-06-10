"""Standalone memory configuration.

Uses pydantic-settings with MEMORY_* environment variable prefix.
Defaults match the TRWConfig memory-related values for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource

from trw_memory.models._config_sources import _TRWConfigYamlSource

__all__ = ["MemoryConfig"]


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
    encryption_algorithm: str = Field(
        default="AES-256-GCM", description="Encryption algorithm for field-level encryption"
    )
    key_source: Literal["keyring", "env", "file"] = Field(default="env", description="Source for encryption master key")
    key_file_path: str = Field(
        default="~/.trw-memory/master.key", description="Path to master key file when key_source='file'"
    )
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
    rrf_importance_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("rrf_importance_alpha", "memory_rrf_importance_alpha"),
        description=(
            "R-FUSION-001: blend weight on the (normalised) RRF position score "
            "vs. the entry's importance in hybrid_search. final = alpha * "
            "rrf_norm + (1 - alpha) * importance. 1.0 = pure position (legacy "
            "behaviour, ignores importance); 0.0 = pure importance. Default 0.7 "
            "lets a high-impact entry edge out an equally-ranked low-impact one "
            "without overriding strong relevance signal."
        ),
    )
    hybrid_search_candidate_pool_size: int = Field(
        default=1000,
        ge=10,
        description=(
            "PRD-DIST-2047 c796: max entries loaded from a namespace for the "
            "hybrid_search candidate pool. Pre-c796 the pool was capped at "
            "limit*5 (=50 for default limit=10), which silently lost targets "
            "ranked past position 50 on namespaces > 50 records. Set higher "
            "for very large namespaces; recall-time cost is O(namespace_size) "
            "for BM25 + O(namespace_size x embedding_dim) for dense search "
            "when the auto-scaled bm25_candidates/vector_candidates lift the "
            "50-cap floor."
        ),
    )
    recall_confidence_filter: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("recall_confidence_filter", "memory_recall_confidence_filter"),
        description=(
            "PRD-DIST-2049 c802: opt-in recall-time confidence floor. When "
            "set, records with metadata['confidence'] < value are suppressed "
            "from MemoryClient.recall() results between merge_tier_results "
            "and apply_source_policy. Default None = filter OFF (current "
            "behavior bit-for-bit). Closes the c800/c801 contamination lever "
            "(2-5pp absolute SC2 lift on full-corpus shapes across "
            "Python/TS/PHP)."
        ),
    )
    recall_filter_historical_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("recall_filter_historical_only", "memory_recall_filter_historical_only"),
        description=(
            "PRD-DIST-2049 c802: opt-in suppression of F2-softened records. "
            "When True, records with metadata['currentness_status'] == "
            "'historical_only' are suppressed from MemoryClient.recall() "
            "results between merge_tier_results and apply_source_policy. "
            "Default False = filter OFF. Mirrors the trw-distill eval-side "
            "_retrieval_policy_filter behaviour into the memory-side recall "
            "path (closes the c800 c763 finding that the F2 label is "
            "necessary but not fully sufficient at recall time)."
        ),
    )
    recall_top_k_multiplier: int = Field(
        default=3,
        ge=1,
        le=50,
        validation_alias=AliasChoices("recall_top_k_multiplier", "memory_recall_top_k_multiplier"),
        description=(
            "PRD-DIST-2050 c804: depth multiplier for the hybrid_search "
            "candidate pool returned by _try_hybrid_recall. Effective top_k "
            "= limit * recall_top_k_multiplier (default 3 → top-30 for "
            "limit=10, matches pre-c804 hardcoded behaviour). Raise to 10 "
            "or higher when the recall-time admission filter (PRD-DIST-2049) "
            "is enabled on pure-zombie corpora — c803 found those filters "
            "can suppress but not promote baseline records ranked past the "
            "current top-30 candidate pool depth. Capped at 50 to keep "
            "downstream per-result cost bounded."
        ),
    )
    recall_preserve_hybrid_order: bool = Field(
        default=True,
        validation_alias=AliasChoices("recall_preserve_hybrid_order", "memory_recall_preserve_hybrid_order"),
        description=(
            "PRD-DIST-2051 c806 / PRD-DIST-2058 c817: when True, "
            "merge_tier_results returns "
            "local_results[:limit] (preserving the BM25+dense+RRF ordering "
            "from _try_hybrid_recall) whenever len(local_results) >= limit. "
            "Skips the compute_importance_score rescore that mixes hybrid "
            "RRF (1/(1+rank)) and tier-only entry_utility (absolute) scales. "
            "c805 trace showed all 4 missing hono baselines were at "
            "hybrid_rank=2 but got pushed past top-10 by the rescore; "
            "c811-c815 showed default-ON is robust across curated-query "
            "oracles, languages, and K-depths. Set "
            "MEMORY_RECALL_PRESERVE_HYBRID_ORDER=false to opt out."
        ),
    )

    # Dedup
    dedup_enabled: bool = Field(default=True, description="Enable semantic deduplication")
    dedup_skip_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Similarity threshold for skipping duplicate entries"
    )
    dedup_merge_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Similarity threshold for merging similar entries"
    )

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

    # Forced importance-tier distribution caps (mirror trw-mcp impact_tier_*_cap)
    impact_tier_critical_cap: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of active entries allowed in the critical importance tier (>=0.9)",
    )
    impact_tier_high_cap: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of active entries allowed in the high importance tier (0.7-0.89)",
    )

    # Scoring
    decay_half_life_days: float = Field(default=14.0, gt=0.0, description="Half-life in days for recency decay")
    decay_use_exponent: float = Field(default=0.6, ge=0.0, le=1.0, description="Exponent for utility-based decay")
    q_learning_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Q-learning update rate")
    score_relevance_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for relevance in composite score"
    )
    score_recency_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for recency in composite score"
    )
    score_importance_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for importance in composite score"
    )

    # Consolidation
    consolidation_enabled: bool = Field(default=True, description="Enable periodic consolidation of similar entries")
    consolidation_similarity_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Cosine similarity threshold for consolidation clustering"
    )
    consolidation_min_cluster: int = Field(default=3, ge=2, description="Minimum cluster size for consolidation")
    consolidation_max_per_cycle: int = Field(
        default=50, gt=0, description="Maximum entries to evaluate in one consolidation cycle"
    )
    consolidation_interval_days: int = Field(default=7, gt=0, description="Days between consolidation sweeps")

    # Audit
    audit_enabled: bool = Field(default=True, description="Enable audit logging of all memory operations")
    audit_log_path: str = Field(default="", description="Path to JSONL audit log file")
    audit_retention_days: int = Field(
        default=365, gt=0, description="Days of audit history to retain before compaction"
    )
    fsync_on_append: bool = Field(
        default=False, description="Call os.fsync() after each audit log write for crash safety"
    )
    security_maintenance_inline: bool = Field(
        default=True,
        description=(
            "When True, audit retention maintenance drains immediately at the operation boundary. "
            "When False, maintenance is enqueued for an explicit deferred drain."
        ),
    )

    # PII
    pii_enabled: bool = Field(default=True, description="Enable PII detection in memory content")
    pii_action: Literal["block", "redact", "warn"] = "warn"
    pii_entropy_threshold: float = Field(default=4.5, gt=0.0, description="Shannon entropy threshold for PII detection")
    pii_custom_patterns: list[str] = Field(
        default_factory=list, description="Additional regex patterns treated as custom PII"
    )

    # Poisoning defense
    poisoning_detection_enabled: bool = Field(default=True, description="Enable statistical poisoning detection")
    poisoning_detection_mode: Literal["observe", "enforce"] = Field(
        default="observe",
        description=(
            "SEC-001 statistical-anomaly (size/tag-count) intake mode. 'observe' "
            "(default) records rolling anomaly stats + emits telemetry but does NOT "
            "quarantine — matching the documented SEC-001 observe-only rollout "
            "(enforce-mode promotion was never signed off). 'enforce' quarantines "
            "anomalous writes. This gates ONLY the statistical size/tag-count "
            "detector; PII redaction, schema validation, write-rate limits, the "
            "trust scorer (own trust_scoring_mode), and canary tamper halts are "
            "unaffected. The per-entry MCP write path accumulates a reference "
            "distribution as it stores, so a single long, well-formed learning can "
            "score as a >3-sigma length outlier against a corpus of short entries "
            "and be silently quarantined — observe-mode prevents that false-positive "
            "from dropping high-value learnings out of recall."
        ),
    )
    poisoning_z_threshold: float = Field(default=3.0, gt=0.0, description="Z-score threshold for anomaly detection")
    anomaly_bypass_source_prefixes: list[str] = Field(
        default_factory=lambda: ["distilled:", "distilled-git:"],
        description=(
            "Source-identifier prefixes (matched against metadata['source']) that "
            "bypass anomaly-based quarantine in prepare_entry_for_store. Intended "
            "for source-grounded automated ingestion paths whose producer pipeline "
            "has already validated record provenance (e.g. trw-distill). The "
            "PRD-SEC-001 anomaly defense remains active for all non-matching writes. "
            "Set to [] to disable the bypass and apply anomaly quarantine to every "
            "write."
        ),
    )
    quarantine_path: str = Field(default="", description="Directory where quarantined entries are written")
    enable_trust_scoring: bool = Field(default=True, description="Enable SEC-001 trust scoring on all ingest paths")
    trust_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="SEC-001 quarantine threshold")
    trust_scoring_mode: Literal["observe", "enforce", "strict"] = Field(
        default="observe",
        description="SEC-001 intake mode: observe logs only, enforce quarantines, strict rejects",
    )
    quarantine_ttl_seconds: int = Field(default=1_209_600, ge=0, description="Retention window for quarantine rows")
    quarantine_db_path: str = Field(default="", description="SQLite DB path for SEC-001 quarantine records")
    enable_recall_filter: bool = Field(default=True, description="Enable SEC-001 recall filtering")
    recall_filter_mode: Literal["strict", "redact", "observe"] = Field(
        default="redact",
        description="SEC-001 recall filter mode",
    )
    canary_injection_rate: int = Field(default=5, ge=3, le=5, description="Number of in-code canaries to seed")
    canary_probe_interval: int = Field(default=25, ge=1, description="Probe canaries every N recalls")
    canary_fail_mode: Literal["halt", "degrade", "log-only"] = Field(
        default="halt",
        description="Fail mode when canary tamper is detected",
    )
    canary_fixtures_path: str = Field(default="", description="Path to shipped SEC-001 canary fixtures")
    provenance_required: bool = Field(default=True, description="Require signed provenance on persisted rows")
    provenance_signing_key_path: str = Field(default="", description="Path to the Ed25519 signing key")
    max_entry_chars: int = Field(default=10_240, gt=0, description="Maximum combined content/detail character count")
    max_memory_writes_per_minute: int = Field(
        default=10, ge=0, description="Per-session write limit enforced over a rolling minute"
    )
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

    # Corruption backup rotation (PRD-CORE-139)
    memory_corrupt_backup_keep: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Number of corruption backup files to retain before oldest-by-filename-timestamp "
            "eviction. Legacy memory.db.corrupt.bak and memory.db.corrupt.bak.1 files count "
            "against this budget but are never selected for deletion."
        ),
    )

    # Cold-tier rebuild on recovery (PRD-CORE-140)
    memory_recovery_rebuild_from_cold: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "memory_recovery_rebuild_from_cold",
            "recovery_rebuild_from_cold",
        ),
        description=(
            "When True AND memory_recovery_policy='strict' AND salvage yields 0 rows from "
            "a non-empty backup, rebuild the DB from the cold YAML tier before raising "
            "CorruptDatabaseUnsalvageableError. Set to False to disable automatic rebuild."
        ),
    )
    memory_recovery_inline_max_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=0,
        description=(
            "Maximum DB size eligible for inline recovery preflight. Larger stores are marked for "
            "degraded-open/background recovery instead of doing heavy recovery work in startup."
        ),
    )

    # Periodic integrity scheduler (PRD-INFRA-063 / B2)
    memory_integrity_check_interval_minutes: int = Field(
        default=0,
        ge=0,
        le=1440,
        validation_alias=AliasChoices(
            "memory_integrity_check_interval_minutes",
            "integrity_check_interval_minutes",
        ),
        description=(
            "Interval in minutes between background PRAGMA quick_check runs on a read-only "
            "connection. 0 disables (default — opt-in). Max 1440 (1 day). Observability-only: "
            "a failed check sets integrity_warning=True and logs db_integrity_regression_detected; "
            "it NEVER triggers auto-recovery."
        ),
    )

    # Multi-writer advisory registry (PRD-INFRA-064 / B3)
    memory_concurrent_writer_warn_threshold: int = Field(
        default=4,
        ge=1,
        le=100,
        validation_alias=AliasChoices(
            "memory_concurrent_writer_warn_threshold",
            "concurrent_writer_warn_threshold",
        ),
        description=(
            "When the count of live writer pids registered in <db_path>.writers/ exceeds this "
            "value, log at WARNING level. Advisory ONLY — the registry NEVER refuses open(). "
            "The 2026-04-12 incident involved 9 concurrent writers; default 4 surfaces the "
            "pattern without being noisy for 1-3 writer workloads."
        ),
    )

    # Snapshot rotation (PRD-INFRA-065 / B4)
    memory_snapshot_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "memory_snapshot_enabled",
            "snapshot_enabled",
        ),
        description=(
            "When True, trw_deliver and the CLI `trw-memory snapshot` subcommand take "
            "VACUUM INTO snapshots under <base_dir>/memory/snapshots/{daily,weekly}/. "
            "Default False (opt-in) to avoid surprising disk usage."
        ),
    )
    memory_snapshot_daily_keep: int = Field(
        default=7,
        ge=1,
        le=365,
        validation_alias=AliasChoices(
            "memory_snapshot_daily_keep",
            "snapshot_daily_keep",
        ),
        description="Number of daily snapshots retained under snapshots/daily/ before oldest-by-filename eviction.",
    )
    memory_snapshot_weekly_keep: int = Field(
        default=4,
        ge=1,
        le=52,
        validation_alias=AliasChoices(
            "memory_snapshot_weekly_keep",
            "snapshot_weekly_keep",
        ),
        description="Number of weekly snapshots retained under snapshots/weekly/ before oldest-by-filename eviction.",
    )

    # Off-box snapshot hash publish (PRD-INFRA-066 / C1)
    memory_snapshot_publish_hash: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "memory_snapshot_publish_hash",
            "snapshot_publish_hash",
        ),
        description=(
            "When True AND sync_enabled=True AND NOT local_only, publish SHA-256 hash of the "
            "latest snapshot (plus size and timestamp metadata — NEVER contents) to the platform "
            "for drift notification on restore. Opt-in. Ignored silently under local_only=True."
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
        storage_root = Path(self.storage_path)
        if storage_root.name == "memory" and storage_root.parent.name == ".trw":
            security_root = storage_root.parent / "security"
        else:
            security_root = storage_root.parent / ".trw" / "security"
        if not self.audit_log_path:
            self.audit_log_path = str(security_root / "audit.jsonl")
        if not self.quarantine_path:
            self.quarantine_path = str(security_root / "quarantine")
        if not self.quarantine_db_path:
            self.quarantine_db_path = str(security_root / "quarantine.db")
        if not self.rate_limit_state_path:
            self.rate_limit_state_path = str(security_root / "rate_limits.yaml")
        if not self.provenance_signing_key_path:
            self.provenance_signing_key_path = str(security_root / "ed25519_signing_key.bin")
        if not self.canary_fixtures_path:
            self.canary_fixtures_path = "package:canary"
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
