"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

__all__ = ["_SecurityConfigMixin"]


class _SecurityConfigMixin(BaseModel):
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
