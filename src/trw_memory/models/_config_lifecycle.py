"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

__all__ = ["_LifecycleConfigMixin"]


class _LifecycleConfigMixin(BaseModel):
    # Dedup
    dedup_enabled: bool = Field(default=True, description="Enable semantic deduplication")
    dedup_skip_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Similarity threshold for skipping duplicate entries"
    )
    dedup_merge_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Similarity threshold for merging similar entries"
    )
    dedup_lexical_fallback: bool = Field(
        default=True,
        description=(
            "When embeddings are unavailable, fall back to exact normalized-text "
            "dedup so identical entries are still caught instead of silently "
            "accumulating. Set False to restore legacy no-op-without-embeddings."
        ),
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
    lifecycle_use_fsrs: bool = Field(
        default=False,
        validation_alias=AliasChoices("lifecycle_use_fsrs", "memory_lifecycle_use_fsrs"),
        description=(
            "When True, entry_utility() uses FSRS-4.5 power-law retention "
            "(R(t,S)=(1+FACTOR*t/S)^DECAY) instead of the Ebbinghaus exponential. "
            "FSRS models spaced-repetition dynamics more accurately for entries "
            "that have been recalled multiple times."
        ),
    )
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
