"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

__all__ = ["_LifecycleConfigMixin"]

#: Inclusive bounds for every value of ``protection_tier_prune_discount``.
#: 0.0 means "never nominated"; 4.0 means "four times easier to nominate than
#: normal". An operator-supplied ``.trw/config.yaml`` cannot escape this band.
PROTECTION_DISCOUNT_MIN = 0.0
PROTECTION_DISCOUNT_MAX = 4.0

#: Number of distinct artifact kinds that can substantiate a ``verified`` claim
#: — a non-whitespace evidence string, a non-empty assertions list, a non-empty
#: anchors list. It is the hard ceiling on
#: ``min_evidence_items_for_verified``: demanding 4 of 3 is not a stricter rule,
#: it is an unsatisfiable one, and an unsatisfiable gate that reads like a
#: configured threshold is precisely the defect class PRD-CORE-244 exists to
#: remove. Kept in lockstep with ``security.poisoning._count_substantiation``.
MAX_SUBSTANTIATION_ITEMS = 3


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

    # Automatic-removal protection (PRD-CORE-244 FR10)
    protection_tier_prune_discount: dict[str, float] = Field(
        default_factory=lambda: {"critical": 0.25, "high": 0.5, "normal": 1.0, "low": 1.5},
        description=(
            "Multiplier applied to the utility threshold an entry must fall BELOW before an "
            "automatic prune, tier demotion or purge may nominate it. A 'critical' entry at 0.25 "
            "must therefore be four times less useful than a 'normal' one before it is a "
            "candidate. The 'protected' and 'permanent' tiers are not listed because they are "
            "exempt outright, not discounted (PRD-CORE-244 FR10). Mirrors the trw-mcp "
            "TRWConfig field of the same name."
        ),
    )

    @field_validator("protection_tier_prune_discount")
    @classmethod
    def _bound_protection_discounts(cls, value: dict[str, float]) -> dict[str, float]:
        for tier, discount in value.items():
            if not PROTECTION_DISCOUNT_MIN <= discount <= PROTECTION_DISCOUNT_MAX:
                raise ValueError(
                    f"protection_tier_prune_discount[{tier!r}]={discount} is outside "
                    f"[{PROTECTION_DISCOUNT_MIN}, {PROTECTION_DISCOUNT_MAX}]"
                )
        return value

    # Write-time substantiation (PRD-CORE-244 FR02)
    min_evidence_items_for_verified: int = Field(
        default=1,
        ge=1,
        le=MAX_SUBSTANTIATION_ITEMS,
        description=(
            "How many substantiating artifacts a confidence='verified' write must carry — "
            "counting a non-whitespace evidence string, a non-empty assertions list and a "
            "non-empty anchors list as one each. It bounds HOW MUCH substantiation is demanded; "
            "it cannot express 'demand none', so it is a tunable and not an off switch. "
            "The ceiling is the number of artifact KINDS that exist (3): a higher value could "
            "never be satisfied by any entry, so it would silently lock out every verified "
            "write instead of tightening the rule."
        ),
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
    feedback_decay_min_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Floor on the PRD-CORE-132 feedback-decay factor (PRD-CORE-244 FR11 residual). "
            "With helpful_count at 0 corpus-wide the term is a pure recall-frequency penalty "
            "of 0.95**recall_count with no lower bound, which buries whatever the retriever "
            "keeps finding. 0.0 restores the unbounded pre-floor behaviour."
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
