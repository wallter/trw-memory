"""Tuning surface for the single entry-utility implementation (PRD-CORE-244 FR11).

Until FR11 there were two independent entry-utility functions reading different
field sets: ``trw_mcp.scoring._decay._entry_utility`` (the LIVE one, ranking
every ``trw_recall``) and ``trw_memory.lifecycle.scoring.entry_utility`` (used
only by trw-memory's own recall). The live one never read ``helpful_count``,
``unhelpful_count`` or ``recall_count`` — the very counters ``trw_learn``'s
docstring credits with feeding decay — and the test that appeared to prove the
wiring imported the function the live path does not call.

Collapsing them needs one thing the two did not share: a knob bundle. trw-mcp
tunes through ``TRWConfig`` (``learning_decay_half_life_days``,
``q_cold_start_threshold``, ...) and trw-memory through ``MemoryConfig``
(``decay_half_life_days``, ...), and ``MemoryConfig`` is a ``BaseSettings`` that
reads ``.trw/config.yaml`` — far too expensive to construct inside a per-entry
scan loop. :class:`UtilityParams` is the narrow, plain-model seam both bind to
once per ranking pass.

Every value here was previously a literal in one of the two implementations.
Naming them is not a retune: the defaults reproduce the pre-FR11 numbers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DEFAULT_TYPE_HALF_LIFE_DAYS", "UtilityParams"]

#: Per-type decay half-lives, previously ``trw_mcp.scoring._decay._TYPE_HALF_LIFE``
#: (PRD-CORE-102, PRD-CORE-110, PRD-CORE-116). A type absent from this table
#: falls back to ``UtilityParams.half_life_days``.
DEFAULT_TYPE_HALF_LIFE_DAYS: dict[str, float] = {
    "incident": 90.0,  # Slow decay until the fix is confirmed
    "pattern": 180.0,  # Very slow — validated patterns are durable
    "convention": 9999.0,  # No auto-decay — stable until human override
    "hypothesis": 7.0,  # Fast — validate or die
    "workaround": 14.0,  # Fast — scheduled expiry, usually paired with expires
}


class UtilityParams(BaseModel):
    """Frozen knob bundle for :func:`trw_memory.lifecycle.scoring.entry_utility`."""

    model_config = ConfigDict(frozen=True)

    half_life_days: float = Field(
        default=14.0,
        gt=0.0,
        description="Fallback decay half-life for a type absent from type_half_life_days.",
    )
    use_exponent: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Sub-linear recurrence exponent modulating the decay rate.",
    )
    cold_start_threshold: int = Field(
        default=3,
        ge=1,
        description="Q-observations required before q_value is trusted over base importance.",
    )
    access_count_boost_cap: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Ceiling on the sub-linear access-frequency boost.",
    )
    source_human_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Flat utility boost for a human-sourced entry.",
    )
    type_half_life_days: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_TYPE_HALF_LIFE_DAYS),
        description="Per-learning-type decay half-life in days.",
    )
    no_decay_half_life_days: float = Field(
        default=9999.0,
        gt=0.0,
        description=(
            "Effectively-infinite half-life applied to an UNVERIFIED incident, so a postmortem "
            "is not decayed away before anyone confirms the fix."
        ),
    )
    expired_utility_floor: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description=(
            "Utility returned for an entry whose author-set expires date has passed "
            "(PRD-CORE-110). Day-exclusive: an entry expiring today still scores normally."
        ),
    )
    feedback_decay_min_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Floor on the feedback-decay factor. helpful_count is 0 on 100% of the corpus, so "
            "the term degenerates to 0.95**recall_count — a pure recall-FREQUENCY penalty, "
            "unbounded below, on a counter that PRD-QUAL-032/D1 established is not evidence of "
            "use. 0.0 restores the pre-floor behaviour."
        ),
    )
    use_fsrs: bool = Field(
        default=False,
        description="Use FSRS-4.5 power-law retention instead of the Ebbinghaus exponential.",
    )

    def half_life_for(self, entry_type: str, confidence: str) -> float:
        """Resolve the decay half-life for one entry.

        An unverified incident gets ``no_decay_half_life_days`` regardless of the
        type table: it records a problem whose resolution has not been confirmed,
        and decaying it away is how a postmortem gets lost.
        """
        if entry_type == "incident" and confidence == "unverified":
            return self.no_decay_half_life_days
        return self.type_half_life_days.get(entry_type, self.half_life_days)
