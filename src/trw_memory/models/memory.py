"""Core memory data models.

MemoryEntry is the universal memory record — agent-agnostic, framework-agnostic.
Designed as the standalone replacement for TRW's LearningEntry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

__all__ = [
    "Anchor",
    "Assertion",
    "AssertionResult",
    "AssertionType",
    "Confidence",
    "MemoryEntry",
    "MemoryIndex",
    "MemoryStatus",
    "MemoryType",
    "ProtectionTier",
]

_logger = logging.getLogger(__name__)

# Valid source values for MemoryEntry provenance tracking.
_VALID_SOURCES = frozenset({"human", "agent", "tool", "consolidated", "team_sync"})


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    OBSOLETE = "obsolete"
    OBSOLETE_POISONED = "obsolete_poisoned"
    ARCHIVED = "archived"


class AssertionType(str, Enum):
    """Type of executable assertion attached to a memory entry."""

    GREP_PRESENT = "grep_present"
    GREP_ABSENT = "grep_absent"
    GLOB_EXISTS = "glob_exists"
    GLOB_ABSENT = "glob_absent"


# PRD-CORE-110: Memory entry type classifications
class MemoryType(str, Enum):
    """Type classification for memory entries."""

    INCIDENT = "incident"
    PATTERN = "pattern"
    CONVENTION = "convention"
    HYPOTHESIS = "hypothesis"
    WORKAROUND = "workaround"


# PRD-CORE-110: Confidence levels for memory validation
class Confidence(str, Enum):
    """Validation confidence level for memory entries."""

    UNVERIFIED = "unverified"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


# PRD-CORE-110: Protection tiers for memory entries
class ProtectionTier(str, Enum):
    """Protection level for memory entries."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    PROTECTED = "protected"
    PERMANENT = "permanent"


class Assertion(BaseModel):
    model_config = ConfigDict(strict=True, use_enum_values=True)

    type: AssertionType
    pattern: str = Field(default="", description="Regex pattern for grep types; ignored for glob types")
    target: str = Field(min_length=1, description="Glob pattern for file matching, relative to project root")
    last_result: bool | None = Field(default=None, description="Result of last verification run")
    last_verified_at: datetime | None = None
    last_evidence: str = Field(default="", description="Human-readable verification evidence")
    first_failed_at: datetime | None = Field(
        default=None, description="When this assertion first started failing consecutively"
    )

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> AssertionType:
        """Accept persisted string enum values before strict validation runs."""
        if isinstance(v, AssertionType):
            return v
        if isinstance(v, str):
            try:
                return AssertionType(v)
            except ValueError as err:
                valid = ", ".join(assertion_type.value for assertion_type in AssertionType)
                raise ValueError(f"type must be one of {valid}") from err
        raise ValueError(f"type must be a string or AssertionType enum, got {type(v).__name__}")

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str, info: ValidationInfo) -> str:
        values = info.data
        assertion_type = values.get("type", "")
        # grep types require non-empty pattern
        # Note: use_enum_values=True normalizes enum members to their string
        # values, so we only need to compare against the string forms.
        if assertion_type in ("grep_present", "grep_absent") and (not v or len(v.strip()) == 0):
            raise ValueError("grep assertion types require a non-empty pattern")
        # pattern length cap for security (ReDoS mitigation)
        if len(v) > 500:
            raise ValueError(f"pattern exceeds 500 character limit ({len(v)} chars)")
        return v

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError("absolute paths not allowed in assertion targets")
        if ".." in v.split("/"):
            raise ValueError("path traversal (..) not allowed in assertion targets")
        return v


class AssertionResult(BaseModel):
    """Result of running a single assertion against the codebase."""

    model_config = ConfigDict(use_enum_values=True)

    type: AssertionType
    pattern: str = ""
    target: str = ""
    passed: bool | None = Field(default=None, description="True=passed, False=failed, None=could not verify")
    evidence: str = Field(default="", description="Human-readable evidence")


class Anchor(BaseModel):
    """Code symbol anchor for validation (PRD-CORE-111).

    Anchors capture code symbols (functions, classes, methods) that
    provide factual grounding for learnings. When the referenced
    code changes or disappears, the learning becomes stale.
    """

    model_config = ConfigDict(use_enum_values=True, strict=True)

    file: str = Field(
        min_length=1,
        description="Relative path to the source file containing the symbol",
    )
    symbol_name: str = Field(
        min_length=1,
        description="Name of the symbol (function, class, method, etc.)",
    )

    @field_validator("symbol_name")
    @classmethod
    def _validate_symbol_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol_name must not be empty or whitespace")
        return v

    symbol_type: Literal["function", "method", "class", "const", "type", "impl"] = Field(
        default="function",
        description="Type of symbol being anchored",
    )
    signature: str = Field(default="", description="Full signature line (truncated to 200 chars)")
    line_range: tuple[int, int] | None = Field(
        default=None,
        description="Line range (start, end) where the symbol is defined",
    )

    @field_validator("line_range", mode="before")
    @classmethod
    def _coerce_line_range(cls, v: object) -> tuple[int, int] | None:
        """Coerce list to tuple for JSON/SQLite round-trip compatibility."""
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        return v  # type: ignore[return-value]

    @field_validator("file")
    @classmethod
    def _validate_file(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError("Anchor.file must be a relative path, not absolute")
        if ".." in v.split("/"):
            raise ValueError("path traversal (..) not allowed in anchor file paths")
        return v

    @field_validator("signature", mode="before")
    @classmethod
    def _truncate_signature(cls, v: str) -> str:
        if len(v) > 200:
            return v[:200]
        return v


class MemoryEntry(BaseModel):
    """Individual memory entry stored in the memory system.

    Fields are intentionally agent-agnostic — any MCP client can
    produce and consume these entries.
    """

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    id: str = Field(min_length=1)
    content: str = Field(description="Core knowledge statement (was 'summary' in LearningEntry)")
    detail: str = Field(default="", description="Extended explanation")
    tags: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0, default=0.5, description="Importance score (was 'impact')")
    status: MemoryStatus = MemoryStatus.ACTIVE
    recurrence: int = Field(ge=0, default=1)
    namespace: str = Field(default="default", description="Isolation scope: project, team, org, global")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime | None = None
    access_count: int = Field(ge=0, default=0)
    session_count: int = Field(ge=0, default=0)

    # Scoring
    q_value: float = Field(ge=0.0, le=1.0, default=0.5)
    q_observations: int = Field(ge=0, default=0)

    # Provenance
    source: Literal["human", "agent", "tool", "consolidated", "team_sync"] = Field(
        default="agent", description="Origin: 'human', 'agent', 'tool', 'consolidated', 'team_sync'"
    )
    source_identity: str = Field(default="", description="Name of source agent/user")
    client_profile: str = Field(
        default="",
        description="IDE/client that created this entry (e.g., 'claude-code', 'opencode', 'cursor')",
    )
    model_id: str = Field(
        default="",
        description="AI model that created this entry (e.g., 'claude-opus-4-6', 'claude-sonnet-4-6')",
    )

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_unknown_source(cls, v: object) -> str:
        """Coerce unknown source values to 'agent' for backward compatibility.

        Existing data may contain arbitrary source strings from before the
        Literal constraint was introduced. Rather than breaking on load, we
        silently fall back to 'agent' and log a warning.
        """
        if not isinstance(v, str) or v not in _VALID_SOURCES:
            _logger.warning("unknown source value %r coerced to 'agent'", v)
            return "agent"
        return v

    @field_validator("nudge_line", mode="before")
    @classmethod
    def _truncate_nudge_line(cls, v: str) -> str:
        """Truncate nudge_line to max 80 chars, preferring word boundaries within [60,80)."""
        if len(v) <= 80:
            return v
        # Try to truncate at a word boundary within [60, 80)
        for i in range(60, 80):
            if v[i] == " ":
                return v[:i] + "\u2026"
        # No space found: hard-cut at 80 chars without ellipsis
        return v[:80]

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> MemoryType:
        """Coerce string values to MemoryType enum."""
        if isinstance(v, str):
            if not v:  # Empty string -> default (backward compat)
                return MemoryType.PATTERN
            try:
                return MemoryType(v)
            except ValueError as err:
                raise ValueError(f"type must be one of {', '.join([t.value for t in MemoryType])}") from err
        if isinstance(v, MemoryType):
            return v
        raise ValueError(f"type must be a string or MemoryType enum, got {type(v).__name__}")

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> Confidence:
        """Coerce string values to Confidence enum."""
        if isinstance(v, str):
            if not v:  # Empty string -> default (backward compat)
                return Confidence.UNVERIFIED
            try:
                return Confidence(v)
            except ValueError as err:
                raise ValueError(f"confidence must be one of {', '.join([c.value for c in Confidence])}") from err
        if isinstance(v, Confidence):
            return v
        raise ValueError(f"confidence must be a string or Confidence enum, got {type(v).__name__}")

    @field_validator("protection_tier", mode="before")
    @classmethod
    def _coerce_protection_tier(cls, v: object) -> ProtectionTier:
        """Coerce string values to ProtectionTier enum."""
        if isinstance(v, str):
            if not v:  # Empty string -> default (backward compat)
                return ProtectionTier.NORMAL
            try:
                return ProtectionTier(v)
            except ValueError as err:
                raise ValueError(
                    f"protection_tier must be one of {', '.join([p.value for p in ProtectionTier])}"
                ) from err
        if isinstance(v, ProtectionTier):
            return v
        raise ValueError(f"protection_tier must be a string or ProtectionTier enum, got {type(v).__name__}")

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: list[str]) -> list[str]:
        """Validate domain list length."""
        if len(v) > 20:
            raise ValueError("domain may have at most 20 entries")
        return v

    @field_validator("phase_affinity")
    @classmethod
    def _validate_phase_affinity(cls, v: list[str]) -> list[str]:
        """Validate phase_affinity list length."""
        if len(v) > 6:
            raise ValueError("phase_affinity may have at most 6 entries")
        return v

    # Sync fields (PRD-CORE-047)
    vector_clock: dict[str, int] = Field(default_factory=dict, description="node_id -> counter for conflict resolution")
    remote_id: str | None = None
    published_to_platform: bool = False
    pending_delete: bool = False

    # Sync pipeline fields (PRD-INFRA-051)
    sync_hash: str = ""
    sync_seq: int = 0
    last_synced_at: datetime | None = None

    # Graph fields (PRD-CORE-048)
    cross_validated: bool = False
    outcome_history: list[str] = Field(
        default_factory=list, description="Structured event log (boost, decay, promote records)"
    )

    # Merge/consolidation tracking
    merged_from: list[str] = Field(default_factory=list)
    consolidated_from: list[str] = Field(default_factory=list)
    consolidated_into: str | None = None

    # Arbitrary metadata
    metadata: dict[str, str] = Field(default_factory=dict)

    # PRD-CORE-110: Typed entries with metadata
    type: MemoryType = Field(default=MemoryType.PATTERN, description="Type classification")
    nudge_line: str = Field(default="", description="Nudge text for summary (max 80 chars)")
    expires: str = Field(default="", description="Expiration date/condition")
    confidence: Confidence = Field(default=Confidence.UNVERIFIED, description="Validation confidence")
    task_type: str = Field(default="", description="Task type identifier")
    domain: list[str] = Field(default_factory=list, description="Domain tags (max 20)")
    phase_origin: str = Field(default="", description="Origin phase")
    phase_affinity: list[str] = Field(default_factory=list, description="Phase affinities (max 6)")
    team_origin: str = Field(default="", description="Team identifier")
    protection_tier: ProtectionTier = Field(default=ProtectionTier.NORMAL, description="Protection level")

    # PRD-CORE-108: Outcome attribution fields
    sessions_surfaced: int = Field(ge=0, default=0, description="Sessions this entry was surfaced in")
    avg_rework_delta: float | None = Field(default=None, description="Rolling average rework impact delta")
    outcome_correlation: str = Field(default="", description="Causal outcome attribution category")

    # Executable assertions (PRD-CORE-086)
    assertions: list[Assertion] = Field(default_factory=list, description="Machine-verifiable assertions")

    # PRD-CORE-111: Code-grounded anchors
    anchors: list[Anchor] = Field(default_factory=list, description="Code symbol anchors for validation", max_length=3)
    anchor_validity: float = Field(ge=0.0, le=1.0, default=1.0, description="Computed validity score (0.0-1.0)")

    # PRD-CORE-132: Feedback lifecycle counters
    recall_count: int = Field(ge=0, default=0, description="Number of times this entry was returned by recall")
    helpful_count: int = Field(ge=0, default=0, description="Number of times marked helpful by the user")
    unhelpful_count: int = Field(ge=0, default=0, description="Number of times marked unhelpful by the user")

    def __repr__(self) -> str:
        """Concise repr showing key identifying fields."""
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"MemoryEntry(id={self.id!r}, content={preview!r}, tags={self.tags!r}, importance={self.importance:.2f})"

    def to_dict(self, *, fields: set[str] | None = None) -> dict[str, object]:
        """Serialize this entry to a plain dict.

        Args:
            fields: If provided, include only these field names.
                When None, all fields are included.

        Returns:
            Dict suitable for YAML/JSON serialization.
        """
        full: dict[str, object] = {
            "id": self.id,
            "content": self.content,
            "detail": self.detail,
            "tags": list(self.tags),
            "evidence": list(self.evidence),
            "importance": self.importance,
            "status": self.status.value if hasattr(self.status, "value") else self.status,
            "recurrence": self.recurrence,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "session_count": self.session_count,
            "q_value": self.q_value,
            "q_observations": self.q_observations,
            "source": self.source,
            "source_identity": self.source_identity,
            "client_profile": self.client_profile,
            "model_id": self.model_id,
            "merged_from": list(self.merged_from),
            "consolidated_from": list(self.consolidated_from),
            "consolidated_into": self.consolidated_into,
            "type": self.type.value if hasattr(self.type, "value") else self.type,
            "nudge_line": self.nudge_line,
            "expires": self.expires,
            "confidence": self.confidence.value if hasattr(self.confidence, "value") else self.confidence,
            "task_type": self.task_type,
            "domain": list(self.domain),
            "phase_origin": self.phase_origin,
            "phase_affinity": list(self.phase_affinity),
            "team_origin": self.team_origin,
            "protection_tier": self.protection_tier.value
            if hasattr(self.protection_tier, "value")
            else self.protection_tier,
            "metadata": dict(self.metadata),
            "vector_clock": dict(self.vector_clock),
            "remote_id": self.remote_id,
            "published_to_platform": self.published_to_platform,
            "pending_delete": self.pending_delete,
            "sync_hash": self.sync_hash,
            "sync_seq": self.sync_seq,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "cross_validated": self.cross_validated,
            "outcome_history": list(self.outcome_history),
            "assertions": [a.model_dump() for a in self.assertions] if self.assertions else [],
            "anchors": [a.model_dump() for a in self.anchors],
            "anchor_validity": self.anchor_validity,
            "sessions_surfaced": self.sessions_surfaced,
            "avg_rework_delta": self.avg_rework_delta,
            "outcome_correlation": self.outcome_correlation,
            "recall_count": self.recall_count,
            "helpful_count": self.helpful_count,
            "unhelpful_count": self.unhelpful_count,
        }
        if fields is not None:
            return {k: v for k, v in full.items() if k in fields}
        return full


class MemoryIndex(BaseModel):
    """Index tracking all memory entries."""

    model_config = ConfigDict(strict=True)

    entries: list[MemoryEntry] = Field(default_factory=list)
    total_count: int = 0

    @model_validator(mode="after")
    def _sync_total_count(self) -> MemoryIndex:
        self.total_count = len(self.entries)
        return self
