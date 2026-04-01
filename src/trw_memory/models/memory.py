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

_logger = logging.getLogger(__name__)

# Valid source values for MemoryEntry provenance tracking.
_VALID_SOURCES = frozenset({"human", "agent", "tool", "consolidated"})


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    OBSOLETE = "obsolete"
    ARCHIVED = "archived"


class AssertionType(str, Enum):
    """Type of executable assertion attached to a memory entry."""

    GREP_PRESENT = "grep_present"
    GREP_ABSENT = "grep_absent"
    GLOB_EXISTS = "glob_exists"
    GLOB_ABSENT = "glob_absent"


class Assertion(BaseModel):
    """Machine-verifiable assertion attached to a memory entry.

    Executes grep/glob patterns against the codebase to verify
    knowledge is still true. Read-only, safe — no shell commands.
    """

    model_config = ConfigDict(use_enum_values=True)

    type: AssertionType
    pattern: str = Field(default="", description="Regex pattern for grep types; ignored for glob types")
    target: str = Field(min_length=1, description="Glob pattern for file matching, relative to project root")
    last_result: bool | None = Field(default=None, description="Result of last verification run")
    last_verified_at: datetime | None = None
    last_evidence: str = Field(default="", description="Human-readable verification evidence")
    first_failed_at: datetime | None = Field(
        default=None, description="When this assertion first started failing consecutively"
    )

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


class MemoryEntry(BaseModel):
    """Individual memory entry stored in the memory system.

    Fields are intentionally agent-agnostic — any MCP client can
    produce and consume these entries.
    """

    model_config = ConfigDict(strict=True, use_enum_values=True)

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

    # Scoring
    q_value: float = Field(ge=0.0, le=1.0, default=0.5)
    q_observations: int = Field(ge=0, default=0)

    # Provenance
    source: Literal["human", "agent", "tool", "consolidated"] = Field(
        default="agent", description="Origin: 'human', 'agent', 'tool', 'consolidated'"
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

    # Sync fields (PRD-CORE-047)
    vector_clock: dict[str, int] = Field(default_factory=dict, description="node_id -> counter for conflict resolution")
    remote_id: str | None = None
    published_to_platform: bool = False
    pending_delete: bool = False

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

    # Executable assertions (PRD-CORE-086)
    assertions: list[Assertion] = Field(default_factory=list, description="Machine-verifiable assertions")

    def __repr__(self) -> str:
        """Concise repr showing key identifying fields."""
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return (
            f"MemoryEntry(id={self.id!r}, content={preview!r}, "
            f"tags={self.tags!r}, importance={self.importance:.2f})"
        )

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
            "status": str(self.status),
            "recurrence": self.recurrence,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "q_value": self.q_value,
            "q_observations": self.q_observations,
            "source": self.source,
            "source_identity": self.source_identity,
            "client_profile": self.client_profile,
            "model_id": self.model_id,
            "merged_from": list(self.merged_from),
            "consolidated_from": list(self.consolidated_from),
            "consolidated_into": self.consolidated_into,
            "metadata": dict(self.metadata),
            "vector_clock": dict(self.vector_clock),
            "remote_id": self.remote_id,
            "published_to_platform": self.published_to_platform,
            "pending_delete": self.pending_delete,
            "cross_validated": self.cross_validated,
            "outcome_history": list(self.outcome_history),
            "assertions": [a.model_dump() for a in self.assertions] if self.assertions else [],
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
