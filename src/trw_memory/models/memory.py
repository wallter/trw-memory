"""Core memory data models.

MemoryEntry is the universal memory record — agent-agnostic, framework-agnostic.
Designed as the standalone replacement for TRW's LearningEntry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    OBSOLETE = "obsolete"
    ARCHIVED = "archived"


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
    source: str = Field(default="agent", description="Origin: 'human', 'agent', 'tool', 'consolidated'")
    source_identity: str = Field(default="", description="Name of source agent/user")

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


class MemoryIndex(BaseModel):
    """Index tracking all memory entries."""

    model_config = ConfigDict(strict=True)

    entries: list[MemoryEntry] = Field(default_factory=list)
    total_count: int = 0

    @model_validator(mode="after")
    def _sync_total_count(self) -> MemoryIndex:
        self.total_count = len(self.entries)
        return self
