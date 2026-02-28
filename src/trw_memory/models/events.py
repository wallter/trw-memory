"""Audit event models for memory operations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MemoryEventType(str, Enum):
    """Types of auditable memory operations."""

    STORE = "store"
    RECALL = "recall"
    UPDATE = "update"
    DELETE = "delete"
    CONSOLIDATE = "consolidate"
    MIGRATE = "migrate"
    TIER_PROMOTE = "tier_promote"
    TIER_DEMOTE = "tier_demote"
    TIER_PURGE = "tier_purge"


class MemoryEvent(BaseModel):
    """Structured audit event for memory operations."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: MemoryEventType
    memory_id: str = ""
    namespace: str = "default"
    actor: str = ""
    detail: dict[str, str] = Field(default_factory=dict)
