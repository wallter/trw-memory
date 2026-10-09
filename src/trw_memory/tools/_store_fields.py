"""``LearningFields`` -- belongs to the ``store.py`` facade, extracted to keep it under the 350-eLOC gate.

Re-exported from ``trw_memory.tools.store`` for back-compat; existing
``from trw_memory.tools.store import LearningFields`` imports are unaffected.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from trw_memory.models.memory import Anchor, Confidence, EvidenceLevel, MemoryType, ProtectionTier

__all__ = ["LearningFields"]


class LearningFields(BaseModel):
    """The typed learning fields ``memory_store`` forwards to :func:`memory_store_impl` (PRD-CORE-294 FR07a).

    One optional object instead of ten parameters keeps the tool definition small;
    an unknown key is refused rather than silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    type: MemoryType | None = None
    confidence: Confidence | None = None
    evidence_level: EvidenceLevel | None = None  # PRD-CORE-312-FR01
    task_type: str | None = None
    domain: list[str] | None = None
    phase_origin: str | None = None
    phase_affinity: list[str] | None = None
    team_origin: str | None = None
    protection_tier: ProtectionTier | None = None
    anchors: list[Anchor] | None = None
    nudge_line: str | None = None
    # PRD-CORE-298 FR01: provenance the daemon store forwards for trw-mcp writes.
    source: Literal["human", "agent", "tool", "consolidated", "distill", "team_sync", "company_sync"] | None = None
    client_profile: str | None = None
    model_id: str | None = None
    anchor_validity: float | None = None
