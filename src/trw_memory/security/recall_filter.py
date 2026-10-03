"""Recall-time integrity filter: hash-pin drift (PRD-SEC-001 FR-004).

An entry whose provenance metadata pins a content hash, and whose content no longer matches it, was changed
outside the signed write path; ``strict`` mode keeps it out of recall, ``observe`` mode returns it and reports it.

The recall-time injection-pattern redaction that used to live here is gone (UF-MEM-03, 2026-10-01): over 15 days
of observe data it caught nothing and only rewrote legitimate learnings that quote a pattern, and every entry it
could flag had already passed the write gate, which refuses known injection shapes at store time.
"""

from __future__ import annotations

import time
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.models.memory import MemoryEntry
from trw_memory.security.provenance import entry_content_hash

__all__ = ["RecallDecision", "RecallFilterResult", "entry_text_fields", "filter_recall_window"]

_LOG = structlog.get_logger(__name__)
_LATENCY_BUDGET_MS = 20.0

Action = Literal["allow", "block"]
Mode = Literal["strict", "observe"]


class RecallDecision(BaseModel):
    """Decision for a single recalled entry."""

    model_config = ConfigDict(strict=True)

    action: Action
    reasons: list[str] = Field(default_factory=list)


class RecallFilterResult(BaseModel):
    """Outcome of filtering a recall window.

    In observe mode ``accepted`` equals the full input; ``would_reject``
    collects entries that strict mode would have blocked, keyed in ``reasons`` by entry id.
    """

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    accepted: list[MemoryEntry]
    would_reject: list[MemoryEntry] = Field(default_factory=list)
    reasons: dict[str, list[str]] = Field(default_factory=dict)
    actions: dict[str, Action] = Field(default_factory=dict)


def _inspect(entry: MemoryEntry) -> list[str]:
    """Return a list of recall-filter reasons; empty list = pass."""
    # This MUST hash the provenance basis (content+detail, bare), not a wider scan surface: hashing more than was
    # signed made every signed row report drift and recall went empty for every provenance-signed entry.
    pinned = entry.metadata.get("provenance_content_hash") or entry.metadata.get("content_hash")
    if pinned and entry_content_hash(entry.content, entry.detail) != pinned:
        return ["hash_pin_drift"]
    return []


def _decide(entry: MemoryEntry, *, mode: Mode) -> RecallDecision:
    reasons = _inspect(entry)
    if not reasons or mode == "observe":
        return RecallDecision(action="allow", reasons=reasons)
    return RecallDecision(action="block", reasons=reasons)


def filter_recall_window(learnings: list[MemoryEntry], *, mode: Mode = "strict") -> RecallFilterResult:
    """Filter a recall window for hash-pin drift.

    - ``strict``: drop an entry whose content no longer matches its pinned hash
    - ``observe``: return every entry and record the ones strict mode would drop
    """
    t0 = time.monotonic_ns()
    would_reject: list[MemoryEntry] = []
    reasons: dict[str, list[str]] = {}
    actions: dict[str, Action] = {}
    accepted_strict: list[MemoryEntry] = []

    for entry in learnings:
        decision = _decide(entry, mode=mode)
        actions[entry.id] = decision.action
        if decision.reasons:
            would_reject.append(entry)
            reasons[entry.id] = decision.reasons
        if decision.action == "allow":
            accepted_strict.append(entry)

    accepted = list(learnings) if mode == "observe" else accepted_strict
    elapsed_ms = (time.monotonic_ns() - t0) / 1_000_000.0
    _LOG.info(
        "recall_filter.observe" if mode == "observe" else "recall_filter.enforce",
        window_size=len(learnings),
        would_reject_count=len(would_reject),
        latency_ms=round(elapsed_ms, 3),
        mode=mode,
    )
    if elapsed_ms > _LATENCY_BUDGET_MS and len(learnings) <= 25:
        _LOG.warning(
            "recall_filter.latency_budget_exceeded",
            latency_ms=round(elapsed_ms, 3),
            budget_ms=_LATENCY_BUDGET_MS,
            window_size=len(learnings),
        )
    return RecallFilterResult(accepted=accepted, would_reject=would_reject, reasons=reasons, actions=actions)


def entry_text_fields(entry: MemoryEntry) -> dict[str, object]:
    """The entry's text-bearing fields, JSON-safe, for a recall result row.

    Both recall surfaces (the client's ``apply_recall_security`` and the tool's ``_apply_sec001_recall_policy``)
    copy these onto the row they return, so a result carries the entry's ``nudge_line``, ``evidence`` and
    ``assertions`` and not just the subset its search index projected. The copy outlived the recall-time
    redaction it was written for (UF-MEM-03): dropping it removed those fields from recall output.
    """
    return {
        "content": entry.content,
        "detail": entry.detail,
        "nudge_line": entry.nudge_line,
        "tags": list(entry.tags),
        "evidence": list(entry.evidence),
        "assertions": [assertion.model_dump(mode="json") for assertion in entry.assertions],
    }
