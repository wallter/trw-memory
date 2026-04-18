"""FR-002 — Recall-time pattern filter.

Sprint 96 W1-E scaffolding in ``observe_mode=True``. Accepts a window of
``MemoryEntry`` (the trw-memory equivalent of the spec's ``LearnIn``)
and returns both the passthrough ``accepted`` list and a ``would_reject``
list for telemetry. In observe mode, ``accepted`` always contains the
full original input.

Target: p95 latency <=20ms for a 25-learning window (PRD-SEC-001 NFR-002).
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.models.memory import MemoryEntry
from trw_memory.security.poisoning import _INJECTION_PATTERNS

__all__ = ["RecallFilterResult", "filter_recall_window"]

_LOG = structlog.get_logger(__name__)
_LATENCY_BUDGET_MS = 20.0


class RecallFilterResult(BaseModel):
    """Outcome of filtering a recall window.

    In observe mode ``accepted`` equals the full input; ``would_reject``
    collects entries that the enforcing filter would have blocked,
    keyed in ``reasons`` by entry id.
    """

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    accepted: list[MemoryEntry]
    would_reject: list[MemoryEntry] = Field(default_factory=list)
    reasons: dict[str, list[str]] = Field(default_factory=dict)


def _inspect(entry: MemoryEntry) -> list[str]:
    """Return a list of rejection reasons; empty list = pass."""
    reasons: list[str] = []
    combined = f"{entry.content}{entry.detail}"
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(combined):
            reasons.append(f"injection_pattern:{pattern.pattern}")
            break

    # Hash-pin drift: if metadata carries ``content_hash`` (provenance
    # record), recompute and compare.
    pinned = entry.metadata.get("content_hash")
    if pinned:
        import hashlib

        current = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if current != pinned:
            reasons.append("hash_pin_drift")
    return reasons


def filter_recall_window(
    learnings: list[MemoryEntry],
    *,
    observe_mode: bool = True,
) -> RecallFilterResult:
    """Filter a recall window for injection patterns and hash drift.

    In ``observe_mode=True`` the ``accepted`` list is always the full
    original ``learnings`` input; ``would_reject`` is a diagnostic
    view of what the enforcing filter would have removed.
    """
    t0 = time.monotonic_ns()
    would_reject: list[MemoryEntry] = []
    reasons: dict[str, list[str]] = {}
    accepted_enforce: list[MemoryEntry] = []

    for entry in learnings:
        entry_reasons = _inspect(entry)
        if entry_reasons:
            would_reject.append(entry)
            reasons[entry.id] = entry_reasons
        else:
            accepted_enforce.append(entry)

    accepted = list(learnings) if observe_mode else accepted_enforce

    elapsed_ms = (time.monotonic_ns() - t0) / 1_000_000.0

    _LOG.info(
        "recall_filter.observe" if observe_mode else "recall_filter.enforce",
        window_size=len(learnings),
        would_reject_count=len(would_reject),
        latency_ms=round(elapsed_ms, 3),
        observe_mode=observe_mode,
    )

    if elapsed_ms > _LATENCY_BUDGET_MS and len(learnings) <= 25:
        _LOG.warning(
            "recall_filter.latency_budget_exceeded",
            latency_ms=round(elapsed_ms, 3),
            budget_ms=_LATENCY_BUDGET_MS,
            window_size=len(learnings),
        )

    return RecallFilterResult(accepted=accepted, would_reject=would_reject, reasons=reasons)
