"""The evidence-level data invariant (PRD-CORE-312, redesigned after 3 review rounds).

CONSTITUTION SS2: "an untagged conclusion must never be re-ingested as fact" --
``confidence='verified'`` requires ``evidence_level`` in ``{observed, verified}``.

The first implementation put this check at ONE write chokepoint
(``poisoning.reject_unsubstantiated_verified``), reached from ``validate_entry_payload``
(the store path) and ``correction._refuse_unsubstantiated`` (the update path). Three
review rounds each found a new bypass on that same boundary: an evidence-only edit
skipped the check (round 1), the round-1 fix then stuck every LEGACY verified row
from ever being edited again (round 2), and round 3 found that consolidation's and
sync's own ``backend.store()``/``backend.update()`` calls never reached either
chokepoint at all -- a per-call-site check cannot be reached from every call site
by construction.

This module makes it a DATA INVARIANT instead, enforced in the storage layer
every entry-producing path already goes through (SQLite ``_crud_ops.store``/
``store_many``/``update``, and ``YAMLBackend.store``/``update``),
using an existing-vs-new comparison rather than a per-caller "did this patch touch
confidence/evidence_level" heuristic:

* :func:`refuse_new_violation` (WRITE time): refuses only when a write would
  INTRODUCE the violation that did not already hold for the row being replaced.
  A brand-new entry (``existing=None``) that violates is always refused. An
  edited entry that was ALREADY violating (a legacy row, grandfathered before
  this invariant existed) is never blocked from an otherwise-unrelated edit
  (retiring it, re-tagging it) merely for carrying an old violation forward
  unchanged -- that would re-create round 2's regression. The exemption covers
  only the SAME claim: a write that changes ``content`` or ``detail`` authors a
  new claim and must meet the invariant itself (delta review v2 round 2). Soundness scope: this
  proves no *newly written* entry claims verified without observed/verified
  evidence; it does not retroactively fix a pre-existing violating row (that is
  what read-time demotion and a deliberate S4 correction are for), and it does
  not prove the evidence content itself is true.
* :func:`served_view` (READ time): applied at every entry-returning backend
  boundary (SQLite ``get``/``update`` and the ``_resilient_fetch`` materializers
  behind search/list/FTS; YAML ``get``/``update``/``_iter_all``). It never
  rewrites storage itself -- though a caller that writes a served copy back
  persists the demotion, the fail-safe direction. A row already
  violating the invariant is presented with its confidence demoted to
  ``unverified`` and a ``served_confidence_demoted`` metadata marker, so a
  legacy violation is never silently claimed as fact -- and never silently
  fixed either.
"""

from __future__ import annotations

import structlog

from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import Confidence, EvidenceLevel, MemoryEntry

logger = structlog.get_logger(__name__)

__all__ = ["refuse_new_violation", "served_view", "violates_evidence_invariant"]

#: Metadata key a demoted-in-the-served-view row carries (never persisted by this
#: module itself -- ``served_view`` returns a copy, and nothing here writes it back).
SERVED_CONFIDENCE_DEMOTED_KEY = "served_confidence_demoted"


def violates_evidence_invariant(entry: MemoryEntry) -> bool:
    """True iff *entry* claims ``confidence='verified'`` without Observed/Verified evidence."""
    return entry.confidence == Confidence.VERIFIED and entry.evidence_level not in (
        EvidenceLevel.OBSERVED,
        EvidenceLevel.VERIFIED,
    )


def refuse_new_violation(existing: MemoryEntry | None, entry: MemoryEntry) -> None:
    """Refuse *entry* only if it introduces a violation *existing* did not already carry.

    Args:
        existing: The row being replaced/updated, or ``None`` for a genuinely new entry.
        entry: The entry as it would be persisted by this write.

    Raises:
        SchemaValidationError: *entry* violates the invariant and is not the same
            claim (``content``/``detail``) as an *existing* row that already
            violated it -- a NEW violation, refused fail-closed.
    """
    if not violates_evidence_invariant(entry):
        return
    if (
        existing is not None
        and violates_evidence_invariant(existing)
        and (existing.content, existing.detail or "") == (entry.content, entry.detail or "")
    ):
        return  # the same legacy claim, carried forward unchanged
    logger.warning(
        "unsubstantiated_verified_rejected",
        entry_id=entry.id,
        reason="verified_requires_observed_or_verified_evidence",
    )
    # Entries built via ``model_copy(update=...)`` or a raw ``setattr`` (no
    # validation) may carry a bare str rather than an EvidenceLevel instance;
    # ``str(enum_member)`` on a ``str, Enum`` mixin renders "ClassName.MEMBER",
    # not the plain value, so the two cases need separate handling.
    level = entry.evidence_level.value if isinstance(entry.evidence_level, EvidenceLevel) else str(entry.evidence_level)
    raise SchemaValidationError(
        f"confidence='verified' requires evidence_level in "
        f"{{{EvidenceLevel.OBSERVED.value!r}, {EvidenceLevel.VERIFIED.value!r}}}, "
        f"got evidence_level={level!r}",
        failed_fields=["confidence", "evidence_level"],
        reason="verified_requires_observed_or_verified_evidence",
    )


def served_view(entry: MemoryEntry) -> MemoryEntry:
    """The entry as it should be SERVED to a caller: demoted, never rewritten.

    A row that already violates the invariant (grandfathered before it existed,
    or otherwise carried forward per :func:`refuse_new_violation`) is never
    presented as a verified fact. Storage is untouched; only the returned copy
    is demoted, so a subsequent read of the same row sees the same demotion
    again -- deterministic, not a one-time silent fix.
    """
    if not violates_evidence_invariant(entry):
        return entry
    return entry.model_copy(
        update={
            "confidence": Confidence.UNVERIFIED,
            "metadata": {**entry.metadata, SERVED_CONFIDENCE_DEMOTED_KEY: "true"},
        }
    )
