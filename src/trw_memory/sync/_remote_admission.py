"""The one admission gate every remote item passes before a caller sees it.

PRD-CORE-245 FR06. Three paths used to bring peer content in and only one was
gated: ``trw-mcp``'s sync pull ran ``prepare_entry_for_store`` and quarantined
refusals, while ``fetch_shared_memories`` here and a duplicate HTTP client in
``trw_mcp.telemetry.remote_recall`` prefixed results with ``[shared]`` and
handed them straight back into the recall response.

That is the more dangerous shape, not the safer one. Content that never lands in
the store is also content no later audit or quarantine sweep can reach: it goes
directly into an agent's context, which is the prompt-injection surface the
2026-04-18 security review recorded. This module closes it by running the SAME
gate on a fetched result before it is returned.

Fail-closed by construction: an item the gate refuses is dropped AND
quarantined, and an item the gate could not evaluate at all -- an adapter error,
a malformed payload -- is treated as a refusal rather than a pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import structlog

from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["SHARED_NAMESPACE", "AdmissionOutcome", "admit_remote_results"]

#: Namespace the gate evaluates peer content under. A real namespace in the
#: existing grammar (``org:`` scope), not a carve-out: a caller that wants shared
#: results must hold it in its ``NamespaceScope`` like any other (PRD-CORE-245
#: FR06). The RESPONSE still labels these results ``shared`` -- that is a display
#: field on the wire vocabulary, which this PRD explicitly does not change.
SHARED_NAMESPACE = "org:shared"


def _lift(result: dict[str, object]) -> MemoryEntry | None:
    """Build the MemoryEntry the gate evaluates, or None if the payload cannot be read.

    Deliberately minimal: it carries the fields the gate actually inspects
    (content, detail, tags, namespace) and nothing that would let a peer-supplied
    value influence the verdict. A payload this cannot read is unevaluable, and
    unevaluable is a refusal.
    """
    try:
        summary = result.get("summary", result.get("content", ""))
        raw_tags = result.get("tags", [])
        return MemoryEntry(
            id=str(result.get("source_learning_id") or result.get("id") or "remote-candidate"),
            content=str(summary or ""),
            detail=str(result.get("detail", "")),
            tags=[str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else [],
            namespace=SHARED_NAMESPACE,
            source="agent",
        )
    except (TypeError, ValueError):
        logger.debug("remote_admission_unliftable", exc_info=True)
        return None


class AdmissionOutcome(NamedTuple):
    """What the gate passed, and how much of what it saw it did not.

    The counts exist because the admitted LIST alone cannot express the
    difference between "the peer returned nothing" and "the peer returned ten
    items and the gate refused all ten" -- both are ``[]``. The second is a
    signal about the peer (or about this store's policy) that a caller may want
    to report; discarding it into a log line put it out of reach of every
    caller.
    """

    admitted: list[dict[str, object]]
    #: Items the gate evaluated and rejected, plus items too malformed to lift.
    refused: int
    #: Subset of ``refused`` where the gate itself raised. Counted separately
    #: because "policy says no" and "policy could not be applied" are different
    #: facts, even though both fail closed.
    gate_errors: int


def admit_remote_results(
    results: list[dict[str, object]],
    *,
    config: MemoryConfig,
    backend: StorageBackend,
) -> AdmissionOutcome:
    """Return the results the admission gate passed, with the refusal counts.

    A refused item is never returned to a caller and therefore never reaches an
    agent's context, and a quarantine record exists for it so the refusal is
    auditable rather than silent.
    """
    from trw_memory.security.runtime import prepare_entry_for_store, store_quarantined_entry

    admitted: list[dict[str, object]] = []
    refused = 0
    gate_errors = 0
    for result in results:
        entry = _lift(result)
        if entry is None:
            refused += 1
            continue
        try:
            decision = prepare_entry_for_store(entry, backend=backend, config=config)
        except Exception:  # justified: an unevaluable item is a REFUSAL, never a pass
            logger.warning("remote_admission_gate_error", outcome="refused", exc_info=True)
            refused += 1
            gate_errors += 1
            continue
        if decision.quarantined:
            store_quarantined_entry(config, decision.entry)
            refused += 1
            continue
        admitted.append(result)
    if refused:
        logger.info("remote_admission_refused", refused=refused, admitted=len(admitted), gate_errors=gate_errors)
    return AdmissionOutcome(admitted, refused, gate_errors)
