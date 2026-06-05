"""Recall-time admission policy — confidence / currentness filter.

Shared recall-policy Module applied by BOTH recall Interfaces so they cannot
drift:

- SDK path: ``MemoryClient.recall`` -> ``_client_recall.recall_impl`` (consumes
  this via the ``_client_recall_helpers`` back-compat re-export).
- MCP tool path: ``tools.recall.memory_recall_impl``.

Lives in ``retrieval/`` next to :mod:`trw_memory.retrieval.source_policy` (the
other shared recall-policy Module) so neither the client cluster nor the tool
adapter has to reach into the other's internals.

The filter is generalized over the result-dict type via a ``TypeVar`` bound to
``Mapping[str, object]`` so the client's typed ``MemoryResultDict`` and the
tool surface's plain ``dict[str, object]`` share a single Implementation — the
seam that previously let ``memory_recall_impl`` bypass the policy the SDK path
enforced.

PRD-DIST-2049 c802 (origin) / recall-policy seam unification.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

_R = TypeVar("_R", bound=Mapping[str, object])


def apply_admission_filter(
    results: list[_R],
    *,
    confidence_floor: float | None,
    exclude_historical_only: bool,
    namespace: str = "",
) -> list[_R]:
    """Opt-in recall-time confidence / currentness filter.

    When both args are falsy (None / False) returns the input list unchanged
    (bit-for-bit; satisfies the default-OFF regression guarantee shared by both
    recall paths).

    OR semantics — a record is suppressed if EITHER:

    - ``confidence_floor`` is a float and the record's
      ``metadata['confidence']`` is below it (records without a confidence
      field fall back to their ``importance`` field, then to 0.0, so they are
      dropped when a floor is set and no signal is present).
    - ``exclude_historical_only`` is True and the record's
      ``metadata['currentness_status']`` equals ``'historical_only'``.

    Emits a ``recall_filter.admission`` structlog event when at least one
    record was suppressed, mirroring the ``recall_filter.enforce`` and
    ``anomaly_quarantine_bypass`` event-name conventions.
    """
    if confidence_floor is None and not exclude_historical_only:
        return results
    kept: list[_R] = []
    dropped_confidence = 0
    dropped_historical = 0
    for r in results:
        raw_meta = r.get("metadata")
        meta: Mapping[str, object] = raw_meta if isinstance(raw_meta, Mapping) else {}
        if exclude_historical_only and meta.get("currentness_status") == "historical_only":
            dropped_historical += 1
            continue
        if confidence_floor is not None:
            # Prefer explicit metadata.confidence when present; fall back to the
            # result's ``importance`` field which carries the producer-supplied
            # confidence on trw-distill ingest paths.
            raw: object = meta.get("confidence")
            if raw is None:
                raw = r.get("importance")
            try:
                conf = float(raw) if raw is not None else 0.0  # type: ignore[arg-type]
            except (TypeError, ValueError):
                conf = 0.0
            if conf < confidence_floor:
                dropped_confidence += 1
                continue
        kept.append(r)
    if dropped_confidence or dropped_historical:
        logger.debug(
            "recall_filter.admission",
            operation="recall_admission_filter",
            namespace=namespace,
            confidence_floor=confidence_floor,
            exclude_historical_only=exclude_historical_only,
            input_count=len(results),
            kept_count=len(kept),
            dropped_confidence=dropped_confidence,
            dropped_historical=dropped_historical,
        )
    return kept
