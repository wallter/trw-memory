"""Distilled-tiering + entry-to-result helpers.

Belongs to ``client.py``. Re-exported there for back-compat.

PRD-DIST-005 FR-6 recall-side tiering for distilled records (records
written by trw-distill — source ``distilled:*`` or tag prefix
``distill:``). Plus the shared ``_entry_to_result`` converter used
by recall + org-shared paths.

5 helpers + 1 constant:

- ``DEFAULT_DISTILLED_RECALL_WEIGHT`` — 0.75 default dampening factor.
- ``get_distilled_recall_weight`` — env override + range validation.
- ``is_distilled_result`` — detect distilled records via tag/metadata.
- ``apply_distilled_tiering`` — dampen distilled scores or filter them.
- ``entry_to_result`` — MemoryEntry → result dict.

Logger lookup goes through the parent module so test patches on
``trw_memory.client.logger`` propagate.

Extracted as PRD-DIST-246 batch 109.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.client import MemoryResultDict

DEFAULT_DISTILLED_RECALL_WEIGHT: float = 0.75


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger


def get_distilled_recall_weight() -> float:
    """Resolve the dampening weight from env or fall back to default.

    Invalid env values log a warning + use the default. Out-of-range
    values (outside [0.0, 1.0]) likewise log + fall back.
    """
    raw = os.environ.get("TRW_MEMORY_DISTILLED_RECALL_WEIGHT")
    if not raw:
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    try:
        weight = float(raw)
    except ValueError:
        _client_logger().warning(
            "distilled_recall_weight_invalid",
            raw=raw,
            default=DEFAULT_DISTILLED_RECALL_WEIGHT,
        )
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    if not 0.0 <= weight <= 1.0:
        _client_logger().warning(
            "distilled_recall_weight_out_of_range",
            raw=weight,
            default=DEFAULT_DISTILLED_RECALL_WEIGHT,
        )
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    return weight


def is_distilled_result(result: MemoryResultDict) -> bool:
    """True if the record was written by trw-distill.

    Recognizes distilled records via two complementary markers:
      - any tag starts with ``distill:`` or ``distilled:``
      - metadata.source starts with ``distilled:``
    """
    tags = result.get("tags", []) or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(("distill:", "distilled:")):
            return True
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        src = str(metadata.get("source", ""))
        if src.startswith("distilled:"):
            return True
    return False


def apply_distilled_tiering(
    results: list[MemoryResultDict],
    *,
    weight: float | None = None,
    include_distilled: bool = True,
) -> list[MemoryResultDict]:
    """Apply PRD-DIST-005 FR-6 tiering to a recall result list.

    When ``include_distilled=False``, distilled records are removed.
    When ``include_distilled=True`` and ``weight < 1.0``, distilled
    record scores are multiplied by ``weight`` and the list is
    re-sorted. At ``weight=1.0`` this is a no-op passthrough.

    The input list is not mutated; a new sorted list is returned.
    """
    if not include_distilled:
        return [r for r in results if not is_distilled_result(r)]

    effective_weight = weight if weight is not None else get_distilled_recall_weight()
    if effective_weight >= 1.0 - 1e-9:
        return list(results)

    dampened: list[MemoryResultDict] = []
    for r in results:
        if is_distilled_result(r):
            new_r = dict(r)
            new_r["score"] = float(r.get("score", 0.0)) * effective_weight
            dampened.append(new_r)  # type: ignore[arg-type]
        else:
            dampened.append(r)
    dampened.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return dampened


def entry_to_result(entry: MemoryEntry, score: float = 0.0) -> MemoryResultDict:
    """Convert a MemoryEntry to a result dict."""
    result: MemoryResultDict = {
        "memory_id": entry.id,
        "content": entry.content,
        "detail": entry.detail,
        "tags": list(entry.tags),
        "importance": entry.importance,
        "score": score,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "namespace": entry.namespace,
        "source": "local",
        "last_accessed_at": entry.last_accessed_at.isoformat() if entry.last_accessed_at is not None else "",
        "q_value": entry.q_value,
        "q_observations": entry.q_observations,
        "recurrence": entry.recurrence,
        "access_count": entry.access_count,
        "_relevance_hint": score,
    }
    if entry.metadata:
        result["metadata"] = dict(entry.metadata)
        if "anomaly_dimension" in entry.metadata:
            result["anomaly_dimension"] = entry.metadata["anomaly_dimension"]
        if "z_score" in entry.metadata:
            try:
                result["z_score"] = float(entry.metadata["z_score"])
            except ValueError:
                pass
    if entry.expires:
        result["expires"] = entry.expires
    return result
