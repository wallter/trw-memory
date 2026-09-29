"""Entry-to-result helpers shared by the recall and org-shared paths.

``entry_to_result`` converts a MemoryEntry to the client result dict and
``candidate_to_result`` projects a LocalCandidate at the client boundary.

The module once also held ``apply_distilled_tiering``, a second distilled
weighting path with no production caller; PRD-CORE-336 FR03 deleted it. The
distilled weight now has one scoring point,
``trw_memory.retrieval.source_policy.weight_distilled``, and its env override
one reader, ``SourcePolicy.resolve``.

Extracted as PRD-DIST-246 batch 109.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate

if TYPE_CHECKING:
    from trw_memory.client import MemoryResultDict


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


def candidate_to_result(candidate: LocalCandidate) -> MemoryResultDict:
    """Project only at the client boundary, retaining raw candidate state elsewhere."""
    result = entry_to_result(candidate.entry, candidate.raw_score)
    result["source"] = candidate.source
    return result
