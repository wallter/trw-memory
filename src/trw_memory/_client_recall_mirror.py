"""Recall-time tier mirroring — keep hot/warm tiers aligned with what a recall returned.

Split from ``_client_recall_helpers`` (effective-LOC gate, 2026-09-17). Both
helpers collect one payload per returned local row and hand the whole batch to
``remember_entries_data_in_tiers`` so the warm sidecar is rewritten once per
recall, not once per row. The names stay importable from
``_client_recall_helpers`` (identity re-export) for existing callers and tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from trw_memory.lifecycle.tiers._runtime import remember_entries_data_in_tiers
from trw_memory.retrieval.recall_selection import LocalCandidate

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict


def remember_results_in_tiers(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> None:
    """Keep the hot/warm tiers aligned with the entries callers actually saw."""
    recalled_at = datetime.now(timezone.utc).isoformat()
    payloads: list[dict[str, object]] = []
    for result in results:
        if result.get("source", "local") != "local":
            continue
        payload: dict[str, object] = {
            "id": result["memory_id"],
            "content": result["content"],
            "detail": result["detail"],
            "tags": result["tags"],
            "importance": result["importance"],
            "namespace": result["namespace"],
            "last_accessed_at": recalled_at,
        }
        if result["created_at"]:
            payload["created_at"] = result["created_at"]
        if result["updated_at"]:
            payload["updated_at"] = result["updated_at"]
        payloads.append(payload)
    remember_entries_data_in_tiers(client._config, payloads)


def remember_selected_candidates(
    client: MemoryClient, candidates: list[LocalCandidate], results: list[MemoryResultDict]
) -> None:
    rows = {(r["namespace"], r["memory_id"]): r for r in results}
    recalled_at = datetime.now(timezone.utc).isoformat()
    payloads: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.source != "local":
            continue
        row = rows[(candidate.entry.namespace, candidate.entry.id)]
        payload = candidate.entry.model_dump(mode="json")
        # Security-masked returned content is what may enter the cache; validity
        # and provenance still come from the authoritative entry, not projection.
        payload.update(content=row["content"], detail=row["detail"], last_accessed_at=recalled_at)
        payloads.append(payload)
    remember_entries_data_in_tiers(client._config, payloads)
