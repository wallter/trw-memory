"""Recall-window filtering and security telemetry for memory retrieval."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from trw_memory.models.memory import MemoryEntry
from trw_memory.security.recall_filter import filter_recall_window
from trw_memory.security.runtime import probe_canaries
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict


def apply_recall_security(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> list[MemoryResultDict]:
    """Apply canary checks and the configured recall-window policy."""
    if client._backend is not None:
        probe_canaries(client._config, backend=client._backend)
    if not client._config.enable_recall_filter:
        return results
    score_by_id: dict[str, float] = {}
    result_by_id: dict[str, MemoryResultDict] = {}
    entries: list[MemoryEntry] = []
    for idx, result in enumerate(results):
        synthetic_id = f"{result['namespace']}::{result['memory_id']}::{idx}"
        score_by_id[synthetic_id] = result["score"]
        result_by_id[synthetic_id] = result
        raw_result = {"id": synthetic_id, **result}
        recalled_at = datetime.now(timezone.utc)
        for timestamp_field in ("created_at", "updated_at"):
            if raw_result.get(timestamp_field) in {"", "None", None}:
                raw_result[timestamp_field] = recalled_at
        if raw_result.get("last_accessed_at") in {"", "None", None}:
            raw_result["last_accessed_at"] = None
        entries.append(MemoryEntry.model_validate(raw_result))
    filtered = filter_recall_window(entries, mode=client._config.recall_filter_mode)
    session_id = os.environ.get("TRW_SESSION_ID", "").strip() or client._namespace
    run_id = os.environ.get("TRW_RUN_ID", "").strip() or None
    emit_security_event(
        client._config,
        emitter="recall_filter",
        session_id=session_id,
        run_id=run_id,
        payload={
            "event_name": "recall_filter_outcome",
            "path": "client_recall",
            "namespace": client._namespace,
            "mode": client._config.recall_filter_mode,
            "window_size": len(entries),
            "accepted_count": len(filtered.accepted),
            "would_reject_count": len(filtered.would_reject),
            "actions": dict(filtered.actions),
            "traceability": build_security_traceability(
                live_path="client.MemoryClient._apply_recall_security",
                requirement_ids=["FR-003", "NFR-010", "NFR-011"],
            ),
        },
    )
    secured: list[MemoryResultDict] = []
    for entry in filtered.accepted:
        if entry.metadata.get("system_canary") == "true":
            continue
        original = dict(result_by_id[entry.id])
        original["content"] = entry.content
        original["detail"] = entry.detail
        original["metadata"] = dict(entry.metadata)
        original["score"] = score_by_id[entry.id]
        secured.append(cast("MemoryResultDict", original))
    return secured
