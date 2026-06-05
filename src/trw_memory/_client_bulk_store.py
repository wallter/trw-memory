"""Bulk-store cluster — dataclasses + async impl.

Belongs to ``client.py``. Re-exported there for back-compat.

Three Pydantic-style dataclasses + one async impl helper amortise
per-item lock + audit + embed overhead for ingestion-heavy workloads
(audits, distill batch ingestion). See learning L-ujVK for the
per-item-overhead measurement that motivated this API.

Public API (all re-exported from ``client.py``):

- ``BulkStoreRequest`` — one record per ``bulk_store(...)`` batch.
- ``BulkStoreItemResult`` — per-item status + diagnostic fields.
- ``BulkStoreSummary`` — aggregate counts + per-item results.
- ``bulk_store_impl`` — async helper invoked by
  ``MemoryClient.bulk_store`` (takes the client instance as first
  arg; uses backend-handle pattern so test patches still propagate).

Extracted as PRD-DIST-246 batch 104.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

from trw_memory.exceptions import SchemaValidationError, StorageError
from trw_memory.graph import schedule_graph_update
from trw_memory.lifecycle.tiers._runtime import embedding_has_consumer, remember_entry_in_tiers
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import (
    append_audit_event,
    prepare_entry_for_store,
    store_quarantined_entry,
)
from trw_memory.sync.conflict import increment_clock, init_clock

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BulkStoreRequest:
    """One record in a ``MemoryClient.bulk_store(...)`` batch.

    Mirrors the per-item kwargs of ``MemoryClient.store`` so callers can
    migrate per-item loops to a single batch call without re-shaping
    their data.
    """

    content: str
    detail: str = ""
    tags: list[str] | None = None
    importance: float = 0.5
    metadata: dict[str, str] | None = None
    source: Literal["human", "agent", "tool", "consolidated"] = "agent"
    source_identity: str = ""
    session_id: str | None = None
    entry_id: str | None = None


@dataclass
class BulkStoreItemResult:
    """Per-item result inside a ``BulkStoreSummary.items`` list (input-order).

    ``status`` is one of ``"stored"`` / ``"updated"`` / ``"quarantined"``
    / ``"rejected"``. ``rejected`` rows carry a ``skipped_reason``;
    ``quarantined`` rows include the anomaly diagnostic fields.
    """

    memory_id: str
    status: str
    quarantined: bool = False
    skipped_reason: str = ""
    anomaly_dimension: str = ""
    z_score: float = 0.0


@dataclass
class BulkStoreSummary:
    """Aggregate + per-item result of a ``MemoryClient.bulk_store(...)`` call."""

    total: int
    stored: int
    updated: int
    quarantined: int
    rejected: int
    duration_ms: float
    items: list[BulkStoreItemResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        """Count of records that landed in the main store (stored + updated)."""
        return self.stored + self.updated

    @property
    def per_item_ms(self) -> float:
        """Mean wall-time per record across the whole batch."""
        return self.duration_ms / self.total if self.total else 0.0


def _make_id() -> str:
    from trw_memory.client import _make_id as _client_make_id

    return _client_make_id()


async def bulk_store_impl(
    client: MemoryClient,
    requests: list[BulkStoreRequest],
    *,
    skip_audit_per_item: bool = True,
    skip_remote_publish: bool = True,
) -> BulkStoreSummary:
    """Async impl for :meth:`MemoryClient.bulk_store`.

    Refer to the docstring on the method for arg/return semantics.
    Implementation here lives outside the class so ``client.py`` can
    stay under the 350 effective-LOC gate (PRD-DIST-246).
    """
    if not requests:
        raise ValueError("bulk_store requires at least one request")

    client._require_permission(Permission.WRITE, "bulk_store")
    client._maybe_start_retry_drain()

    start_ts = datetime.now(timezone.utc)

    prepared: list[tuple[BulkStoreRequest, str | None]] = []
    for req in requests:
        try:
            validate_store_inputs(
                content=req.content,
                detail=req.detail,
                tags=req.tags,
                metadata=req.metadata,
                importance=req.importance,
            )
            prepared.append((req, None))
        except SchemaValidationError as exc:
            prepared.append((req, f"schema_invalid:{','.join(exc.failed_fields)}"))

    items: list[BulkStoreItemResult] = []
    embedder = client._get_embedder()
    accepted_indices: list[int] = []
    accepted_entries: list[MemoryEntry] = []

    async with client._lock:
        backend = client._get_backend()
        now = datetime.now(timezone.utc)

        decisions: list[Any] = [None] * len(prepared)
        for i, (req, validation_error) in enumerate(prepared):
            if validation_error is not None:
                items.append(
                    BulkStoreItemResult(
                        memory_id=req.entry_id or "",
                        status="rejected",
                        skipped_reason=validation_error,
                    )
                )
                continue

            memory_id = req.entry_id or _make_id()
            existing = backend.get(memory_id) if req.entry_id is not None else None
            entry_metadata = dict(existing.metadata) if existing is not None else {}
            entry_metadata.update(req.metadata or {})
            entry_metadata.setdefault("installation_id", client._installation_id)

            if existing is None:
                entry = MemoryEntry(
                    id=memory_id,
                    content=req.content.strip(),
                    detail=req.detail,
                    tags=req.tags or [],
                    importance=req.importance,
                    namespace=client._namespace,
                    metadata=entry_metadata,
                    created_at=now,
                    updated_at=now,
                    source=req.source,
                    source_identity=req.source_identity,
                    vector_clock=init_clock(client._local_node_id),
                )
            else:
                entry = existing.model_copy(
                    update={
                        "content": req.content.strip(),
                        "detail": req.detail,
                        "tags": req.tags or [],
                        "importance": req.importance,
                        "metadata": entry_metadata,
                        "updated_at": now,
                        "source": req.source,
                        "source_identity": req.source_identity or existing.source_identity,
                        # FR04: advance, do not reset, the local node's counter on
                        # edit (matches store_impl); a reset stalled causality at
                        # {node: 1}. See test_sync_clock_monotonic.
                        "vector_clock": increment_clock(existing.vector_clock, client._local_node_id),
                    }
                )

            try:
                decision = prepare_entry_for_store(
                    entry,
                    backend=backend,
                    config=client._config,
                    session_id=req.session_id,
                )
            except Exception as exc:
                items.append(
                    BulkStoreItemResult(
                        memory_id=memory_id,
                        status="rejected",
                        skipped_reason=f"{type(exc).__name__}:{str(exc)[:80]}",
                    )
                )
                continue

            if decision.quarantined:
                store_quarantined_entry(client._config, decision.entry)
                items.append(
                    BulkStoreItemResult(
                        memory_id=decision.entry.id,
                        status="quarantined",
                        quarantined=True,
                        anomaly_dimension=decision.anomaly_dimension,
                        z_score=decision.anomaly_z_score,
                    )
                )
                continue

            accepted_indices.append(i)
            accepted_entries.append(decision.entry)
            decisions[i] = decision

        embeddings: list[list[float] | None] = []
        # Skip the batch embed when no vector sink can consume the result — the
        # warm tier, primary vector store, and remote publish are the only
        # consumers, and all three are inert here when this returns False.
        if accepted_entries and embedder is not None and embedding_has_consumer(client._config, backend):
            try:
                texts = [f"{e.content} {e.detail}" for e in accepted_entries]
                embeddings = embedder.embed_batch(texts)
            except Exception as exc:
                logger.warning(
                    "bulk_store_embed_batch_failed",
                    op="bulk_store",
                    n=len(accepted_entries),
                    error=str(exc),
                )
                embeddings = [None] * len(accepted_entries)
        else:
            embeddings = [None] * len(accepted_entries)

        for j, (orig_i, entry) in enumerate(zip(accepted_indices, accepted_entries, strict=False)):
            decision = decisions[orig_i]
            assert decision is not None  # noqa: S101
            embedding = embeddings[j] if j < len(embeddings) else None

            if client._namespace.startswith("team:"):
                NamespaceManager(backend).ensure_team_namespace(client._namespace, created_at=now)

            # S1 fix (mirrors _client_store): per-entry row+vector atomicity.
            # Each entry keeps its own commit granularity (matching the prior
            # per-entry commit), but row and vector now land in ONE transaction
            # so a crash between them can't leave a row with no vector. No manual
            # compensating delete — the block's ROLLBACK handles failure.
            try:
                with backend.transaction():
                    backend.store(entry)
                    if embedding is not None:
                        backend.upsert_vector(entry.id, embedding)
            except Exception as exc:
                raise StorageError(
                    f"failed to persist entry+vector for {entry.id!r}; transaction rolled back"
                ) from exc

            try:
                schedule_graph_update(entry, backend, embedding=embedding, config=client._config)
            except RuntimeError:
                logger.warning(
                    "bulk_store_graph_schedule_failed",
                    memory_id=entry.id,
                    exc_info=True,
                )

            remember_entry_in_tiers(client._config, client._namespace, entry, embedding)

            if not skip_audit_per_item:
                append_audit_event(
                    client._config,
                    decision.op,
                    entry_id=entry.id,
                    actor=entry.source_identity or entry.source,
                    namespace=client._namespace,
                    data={
                        "status": "updated" if decision.op == "update" else "stored",
                        "session_id": prepared[orig_i][0].session_id,
                        "pii_types": sorted({m.pii_type for m in decision.pii_matches}),
                        "quarantined": False,
                    },
                )

            items.append(
                BulkStoreItemResult(
                    memory_id=entry.id,
                    status="updated" if decision.op == "update" else "stored",
                )
            )

            if not skip_remote_publish and client._should_attempt_remote_publish(entry):
                client._schedule_background_task(client._publish_entry(entry, embedding))

    stored_count = sum(1 for it in items if it.status == "stored")
    updated_count = sum(1 for it in items if it.status == "updated")
    quarantined_count = sum(1 for it in items if it.status == "quarantined")
    rejected_count = sum(1 for it in items if it.status == "rejected")
    end_ts = datetime.now(timezone.utc)
    duration_ms = (end_ts - start_ts).total_seconds() * 1000.0

    if skip_audit_per_item:
        append_audit_event(
            client._config,
            "bulk_store",
            entry_id="",
            actor=requests[0].source_identity or requests[0].source if requests else "agent",
            namespace=client._namespace,
            data={
                "total": len(requests),
                "stored": stored_count,
                "updated": updated_count,
                "quarantined": quarantined_count,
                "rejected": rejected_count,
                "duration_ms": round(duration_ms, 2),
            },
        )

    logger.info(
        "memory_bulk_stored",
        op="bulk_store",
        outcome="success",
        namespace=client._namespace,
        total=len(requests),
        stored=stored_count,
        updated=updated_count,
        quarantined=quarantined_count,
        rejected=rejected_count,
        duration_ms=round(duration_ms, 2),
    )

    return BulkStoreSummary(
        total=len(requests),
        stored=stored_count,
        updated=updated_count,
        quarantined=quarantined_count,
        rejected=rejected_count,
        duration_ms=duration_ms,
        items=items,
    )
