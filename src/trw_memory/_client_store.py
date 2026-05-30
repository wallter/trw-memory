"""Store async impl.

Belongs to ``client.py``. Re-exported there for back-compat.

Single helper extracted from ``MemoryClient.store(...)``:

- ``store_impl`` — full per-entry write path: schema validation,
  permission check, retry-drain, prepare_entry_for_store gate
  (PII / poisoning / anomaly), backend.store, vector upsert with
  rollback-on-failure, graph schedule, tier-runtime register,
  audit event, debug log, optional remote publish.

Uses backend-handle pattern (takes ``client: MemoryClient`` as first
arg). Logger lookup goes through the parent module so test patches on
``trw_memory.client.logger`` propagate.

Extracted as PRD-DIST-246 batch 110.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

from trw_memory.exceptions import SchemaValidationError, StorageError
from trw_memory.graph import schedule_graph_update
from trw_memory.lifecycle.tiers._runtime import remember_entry_in_tiers
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import (
    append_audit_event,
    prepare_entry_for_store,
    store_quarantined_entry,
)
from trw_memory.sync.conflict import init_clock

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, StoreResultDict

logger = structlog.get_logger(__name__)


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger


def _make_id() -> str:
    from trw_memory.client import _make_id as _impl

    return _impl()


async def store_impl(
    client: MemoryClient,
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
    expires: str = "",
    *,
    source: Literal["human", "agent", "tool", "consolidated"] = "agent",
    source_identity: str = "",
    session_id: str | None = None,
    entry_id: str | None = None,
) -> StoreResultDict:
    """Async impl for :meth:`MemoryClient.store`.

    See the method docstring for full arg/return semantics. Implementation
    here lives outside the class so ``client.py`` clears the 350-LOC gate.
    """
    try:
        validate_store_inputs(content=content, detail=detail, tags=tags, metadata=metadata, importance=importance)
    except SchemaValidationError as exc:
        append_audit_event(
            client._config,
            "store_rejected",
            entry_id=entry_id or "",
            actor=source_identity or source,
            namespace=client._namespace,
            data={"reason": "schema_invalid", "failed_fields": exc.failed_fields, "session_id": session_id},
        )
        raise
    client._require_permission(Permission.WRITE, "store")
    client._maybe_start_retry_drain()

    memory_id = entry_id or _make_id()
    async with client._lock:
        backend = client._get_backend()
        existing = backend.get(memory_id) if entry_id is not None else None
        now = datetime.now(timezone.utc)
        entry_metadata = dict(existing.metadata) if existing is not None else {}
        entry_metadata.update(metadata or {})
        entry_metadata.setdefault("installation_id", client._installation_id)
        entry_expires = expires or (existing.expires if existing is not None else "")

        if existing is None:
            entry = MemoryEntry(
                id=memory_id,
                content=content.strip(),
                detail=detail,
                tags=tags or [],
                importance=importance,
                namespace=client._namespace,
                metadata=entry_metadata,
                created_at=now,
                updated_at=now,
                expires=entry_expires,
                source=source,
                source_identity=source_identity,
                vector_clock=init_clock(client._local_node_id),
            )
        else:
            entry = existing.model_copy(
                update={
                    "content": content.strip(),
                    "detail": detail,
                    "tags": tags or [],
                    "importance": importance,
                    "metadata": entry_metadata,
                    "updated_at": now,
                    "expires": entry_expires,
                    "source": source,
                    "source_identity": source_identity or existing.source_identity,
                    "vector_clock": init_clock(client._local_node_id),
                }
            )

        decision = prepare_entry_for_store(
            entry,
            backend=backend,
            config=client._config,
            session_id=session_id,
        )
        if decision.quarantined:
            store_quarantined_entry(client._config, decision.entry)
            append_audit_event(
                client._config,
                "quarantine",
                entry_id=decision.entry.id,
                actor=decision.entry.source_identity or decision.entry.source,
                namespace=client._namespace,
                data={
                    "stored": False,
                    "quarantined": True,
                    "anomaly_dimension": decision.anomaly_dimension,
                    "z_score": decision.anomaly_z_score,
                },
            )
            quarantined_result: StoreResultDict = {
                "memory_id": decision.entry.id,
                "namespace": client._namespace,
                "status": "quarantined",
                "timestamp": now.isoformat(),
                "quarantined": True,
                "stored": False,
                "anomaly_dimension": decision.anomaly_dimension,
                "z_score": decision.anomaly_z_score,
            }
            return quarantined_result

        entry = decision.entry
        embedder = client._get_embedder()
        embedding = embedder.embed(f"{entry.content} {entry.detail}") if embedder is not None else None
        if client._namespace.startswith("team:"):
            NamespaceManager(backend).ensure_team_namespace(client._namespace, created_at=now)
        # S1 fix: commit the row + its vector in ONE transaction so a crash
        # between the two writes can no longer leave a row with no vector.
        # store() and upsert_vector() defer their commit inside the block
        # (PRD S9/S3); the outermost COMMIT lands both atomically, and any
        # exception triggers a single ROLLBACK — no manual compensating delete.
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
            _client_logger().warning("memory_store_graph_schedule_failed", memory_id=entry.id, exc_info=True)
        remember_entry_in_tiers(client._config, client._namespace, entry, embedding)
        append_audit_event(
            client._config,
            decision.op,
            entry_id=entry.id,
            actor=entry.source_identity or entry.source,
            namespace=client._namespace,
            data={
                "status": "updated" if decision.op == "update" else "stored",
                "session_id": session_id,
                "pii_types": sorted({match.pii_type for match in decision.pii_matches}),
                "quarantined": False,
            },
        )

    _client_logger().debug(
        "memory_stored",
        op="store",
        outcome="success",
        memory_id=memory_id,
        namespace=client._namespace,
        content_len=len(content),
        tag_count=len(tags or []),
        importance=importance,
    )
    if client._should_attempt_remote_publish(entry):
        client._schedule_background_task(client._publish_entry(entry, embedding))
    store_result: StoreResultDict = {
        "memory_id": memory_id,
        "namespace": client._namespace,
        "status": "updated" if decision.op == "update" else "stored",
        "timestamp": now.isoformat(),
    }
    return store_result
