"""Store async impl.

Belongs to ``client.py``. Re-exported there for back-compat.

Helpers extracted from ``MemoryClient.store(...)``:

- ``_build_store_entry`` — shared new/update entry construction for single and bulk stores.
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

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from trw_memory._client_hype import expand_hype_siblings
from trw_memory.exceptions import MemoryNotFoundError, SchemaValidationError, StorageError
from trw_memory.graph import schedule_graph_update
from trw_memory.lifecycle.tiers._runtime import embedding_has_consumer, remember_entry_in_tiers
from trw_memory.models.entry_factory import new_entry, revise_entry
from trw_memory.models.memory import Assertion, MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import (
    append_audit_event,
    prepare_entry_for_store,
    store_quarantined_entry,
)

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, StoreResultDict
    from trw_memory.storage.interface import StorageBackend


def _existing_entry_for_namespace(backend: StorageBackend, entry_id: str, namespace: str) -> MemoryEntry | None:
    """Resolve an update target without exposing or mutating another namespace."""
    existing = backend.get(entry_id, namespace=namespace)
    if not isinstance(existing, MemoryEntry):
        return None
    # Defence in depth. Containment is enforced by the query itself under
    # PRD-CORE-245 FR03, so this can only fire for a backend implementation that
    # ignores its own namespace predicate — in which case failing loudly beats
    # handing another namespace's row to an update path.
    if existing.namespace != namespace:
        raise MemoryNotFoundError(f"Memory entry {entry_id!r} not found in namespace {namespace!r}")
    return existing


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger


def _make_id() -> str:
    from trw_memory.client import _make_id as _impl

    return _impl()


def _build_store_entry(
    *,
    memory_id: str,
    existing: MemoryEntry | None,
    content: str,
    detail: str,
    tags: list[str] | None,
    evidence: list[str] | None,
    importance: float,
    namespace: str,
    metadata: dict[str, str] | None,
    expires: str,
    assertions: list[Assertion] | None,
    source: Literal["human", "agent", "tool", "consolidated"],
    source_identity: str,
    now: datetime,
    installation_id: str,
    local_node_id: str,
) -> MemoryEntry:
    """Build the identical new/update entry shape for single and bulk stores."""
    entry_metadata = dict(existing.metadata) if existing is not None else {}
    entry_metadata.update(metadata or {})
    entry_metadata.setdefault("installation_id", installation_id)
    entry_expires = expires or (existing.expires if existing is not None else "")

    if existing is None:
        return new_entry(
            entry_id=memory_id,
            content=content.strip(),
            namespace=namespace,
            local_node_id=local_node_id,
            now=now,
            fields={
                "detail": detail,
                "tags": tags or [],
                "evidence": list(evidence or []),
                "importance": importance,
                "metadata": entry_metadata,
                "expires": entry_expires,
                "assertions": list(assertions or []),
                "source": source,
                "source_identity": source_identity,
            },
        )

    return revise_entry(
        existing,
        local_node_id=local_node_id,
        now=now,
        fields={
            "content": content.strip(),
            "detail": detail,
            "tags": tags or [],
            "evidence": list(evidence) if evidence is not None else existing.evidence,
            "importance": importance,
            "metadata": entry_metadata,
            "expires": entry_expires,
            "assertions": list(assertions) if assertions is not None else existing.assertions,
            "source": source,
            "source_identity": source_identity or existing.source_identity,
        },
    )


async def store_impl(
    client: MemoryClient,
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
    expires: str = "",
    evidence: list[str] | None = None,
    assertions: list[Assertion] | None = None,
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
        existing = (
            _existing_entry_for_namespace(backend, memory_id, client._namespace) if entry_id is not None else None
        )
        now = datetime.now(timezone.utc)
        entry = _build_store_entry(
            memory_id=memory_id,
            existing=existing,
            content=content,
            detail=detail,
            tags=tags,
            evidence=evidence,
            importance=importance,
            namespace=client._namespace,
            metadata=metadata,
            expires=expires,
            assertions=assertions,
            source=source,
            source_identity=source_identity,
            now=now,
            installation_id=client._installation_id,
            local_node_id=client._local_node_id,
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
        # Only pay for the embedding model when a vector sink can actually
        # consume the result; otherwise every downstream upsert_vector no-ops.
        embedder = client._get_embedder() if embedding_has_consumer(client._config, backend) else None
        embedding = (
            await asyncio.to_thread(embedder.embed, f"{entry.content} {entry.detail}") if embedder is not None else None
        )
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
                    backend.upsert_vector(entry.id, embedding, namespace=entry.namespace)
                # PRD-CORE-195 FR03/FR05: generate + store HyPE sibling vectors
                # inside the SAME transaction (purge-then-regenerate on UPDATE).
                # Gated on hype_enabled; fail-open so the canonical row + primary
                # vector always commit even if generation/embedding raises.
                expand_hype_siblings(
                    backend=backend,
                    config=client._config,
                    entry=entry,
                    embedder=embedder,
                    generator=client._question_generator,
                )
        except Exception as exc:
            raise StorageError(f"failed to persist entry+vector for {entry.id!r}; transaction rolled back") from exc
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
