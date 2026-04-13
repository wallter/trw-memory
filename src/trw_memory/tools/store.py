"""MCP tool: memory_store — persist a new memory entry.

Validates namespace, creates a MemoryEntry with a unique M-prefixed ID,
stores it via the backend, and returns the memory_id and status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import structlog

from trw_memory.embeddings import get_local_embedder
from trw_memory.exceptions import (
    ConfigError,
    PIIBlockError,
    PoisoningError,
    RateLimitError,
    SchemaValidationError,
    StorageError,
)
from trw_memory.graph import schedule_graph_update
from trw_memory.lifecycle.tiers._runtime import remember_entry_in_tiers, supports_tier_runtime
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import append_audit_event, prepare_entry_for_store, store_quarantined_entry
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def memory_store_impl(
    content: str,
    namespace: str,
    *,
    backend: StorageBackend,
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
    config: MemoryConfig | None = None,
    source: Literal["human", "agent", "tool", "consolidated"] = "tool",
    source_identity: str = "",
    session_id: str | None = None,
    entry_id: str | None = None,
    raise_security_errors: bool = False,
) -> dict[str, object]:
    """Core implementation of memory_store (callable without MCP).

    Args:
        content: Core knowledge statement to store. Must be non-empty.
        namespace: Namespace scope (e.g., "project:default", "global").
        backend: Storage backend instance.
        tags: Optional list of tags to associate with the entry.
        importance: Importance score in [0.0, 1.0]. Defaults to 0.5.
        detail: Extended explanation or context. Defaults to "".
        metadata: Optional string key-value metadata. Defaults to {}.

    Returns:
        {"memory_id": str, "status": "stored", "namespace": str}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    # Validate namespace
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.WRITE, "store")
    try:
        validate_store_inputs(content=content, detail=detail, tags=tags, metadata=metadata, importance=importance)
    except SchemaValidationError as exc:
        append_audit_event(
            cfg,
            "store_rejected",
            entry_id=entry_id or "",
            actor=source_identity or source,
            namespace=namespace,
            data={"reason": "schema_invalid", "failed_fields": exc.failed_fields, "session_id": session_id},
        )
        if raise_security_errors:
            raise
        return {"error": str(exc), "status": "invalid", "namespace": namespace}

    entry_id = entry_id or ("M-" + uuid4().hex[:16])
    now = datetime.now(timezone.utc)
    existing_raw = backend.get(entry_id) if entry_id is not None else None
    existing = existing_raw if isinstance(existing_raw, MemoryEntry) else None
    entry_metadata = dict(existing.metadata) if existing is not None else {}
    entry_metadata.update(metadata or {})
    if existing is None:
        entry = MemoryEntry(
            id=entry_id,
            content=content.strip(),
            detail=detail,
            tags=tags or [],
            importance=importance,
            metadata=entry_metadata,
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            source=source,
            source_identity=source_identity,
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
                "source": source,
                "source_identity": source_identity or existing.source_identity,
            }
        )

    try:
        decision = prepare_entry_for_store(entry, backend=backend, config=cfg, session_id=session_id)
        if decision.quarantined:
            store_quarantined_entry(cfg, decision.entry)
            append_audit_event(
                cfg,
                "quarantine",
                entry_id=decision.entry.id,
                actor=decision.entry.source_identity or decision.entry.source,
                namespace=namespace,
                data={
                    "stored": False,
                    "quarantined": True,
                    "anomaly_dimension": decision.anomaly_dimension,
                    "z_score": decision.anomaly_z_score,
                },
            )
            return {
                "memory_id": decision.entry.id,
                "status": "quarantined",
                "namespace": namespace,
                "stored": False,
                "quarantined": True,
                "anomaly_dimension": decision.anomaly_dimension,
                "z_score": decision.anomaly_z_score,
            }

        entry = decision.entry
        if namespace.startswith("team:") and isinstance(backend, SQLiteBackend):
            NamespaceManager(backend).ensure_team_namespace(namespace, created_at=now)
        backend.store(entry)
        embedding: list[float] | None = None
        # Mirror MemoryClient.store(): tool writes should populate vectors too,
        # otherwise tool-created memories rank differently from SDK-created ones.
        embedder = get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim)
        if embedder is not None:
            try:
                embedding = embedder.embed(f"{entry.content} {entry.detail}")
                if embedding is not None:
                    backend.upsert_vector(entry.id, embedding)
            except Exception as exc:
                try:
                    backend.delete(entry.id)
                except Exception:
                    logger.exception("memory_store_vector_rollback_failed", entry_id=entry_id)
                    raise StorageError(
                        f"failed to persist vector for {entry_id!r}; rollback did not complete cleanly"
                    ) from exc
                raise StorageError(
                    f"failed to persist vector for {entry_id!r}; entry write was rolled back"
                ) from exc
        try:
            # Graph enrichment is a secondary index over the stored entry, so we
            # dispatch it after the canonical row/vector write succeeds.
            schedule_graph_update(entry, backend, embedding=embedding, config=cfg)
        except RuntimeError:
            logger.warning("memory_store_graph_schedule_failed", entry_id=entry_id, exc_info=True)
        if supports_tier_runtime(backend):
            remember_entry_in_tiers(cfg, namespace, entry, embedding)
        append_audit_event(
            cfg,
            decision.op,
            entry_id=entry.id,
            actor=entry.source_identity or entry.source,
            namespace=namespace,
            data={
                "status": "updated" if decision.op == "update" else "stored",
                "session_id": session_id,
                "pii_types": sorted({match.pii_type for match in decision.pii_matches}),
                "quarantined": False,
            },
        )
    except SchemaValidationError as exc:
        if raise_security_errors:
            raise
        return {"error": str(exc), "status": "invalid", "namespace": namespace}
    except (PIIBlockError, PoisoningError, RateLimitError) as exc:
        if raise_security_errors:
            raise
        return {"error": str(exc), "status": "blocked", "namespace": namespace}
    except (StorageError, RuntimeError, ValueError) as exc:
        logger.exception("memory_store_failed", entry_id=entry_id, error=str(exc))
        return {"error": f"storage error: {exc}", "status": "error"}

    logger.info(
        "memory_stored",
        entry_id=entry_id,
        namespace=namespace,
        tags=tags or [],
    )

    return {
        "memory_id": entry_id,
        "status": "updated" if decision.op == "update" else "stored",
        "namespace": namespace,
    }


def register_store_tool(mcp: McpServer) -> None:
    """Register memory_store with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config
    @mcp.tool()
    async def memory_store(
        content: str,
        namespace: str = "project:default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        detail: str = "",
        metadata: dict[str, str] | None = None,
        source_identity: str = "",
        session_id: str | None = None,
        entry_id: str | None = None,
    ) -> dict[str, object]:
        """Store a new memory entry in the memory system.

        Args:
            content: Core knowledge statement to remember. Must be non-empty.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            tags: Optional list of tags for categorisation.
            importance: Importance score 0.0-1.0 (default 0.5).
            detail: Extended explanation or context.
            metadata: Optional key-value string metadata.

        Returns:
            {"memory_id": str, "status": "stored", "namespace": str}
        """
        cfg = MemoryConfig()
        with create_backend_from_config(cfg, namespace) as backend:
            return memory_store_impl(
                content,
                namespace,
                backend=backend,
                tags=tags,
                importance=importance,
                detail=detail,
                metadata=metadata,
                config=cfg,
                source_identity=source_identity,
                session_id=session_id,
                entry_id=entry_id,
                raise_security_errors=True,
            )
