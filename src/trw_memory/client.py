"""MemoryClient — Python SDK for trw-memory.

Provides a high-level async API for storing, recalling, searching, and
forgetting memories.  Supports local (SQLite/YAML) backends with a stub
for future MCP transport mode.

Usage::

    async with MemoryClient(namespace="project:my-app") as client:
        await client.store("Pydantic v2 requires strict=True", tags=["pydantic"])
        results = await client.recall("pydantic validation")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os
import socket
import threading
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

import structlog
from typing_extensions import NotRequired

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import (
    MemoryConnectionError,
    MemoryNotFoundError,
    SchemaValidationError,
    StorageError,
    ToolAlreadyRegisteredError,
)
from trw_memory.graph import list_org_shared_entries, schedule_graph_update
from trw_memory.lifecycle._recall import record_recall_access
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.lifecycle.tiers._runtime import (
    get_tier_manager,
    remember_entry_data_in_tiers,
    remember_entry_in_tiers,
    remove_entry_from_tiers,
    tier_candidates,
    tier_runtime_enabled,
    warmup_tier_manager,
)
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.retrieval.source_policy import apply_source_policy
from trw_memory.security.pii import anonymize_installation_id
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.recall_filter import filter_recall_window
from trw_memory.security.runtime import (
    append_audit_event,
    audit_entry,
    delete_quarantined_entries,
    initialize_canaries,
    list_quarantined_entries,
    prepare_entry_for_store,
    probe_canaries,
    review_quarantined_entry,
    should_halt_recalls,
    store_quarantined_entry,
)
from trw_memory.security.startup import verify_defaults
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.conflict import init_clock
from trw_memory.sync.remote import (
    _anonymize_entry,
    drain_retry_queue,
    fetch_shared_memories,
    publish_memory_result,
    retire_remote_memory,
)
from trw_memory.sync.retry_queue import RetryQueue
from trw_memory.sync.subscriber import SSESubscriber

logger = structlog.get_logger(__name__)
SHARED_EVENT_CACHE_MAX = 256

__all__ = [
    "BulkStoreItemResult",
    "BulkStoreRequest",
    "BulkStoreSummary",
    "ForgetResultDict",
    "MemoryClient",
    "MemoryResultDict",
    "StoreResultDict",
]


# Bulk-store dataclasses extracted to _client_bulk_store.py
# (PRD-DIST-246 batch 104). Re-exports preserve the public API.
from trw_memory._client_bulk_store import (
    BulkStoreItemResult as BulkStoreItemResult,
    BulkStoreRequest as BulkStoreRequest,
    BulkStoreSummary as BulkStoreSummary,
    bulk_store_impl as _bulk_store_impl,
)


# Fallback recall scoring (used when hybrid retrieval pipeline is unavailable).
# Blends term-frequency relevance with stored importance.
_FALLBACK_TF_WEIGHT: float = 0.7
_FALLBACK_IMPORTANCE_WEIGHT: float = 0.3
_FALLBACK_TF_SCALE: float = 10.0  # amplify raw TF ratio to [0, 1] range

# Re-export old names for backward compatibility (internal only).
_TF_WEIGHT = _FALLBACK_TF_WEIGHT
_IMPORTANCE_WEIGHT = _FALLBACK_IMPORTANCE_WEIGHT
_TF_SCALE = _FALLBACK_TF_SCALE


def _make_id() -> str:
    """Generate a unique memory ID with ``M-`` prefix and 16 hex characters.

    Uses 16 hex characters (64 bits of entropy) from a UUID4 to minimise
    collision probability.  At 10k entries the birthday-paradox collision
    chance is ~2.7e-12 (vs ~1.2e-5 with the previous 8-char / 32-bit scheme).

    Returns:
        A string matching ``M-[0-9a-f]{16}``, e.g. ``"M-a1b2c3d4e5f6a7b8"``.
    """
    return f"M-{uuid.uuid4().hex[:16]}"


class MemoryResultDict(TypedDict):
    """Shape of a single result dict returned by recall/search."""

    memory_id: str
    content: str
    detail: str
    tags: list[str]
    importance: float
    score: float
    created_at: str
    updated_at: str
    namespace: str
    source: str
    last_accessed_at: NotRequired[str]
    q_value: NotRequired[float]
    q_observations: NotRequired[int]
    recurrence: NotRequired[int]
    access_count: NotRequired[int]
    expires: NotRequired[str]
    metadata: NotRequired[dict[str, str]]
    anomaly_dimension: NotRequired[str]
    z_score: NotRequired[float]
    _relevance_hint: NotRequired[float]


class StoreResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.store()."""

    memory_id: str
    namespace: str
    status: str
    timestamp: str
    quarantined: NotRequired[bool]
    stored: NotRequired[bool]
    anomaly_dimension: NotRequired[str]
    z_score: NotRequired[float]


class ForgetResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.forget()."""

    memory_id: str
    status: str
    namespace: str
    entries_deleted: NotRequired[int]


@runtime_checkable
class AgentWithRegisterTool(Protocol):
    """Protocol for agent objects that expose a ``register_tool`` method."""

    def register_tool(self, name: str, fn: Callable[..., Coroutine[object, object, object]]) -> None: ...


@runtime_checkable
class AgentWithToolDecorator(Protocol):
    """Protocol for agent objects that expose a ``tool()`` decorator factory."""

    def tool(self) -> Callable[[Callable[..., Coroutine[object, object, object]]], None]: ...


# Distilled-tiering helpers + entry-to-result extracted to
# _client_distilled_tiering.py (PRD-DIST-246 batch 109). Re-exports
# preserve the public API surface (`apply_distilled_tiering` is part of
# the documented client.py exports) and the internal lookup paths used
# by `_client_recall.py` / `_client_recall_helpers.py` /
# `_client_org_shared.py`.
from trw_memory._client_distilled_tiering import (  # noqa: E402
    DEFAULT_DISTILLED_RECALL_WEIGHT as DEFAULT_DISTILLED_RECALL_WEIGHT,
    apply_distilled_tiering as apply_distilled_tiering,
    entry_to_result as _entry_to_result,
    get_distilled_recall_weight as _get_distilled_recall_weight,
    is_distilled_result as _is_distilled_result,
)


def _create_local_backend(config: MemoryConfig, namespace: str) -> StorageBackend:
    """Create a storage backend from config, isolated by namespace.

    Args:
        config: Memory configuration.
        namespace: Namespace for path isolation.

    Returns:
        Configured StorageBackend instance.
    """
    from trw_memory.integrations._backend import create_backend_from_config

    return create_backend_from_config(config, namespace)


# Type alias for the async tool functions produced by _make_tool_functions.
_ToolFn = Callable[..., Coroutine[object, object, object]]


class MemoryClient:
    """High-level async client for the trw-memory system.

    Args:
        namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
        mode: Transport mode — ``"local"`` (SQLite/YAML), ``"mcp"`` (stdio),
            or ``"auto"`` (try local first).
        timeout: Timeout in seconds for remote operations.
    """

    def __init__(
        self,
        namespace: str,
        mode: Literal["local", "mcp", "auto"] = "auto",
        timeout: float = 5.0,
    ) -> None:
        """Initialise a MemoryClient with namespace isolation and mode selection.

        Mode selection logic:

        - ``"local"`` — create a SQLite or YAML backend directly (controlled
          by ``MEMORY_STORAGE_BACKEND`` env var).  Raises
          :class:`MemoryConnectionError` if backend creation fails.
        - ``"mcp"`` — reserved for future MCP stdio transport (currently
          raises :class:`NotImplementedError`).
        - ``"auto"`` (default) — attempt ``"local"`` first; if that fails,
          raise :class:`MemoryConnectionError` (MCP fallback not yet available).

        Args:
            namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
                Must pass :func:`~trw_memory.namespaces.validation.validate_namespace`.
            mode: Transport mode — ``"local"``, ``"mcp"``, or ``"auto"``.
            timeout: Timeout in seconds for future remote operations.

        Raises:
            ValueError: If *namespace* fails validation.
            NotImplementedError: If *mode* is ``"mcp"``.
            MemoryConnectionError: If no backend can be established.
        """
        validate_namespace(namespace)
        self._namespace = namespace
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._tools_registered = False
        self._backend: StorageBackend | None = None
        self._resolved_mode: str = ""
        self._config = MemoryConfig()
        self._project_root = str(Path.cwd())
        self._installation_id = f"{socket.gethostname()}:{Path(self._config.storage_path).resolve()}"
        self._local_node_id = anonymize_installation_id(self._installation_id)
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._retry_queue = RetryQueue(Path(self._config.storage_path) / "sync_queue.jsonl")
        self._retry_drain_started = False
        self._shared_event_cache: list[MemoryResultDict] = []
        self._shared_event_cache_lock = threading.Lock()
        self._pending_remote_retirements: set[str] = set()
        self._pending_remote_retirements_lock = threading.Lock()
        self._sse_subscriber: SSESubscriber | None = None
        self._sse_subscriber_started = False
        self._tier_manager = None

        if mode == "mcp":
            raise NotImplementedError("MCP mode is not yet implemented")

        if mode in ("local", "auto"):
            try:
                self._backend = _create_local_backend(self._config, namespace)
                verify_defaults(self._config)
                initialize_canaries(self._config, backend=self._backend)
                self._resolved_mode = "local"
                logger.debug(
                    "client_initialized",
                    op="init",
                    namespace=namespace,
                    mode=self._resolved_mode,
                    backend=self._config.storage_backend,
                )
                if tier_runtime_enabled(self._config):
                    self._tier_manager = warmup_tier_manager(self._config, namespace, self._backend)
            except (OSError, ValueError, ImportError) as exc:
                if mode == "local":
                    raise MemoryConnectionError(f"Failed to create local backend: {exc}") from exc

        if mode == "auto" and self._backend is None:
            raise MemoryConnectionError("No connection mode available. Tried: local.")

        self._maybe_start_sse_subscription()

    def __repr__(self) -> str:
        return f"MemoryClient(namespace={self._namespace!r}, mode={self._resolved_mode!r})"

    @property
    def resolved_mode(self) -> str:
        """The actual transport mode in use."""
        return self._resolved_mode

    @property
    def namespace(self) -> str:
        """The namespace this client operates in."""
        return self._namespace

    def _get_backend(self) -> StorageBackend:
        """Return the active backend, raising if the client is closed."""
        if self._backend is None:
            raise MemoryConnectionError("Client is closed or has no backend")
        return self._backend

    def _require_permission(self, permission: Permission, operation: str) -> None:
        """Enforce the configured RBAC policy for a client operation."""
        require_namespace_permission(self._config, self._namespace, permission, operation)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def store(
        self,
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
        """Store a new memory entry.

        Args:
            content: Core knowledge statement (must not be empty).
            tags: Categorisation tags.
            importance: Importance score in ``[0.0, 1.0]``.
            detail: Extended explanation.
            metadata: Arbitrary key-value pairs.

        Returns:
            Dict with ``memory_id``, ``namespace``, ``status``, ``timestamp``.

        Raises:
            ValueError: If *content* is empty or *importance* out of range.
        """
        try:
            validate_store_inputs(content=content, detail=detail, tags=tags, metadata=metadata, importance=importance)
        except SchemaValidationError as exc:
            append_audit_event(
                self._config,
                "store_rejected",
                entry_id=entry_id or "",
                actor=source_identity or source,
                namespace=self._namespace,
                data={"reason": "schema_invalid", "failed_fields": exc.failed_fields, "session_id": session_id},
            )
            raise
        self._require_permission(Permission.WRITE, "store")
        self._maybe_start_retry_drain()

        memory_id = entry_id or _make_id()
        async with self._lock:
            backend = self._get_backend()
            existing = backend.get(memory_id) if entry_id is not None else None
            now = datetime.now(timezone.utc)
            entry_metadata = dict(existing.metadata) if existing is not None else {}
            entry_metadata.update(metadata or {})
            entry_metadata.setdefault("installation_id", self._installation_id)
            entry_expires = expires or (existing.expires if existing is not None else "")

            if existing is None:
                entry = MemoryEntry(
                    id=memory_id,
                    content=content.strip(),
                    detail=detail,
                    tags=tags or [],
                    importance=importance,
                    namespace=self._namespace,
                    metadata=entry_metadata,
                    created_at=now,
                    updated_at=now,
                    expires=entry_expires,
                    source=source,
                    source_identity=source_identity,
                    # We need a stable local node marker even before the backend grows
                    # first-class sync metadata, otherwise concurrent local edits have no
                    # causal anchor at all.
                    vector_clock=init_clock(self._local_node_id),
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
                        "vector_clock": init_clock(self._local_node_id),
                    }
                )

            decision = prepare_entry_for_store(
                entry,
                backend=backend,
                config=self._config,
                session_id=session_id,
            )
            if decision.quarantined:
                store_quarantined_entry(self._config, decision.entry)
                append_audit_event(
                    self._config,
                    "quarantine",
                    entry_id=decision.entry.id,
                    actor=decision.entry.source_identity or decision.entry.source,
                    namespace=self._namespace,
                    data={
                        "stored": False,
                        "quarantined": True,
                        "anomaly_dimension": decision.anomaly_dimension,
                        "z_score": decision.anomaly_z_score,
                    },
                )
                quarantined_result: StoreResultDict = {
                    "memory_id": decision.entry.id,
                    "namespace": self._namespace,
                    "status": "quarantined",
                    "timestamp": now.isoformat(),
                    "quarantined": True,
                    "stored": False,
                    "anomaly_dimension": decision.anomaly_dimension,
                    "z_score": decision.anomaly_z_score,
                }
                return quarantined_result

            entry = decision.entry
            # Persist the dense vector on the normal write path so later recall can
            # actually use hybrid ranking instead of silently degrading to BM25-only.
            embedder = self._get_embedder()
            embedding = embedder.embed(f"{entry.content} {entry.detail}") if embedder is not None else None
            if self._namespace.startswith("team:"):
                NamespaceManager(backend).ensure_team_namespace(self._namespace, created_at=now)
            backend.store(entry)
            if embedding is not None:
                try:
                    backend.upsert_vector(entry.id, embedding)
                except Exception as exc:
                    # Keep store() atomic: callers should never see a failed
                    # write while the primary row is still committed.
                    try:
                        backend.delete(entry.id)
                    except Exception:
                        logger.exception("memory_store_vector_rollback_failed", memory_id=entry.id)
                        raise StorageError(
                            f"failed to persist vector for {entry.id!r}; rollback did not complete cleanly"
                        ) from exc
                    raise StorageError(
                        f"failed to persist vector for {entry.id!r}; entry write was rolled back"
                    ) from exc
            try:
                # Graph enrichment must still run without embeddings so tag and
                # lineage edges stay consistent, but it should not hold up store().
                schedule_graph_update(entry, backend, embedding=embedding, config=self._config)
            except RuntimeError:
                logger.warning("memory_store_graph_schedule_failed", memory_id=entry.id, exc_info=True)
            remember_entry_in_tiers(self._config, self._namespace, entry, embedding)
            append_audit_event(
                self._config,
                decision.op,
                entry_id=entry.id,
                actor=entry.source_identity or entry.source,
                namespace=self._namespace,
                data={
                    "status": "updated" if decision.op == "update" else "stored",
                    "session_id": session_id,
                    "pii_types": sorted({match.pii_type for match in decision.pii_matches}),
                    "quarantined": False,
                },
            )

        logger.debug(
            "memory_stored",
            op="store",
            outcome="success",
            memory_id=memory_id,
            namespace=self._namespace,
            content_len=len(content),
            tag_count=len(tags or []),
            importance=importance,
        )
        if self._should_attempt_remote_publish(entry):
            self._schedule_background_task(self._publish_entry(entry, embedding))
        store_result: StoreResultDict = {
            "memory_id": memory_id,
            "namespace": self._namespace,
            "status": "updated" if decision.op == "update" else "stored",
            "timestamp": now.isoformat(),
        }
        return store_result

    async def bulk_store(
        self,
        requests: list[BulkStoreRequest],
        *,
        skip_audit_per_item: bool = True,
        skip_remote_publish: bool = True,
    ) -> BulkStoreSummary:
        """Store many records in one batched operation.

        Implementation lives in ``_client_bulk_store.bulk_store_impl``
        (PRD-DIST-246 batch 104). See that helper's docstring for full
        arg/return semantics. Trades per-item audit + remote-publish
        overhead for throughput; per-item security checks (PII /
        poisoning) still run on every record.
        """
        return await _bulk_store_impl(
            self,
            requests,
            skip_audit_per_item=skip_audit_per_item,
            skip_remote_publish=skip_remote_publish,
        )

    async def recall(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        *,
        include_org_memories: bool = True,
        include_shared: bool = False,
        token_budget: int | None = None,
        include_distilled: bool = True,
        distilled_weight: float | None = None,
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
        exclude_expired: bool = True,
    ) -> list[MemoryResultDict]:
        """Search memories by keyword query using hybrid retrieval.

        Two-tier strategy: hybrid pipeline (BM25 + dense + RRF) preferred;
        falls back to LIKE + TF + importance scoring when the hybrid
        pipeline is unavailable. Both paths apply tag filtering, min_score
        thresholds, limit capping, and optional token-budget fitting.
        Implementation lives in ``_client_recall.recall_impl`` (PRD-DIST-246
        batch 105).
        """
        from trw_memory._client_recall import recall_impl as _recall_impl

        return await _recall_impl(
            self,
            query,
            limit=limit,
            tags=tags,
            min_score=min_score,
            include_org_memories=include_org_memories,
            include_shared=include_shared,
            token_budget=token_budget,
            include_distilled=include_distilled,
            distilled_weight=distilled_weight,
            include_source_kinds=include_source_kinds,
            exclude_source_kinds=exclude_source_kinds,
            source_weights=source_weights,
            exclude_expired=exclude_expired,
        )

    def _get_embedder(self) -> EmbeddingProvider | None:
        """Try to obtain a local embedding provider; return None on failure.

        This is a best-effort helper: when sentence-transformers is not
        installed or the provider reports itself as unavailable, dense
        retrieval is silently skipped and the hybrid pipeline degrades
        to BM25-only mode.
        """
        from trw_memory.embeddings import get_local_embedder

        return get_local_embedder(
            model_name=self._config.embedding_model,
            dim=self._config.embedding_dim,
        )

    # ---- Recall helper aliases (PRD-DIST-246 batch 105) -------------------
    # Re-export thin wrappers so existing test patches on
    # `trw_memory.client.MemoryClient._<helper>` keep working after the
    # implementation moved to ``_client_recall.py``. Tests monkeypatch via
    # ``setattr(client, "_merge_tier_results", ...)`` and direct
    # ``MemoryClient._<helper>(...)`` calls.

    def _apply_recall_security(self, results: list[MemoryResultDict]) -> list[MemoryResultDict]:
        from trw_memory._client_recall import apply_recall_security as _impl

        return _impl(self, results)

    @staticmethod
    def _apply_budget(
        results: list[MemoryResultDict],
        token_budget: int | None,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import apply_budget as _impl

        return _impl(results, token_budget)

    async def _merge_org_results(
        self,
        query: str,
        local_results: list[MemoryResultDict],
        limit: int,
        tags: list[str] | None,
        min_score: float,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import merge_org_results as _impl

        return await _impl(self, query, local_results, limit, tags, min_score)

    async def _try_hybrid_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
    ) -> list[MemoryResultDict] | None:
        from trw_memory._client_recall import try_hybrid_recall as _impl

        return await _impl(self, query, limit, tags)

    async def _fallback_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
        min_score: float,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import fallback_recall as _impl

        return await _impl(self, query, limit, tags, min_score)

    async def _record_recall_access(self, results: list[MemoryResultDict]) -> None:
        from trw_memory._client_recall import record_recall_access_impl as _impl

        await _impl(self, results)

    def _tier_results(
        self,
        backend: StorageBackend,
        query: str,
        tags: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import tier_results as _impl

        return _impl(self, backend, query, tags, limit, query_embedding)

    def _remember_results_in_tiers(self, results: list[MemoryResultDict]) -> None:
        from trw_memory._client_recall import remember_results_in_tiers as _impl

        _impl(self, results)

    @staticmethod
    def _merge_tier_results(
        local_results: list[MemoryResultDict],
        tier_only_results: list[MemoryResultDict],
        limit: int,
        query_tokens: list[str],
        config: MemoryConfig,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import merge_tier_results as _impl

        return _impl(local_results, tier_only_results, limit, query_tokens, config, query_embedding)

    @staticmethod
    def _tier_result_from_entry(entry: dict[str, object]) -> MemoryResultDict:
        from trw_memory._client_recall import tier_result_from_entry as _impl

        return _impl(entry)

    def _should_attempt_remote_publish(self, entry: MemoryEntry) -> bool:
        """Return whether this entry should attempt the remote publish path."""
        return (
            not self._config.local_only
            and self._config.sync_enabled
            and bool(self._config.platform_url)
            and entry.importance >= self._config.sync_min_importance
        )

    def _schedule_background_task(self, coro: Coroutine[object, object, None]) -> None:
        """Track a background task so shutdown can await sync side effects safely."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _publish_entry(self, entry: MemoryEntry, embedding: list[float] | None) -> None:
        """Best-effort remote publish for a freshly stored entry."""
        publish_result = await asyncio.to_thread(
            functools.partial(
                publish_memory_result,
                entry,
                self._config,
                embedding=embedding,
                project_root=self._project_root,
            )
        )
        if publish_result["success"]:
            async with self._lock:
                backend = self._get_backend()
                backend.update(
                    entry.id,
                    published_to_platform=True,
                    remote_id=publish_result["remote_id"],
                    last_synced_at=datetime.now(timezone.utc),
                )
            return

        retryable = publish_result.get("retryable", not publish_result["success"])
        if not retryable:
            return

        payload = await asyncio.to_thread(_anonymize_entry, entry, self._project_root)
        if embedding is not None:
            payload["embedding"] = embedding
        queue_payload = cast("dict[str, object]", payload)
        enqueued = await asyncio.to_thread(self._retry_queue.enqueue, entry.id, queue_payload)
        if not enqueued:
            logger.warning(
                "memory_sync_queue_full",
                op="store",
                outcome="failure",
                memory_id=entry.id,
                namespace=self._namespace,
            )

    # ---- Org-shared helper aliases (PRD-DIST-246 batch 107) ---------------
    # Implementations live in `_client_org_shared.py`; thin wrappers here
    # preserve `monkeypatch.setattr(client, "_X", ...)` test patches and
    # `MemoryClient._coerce_float(...)` static-call sites.

    async def _merge_shared_results(
        self,
        query: str,
        local_results: list[MemoryResultDict],
        limit: int,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import merge_shared_results as _impl
        return await _impl(self, query, local_results, limit)

    async def _load_entries_for_results(self, results: list[MemoryResultDict]) -> list[MemoryEntry]:
        from trw_memory._client_org_shared import load_entries_for_results as _impl
        return await _impl(self, results)

    @staticmethod
    def _shared_result_to_result(result: dict[str, object]) -> MemoryResultDict:
        from trw_memory._client_org_shared import shared_result_to_result as _impl
        return _impl(result)

    @staticmethod
    def _coerce_float(value: object) -> float:
        from trw_memory._client_org_shared import coerce_float as _impl
        return _impl(value)

    @staticmethod
    def _is_retired_shared_result(result: dict[str, object]) -> bool:
        from trw_memory._client_org_shared import is_retired_shared_result as _impl
        return _impl(result)

    def _merge_shared_candidates(
        self,
        local_results: list[MemoryResultDict],
        shared_results: list[MemoryResultDict],
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import merge_shared_candidates as _impl
        return _impl(local_results, shared_results)

    def _snapshot_cached_shared_results(self, query: str) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import snapshot_cached_shared_results as _impl
        return _impl(self, query)

    @staticmethod
    def _matches_query(result: MemoryResultDict, query: str) -> bool:
        from trw_memory._client_org_shared import matches_query as _impl
        return _impl(result, query)

    async def _dedupe_cached_shared_results(
        self,
        cached_results: list[MemoryResultDict],
        *,
        local_entries: list[MemoryEntry],
        embedder: EmbeddingProvider | None,
        dedup_threshold: float = 0.92,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import dedupe_cached_shared_results as _impl
        return await _impl(
            self,
            cached_results,
            local_entries=local_entries,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
        )

    @staticmethod
    def _strip_shared_prefix(content: str) -> str:
        from trw_memory._client_org_shared import strip_shared_prefix as _impl
        return _impl(content)

    async def _mark_fetch_retirements(self, shared_results: list[dict[str, object]]) -> None:
        from trw_memory._client_org_shared import mark_fetch_retirements as _impl
        await _impl(self, shared_results)

    async def forget(self, memory_id: str | None = None, *, actor: str | None = None) -> ForgetResultDict:
        """Delete a memory entry.

        Implementation lives in ``_client_forget_search.forget_impl``
        (PRD-DIST-246 batch 106). Supports memory_id-targeted delete and
        actor-scoped GDPR bulk erasure; quarantine-aware.
        """
        from trw_memory._client_forget_search import forget_impl as _impl

        return await _impl(self, memory_id, actor=actor)

    async def search(
        self,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        since: datetime | None = None,
        limit: int = 50,
        *,
        actor: str | None = None,
        status: str | None = None,
    ) -> list[MemoryResultDict]:
        """Filtered search (tags + min_importance + since + status filter).

        Implementation lives in ``_client_forget_search.search_impl``
        (PRD-DIST-246 batch 106).
        """
        from trw_memory._client_forget_search import search_impl as _impl

        return await _impl(
            self,
            tags=tags,
            min_importance=min_importance,
            since=since,
            limit=limit,
            actor=actor,
            status=status,
        )

    async def audit_learning(self, learning_id: str) -> dict[str, object]:
        """Return SEC-001 audit data for an active or quarantined learning."""
        return audit_entry(self._config, learning_id=learning_id, active_backend=self._get_backend())

    async def review_quarantined(
        self,
        learning_id: str,
        *,
        decision: Literal["approve", "reject"],
        reviewer_id: str,
    ) -> dict[str, str]:
        """Review a quarantined learning and either promote or reject it."""
        return review_quarantined_entry(
            self._config,
            active_backend=self._get_backend(),
            learning_id=learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
        )

    # ------------------------------------------------------------------
    # Tool registration (FR09)
    # ------------------------------------------------------------------

    # ---- Tools-binding aliases (PRD-DIST-246 batch 108) -------------------
    # Implementations live in `_client_tools_binding.py`.

    def _make_tool_functions(self) -> dict[str, _ToolFn]:
        from trw_memory._client_tools_binding import make_tool_functions as _impl
        return _impl(self)

    def register_tools(self, agent: AgentWithRegisterTool | AgentWithToolDecorator) -> None:
        from trw_memory._client_tools_binding import register_tools as _impl
        _impl(self, agent)

    def auto_recall(
        self,
        query_from: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Callable[[Callable[..., Coroutine[object, object, object]]], Callable[..., Coroutine[object, object, object]]]:
        from trw_memory._client_tools_binding import auto_recall as _impl
        return _impl(self, query_from, limit=limit, min_score=min_score)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MemoryClient:
        """Enter the async context manager."""
        self._maybe_start_retry_drain()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the async context manager, closing the backend."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying backend and release resources."""
        if self._sse_subscriber is not None:
            self._sse_subscriber.stop()
            self._sse_subscriber = None
            self._sse_subscriber_started = False
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        if self._backend is not None:
            self._backend.close()
            self._backend = None
            logger.debug("client_closed", op="close", namespace=self._namespace)

    def _should_start_retry_drain(self) -> bool:
        """Return whether entering a client session should drain queued publishes."""
        return (
            not self._retry_drain_started
            and not self._config.local_only
            and self._config.sync_enabled
            and bool(self._config.platform_url)
            and self._retry_queue.depth() > 0
        )

    def _should_start_sse_subscription(self) -> bool:
        """Return whether this client should own a live shared-learning subscription."""
        return (
            not self._sse_subscriber_started
            and not self._config.local_only
            and self._config.sync_enabled
            and bool(self._config.platform_url)
            and bool(self._config.platform_api_key)
        )

    def _maybe_start_sse_subscription(self) -> None:
        """Start the SSE subscriber once per client when remote sync is enabled."""
        if not self._should_start_sse_subscription():
            return
        subscriber = SSESubscriber(self._config, on_event=self._handle_sse_event)
        subscriber.start()
        self._sse_subscriber = subscriber
        self._sse_subscriber_started = True

    def _maybe_start_retry_drain(self) -> None:
        """Start queue recovery once the client is actively used in a live loop."""
        if self._should_start_retry_drain():
            self._retry_drain_started = True
            self._schedule_background_task(self._drain_retry_queue())

    def _handle_sse_event(self, event: dict[str, object]) -> None:
        """Merge SSE learning events into the next shared recall path."""
        event_type = str(event.get("type", ""))
        if event_type in {"learning_published", "learning_updated"}:
            self._cache_shared_event(event)
            return
        if event_type == "learning_retired":
            remote_id = str(event.get("id", ""))
            if not remote_id:
                return
            with self._pending_remote_retirements_lock:
                self._pending_remote_retirements.add(remote_id)
            with self._shared_event_cache_lock:
                self._shared_event_cache = [
                    cached for cached in self._shared_event_cache if cached["memory_id"] != remote_id
                ]

    def _cache_shared_event(self, event: dict[str, object]) -> None:
        """Store a lightweight shared result so the next recall can surface it."""
        remote_id = str(event.get("id", "")).strip()
        summary = str(event.get("summary", "")).strip()
        if not remote_id or not summary:
            return
        shared_content = summary if summary.startswith("[shared] ") else f"[shared] {summary}"
        cached: MemoryResultDict = {
            "memory_id": remote_id,
            "content": shared_content,
            "detail": "",
            "tags": [],
            "importance": 0.0,
            "score": 0.0,
            "created_at": "",
            "updated_at": "",
            "namespace": "shared",
            "source": "shared",
        }
        with self._shared_event_cache_lock:
            self._shared_event_cache = [
                existing for existing in self._shared_event_cache if existing["memory_id"] != remote_id
            ]
            self._shared_event_cache.append(cached)
            if len(self._shared_event_cache) > SHARED_EVENT_CACHE_MAX:
                # Keep the most recent shared events so a burst cannot grow memory usage without bound.
                self._shared_event_cache = self._shared_event_cache[-SHARED_EVENT_CACHE_MAX:]

    async def _drain_retry_queue(self) -> None:
        """Best-effort background drain for queued publish payloads."""
        queued_before = {record["entry_id"] for record in self._retry_queue.snapshot()}
        result = await asyncio.to_thread(drain_retry_queue, self._retry_queue, self._config)
        queued_after = {record["entry_id"] for record in self._retry_queue.snapshot()}
        drained_ids = queued_before - queued_after
        if drained_ids:
            async with self._lock:
                backend = self._get_backend()
                synced_at = datetime.now(timezone.utc)
                for entry_id in drained_ids:
                    remote_id = result["remote_ids"].get(entry_id)
                    if remote_id is not None:
                        backend.update(
                            entry_id,
                            published_to_platform=True,
                            remote_id=remote_id,
                            last_synced_at=synced_at,
                        )
                    else:
                        backend.update(
                            entry_id,
                            published_to_platform=True,
                            last_synced_at=synced_at,
                        )
        logger.debug(
            "memory_sync_queue_drained",
            op="session_start",
            outcome="success",
            namespace=self._namespace,
            drained=result["drained"],
            failed=result["failed"],
            skipped=result["skipped"],
        )

    async def _retire_remote_entry(self, memory_id: str, remote_id: str) -> None:
        """Best-effort remote retirement for a locally deleted published entry."""
        retired = await asyncio.to_thread(retire_remote_memory, remote_id, self._config)
        if retired:
            logger.debug(
                "memory_remote_retired",
                op="forget",
                outcome="success",
                memory_id=memory_id,
                remote_id=remote_id,
                namespace=self._namespace,
            )
            return
        logger.warning(
            "memory_remote_retire_failed",
            op="forget",
            outcome="failure",
            memory_id=memory_id,
            remote_id=remote_id,
            namespace=self._namespace,
        )

    async def _apply_pending_remote_retirements(self) -> None:
        """Mark matching local entries as pending_delete when remote retirement wins."""
        with self._pending_remote_retirements_lock:
            remote_ids = set(self._pending_remote_retirements)
            self._pending_remote_retirements.clear()
        if not remote_ids:
            return

        async with self._lock:
            backend = self._get_backend()
            limit = max(backend.count(namespace=self._namespace), 1)
            entries = backend.list_entries(namespace=self._namespace, limit=limit)
            unresolved = set(remote_ids)
            for entry in entries:
                if entry.remote_id not in unresolved:
                    continue
                # Package-local sync does not yet increment vector clocks on every
                # mutation, so last_synced_at is the only trustworthy shipped
                # signal that the local row has previously converged with remote.
                if entry.last_synced_at is None:
                    continue
                updated = backend.update(
                    entry.id,
                    pending_delete=True,
                )
                if updated is not None:
                    unresolved.discard(str(entry.remote_id))
        if unresolved:
            with self._pending_remote_retirements_lock:
                self._pending_remote_retirements.update(unresolved)
