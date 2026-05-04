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


# --- PRD-DIST-005 FR-6: recall-side tiering for distilled records ---

# Default dampening factor for records with source="distilled:*" or
# tags prefixed "distill:". On new-repo installs the distilled tier may
# flood memory; dampening ensures curated learnings retain priority while
# distilled records still contribute.
#
# Overridable via TRW_MEMORY_DISTILLED_RECALL_WEIGHT env var (float).
# Set to 1.0 to disable dampening; per-query opt-out via the
# `include_distilled` kwarg on MemoryClient.recall.
DEFAULT_DISTILLED_RECALL_WEIGHT: float = 0.75


def _get_distilled_recall_weight() -> float:
    """Resolve the dampening weight from env or fall back to default.

    Invalid env values log a warning + use the default.
    """
    raw = os.environ.get("TRW_MEMORY_DISTILLED_RECALL_WEIGHT")
    if not raw:
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    try:
        weight = float(raw)
    except ValueError:
        logger.warning(
            "distilled_recall_weight_invalid",
            raw=raw,
            default=DEFAULT_DISTILLED_RECALL_WEIGHT,
        )
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    if not 0.0 <= weight <= 1.0:
        logger.warning(
            "distilled_recall_weight_out_of_range",
            raw=weight,
            default=DEFAULT_DISTILLED_RECALL_WEIGHT,
        )
        return DEFAULT_DISTILLED_RECALL_WEIGHT
    return weight


def _is_distilled_result(result: MemoryResultDict) -> bool:
    """True if the record was written by trw-distill.

    Recognizes distilled records via two complementary markers:
      - any tag starts with ``distill:`` (e.g. ``distill:decision``,
        ``distill:rationale``)
      - metadata.source starts with ``distilled:`` (e.g.
        ``distilled:git:<sha>..<sha>``)

    Both markers are set by trw-distill's ingestion writer so recall
    can match regardless of which one survives a serialization path.
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
    When ``include_distilled=True`` and ``weight < 1.0``, distilled record
    scores are multiplied by ``weight`` and the list is re-sorted. At
    ``weight=1.0`` (or env-override), this is a no-op passthrough.

    The input list is not mutated; a new sorted list is returned.
    """
    if not include_distilled:
        return [r for r in results if not _is_distilled_result(r)]

    effective_weight = weight if weight is not None else _get_distilled_recall_weight()
    if effective_weight >= 1.0 - 1e-9:
        return list(results)

    dampened: list[MemoryResultDict] = []
    for r in results:
        if _is_distilled_result(r):
            # Create a shallow copy so we don't mutate the caller's input.
            new_r = dict(r)
            new_r["score"] = float(r.get("score", 0.0)) * effective_weight
            dampened.append(new_r)  # type: ignore[arg-type]
        else:
            dampened.append(r)
    dampened.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return dampened


# --- end FR-6 helpers ---


def _entry_to_result(entry: MemoryEntry, score: float = 0.0) -> MemoryResultDict:
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

    async def _merge_shared_results(
        self,
        query: str,
        local_results: list[MemoryResultDict],
        limit: int,
    ) -> list[MemoryResultDict]:
        """Fetch shared memories and append them after local results."""
        try:
            await self._apply_pending_remote_retirements()
            local_entries = await self._load_entries_for_results(local_results)
            embedder = self._get_embedder()
            cached_shared = await self._dedupe_cached_shared_results(
                self._snapshot_cached_shared_results(query),
                local_entries=local_entries,
                embedder=embedder,
            )
            query_embedding: list[float] | None = None
            if embedder is not None and query.strip():
                query_embedding = await asyncio.to_thread(embedder.embed, query)

            shared = await asyncio.to_thread(
                functools.partial(
                    fetch_shared_memories,
                    query,
                    self._config,
                    embedding=query_embedding,
                    limit=limit,
                    local_entries=local_entries,
                    embedder=embedder,
                )
            )
        except Exception:
            # Shared recall must never block or break local recall; the local
            # path is the contract, and remote enrichment is opportunistic.
            logger.debug(
                "memory_shared_recall_failed",
                op="recall",
                outcome="failure",
                namespace=self._namespace,
                exc_info=True,
            )
            return self._merge_shared_candidates(local_results, self._snapshot_cached_shared_results(query))
        await self._mark_fetch_retirements(shared)
        live_shared = [
            self._shared_result_to_result(item) for item in shared if not self._is_retired_shared_result(item)
        ]
        return self._merge_shared_candidates(local_results, [*live_shared, *cached_shared])

    async def _load_entries_for_results(self, results: list[MemoryResultDict]) -> list[MemoryEntry]:
        """Materialize local entries for dedup against shared results."""
        result_ids = [result["memory_id"] for result in results if result.get("source", "local") == "local"]
        if not result_ids:
            return []

        async with self._lock:
            backend = self._get_backend()
            loaded: list[MemoryEntry] = []
            for entry_id in result_ids:
                entry = backend.get(entry_id)
                if entry is not None:
                    loaded.append(entry)
            return loaded

    @staticmethod
    def _shared_result_to_result(result: dict[str, object]) -> MemoryResultDict:
        """Normalize a shared remote result into the client result shape."""
        memory_id = str(result.get("memory_id", result.get("id", result.get("remote_id", ""))))
        detail = str(result.get("detail", ""))
        raw_tags = result.get("tags", [])
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        importance_raw = result.get("importance", result.get("impact", 0.0))
        score_raw = result.get("score", importance_raw)
        namespace = str(result.get("namespace", "shared"))
        created_at = str(result.get("created_at", ""))
        updated_at = str(result.get("updated_at", created_at))
        source = str(result.get("source", "shared"))
        shared_result: MemoryResultDict = {
            "memory_id": memory_id,
            "content": str(result.get("content", "")),
            "detail": detail,
            "tags": tags,
            "importance": MemoryClient._coerce_float(importance_raw),
            "score": MemoryClient._coerce_float(score_raw),
            "created_at": created_at,
            "updated_at": updated_at,
            "namespace": namespace,
            "source": source,
        }
        return shared_result

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Convert loosely typed payload values into floats with a safe default."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _is_retired_shared_result(result: dict[str, object]) -> bool:
        """Return whether a shared result represents a remote retirement."""
        status = str(result.get("status", "")).lower()
        return status in {"obsolete", "deleted"}

    def _merge_shared_candidates(
        self,
        local_results: list[MemoryResultDict],
        shared_results: list[MemoryResultDict],
    ) -> list[MemoryResultDict]:
        """Append shared results after local ones while suppressing exact duplicates."""
        seen_ids = {result["memory_id"] for result in local_results}
        seen_content = {result["content"] for result in local_results}
        merged = list(local_results)
        for result in shared_results:
            if result["memory_id"] in seen_ids or result["content"] in seen_content:
                continue
            merged.append(result)
            seen_ids.add(result["memory_id"])
            seen_content.add(result["content"])
        return merged

    def _snapshot_cached_shared_results(self, query: str) -> list[MemoryResultDict]:
        """Return cached SSE shared results relevant to the current query."""
        with self._shared_event_cache_lock:
            cached = list(self._shared_event_cache)
        if not query.strip():
            return cached
        return [result for result in cached if self._matches_query(result, query)]

    @staticmethod
    def _matches_query(result: MemoryResultDict, query: str) -> bool:
        """Apply the same simple token matching used by fallback recall."""
        query_terms = {term for term in query.lower().split() if term}
        if not query_terms:
            return True
        text = f"{result['content']} {result['detail']} {' '.join(result['tags'])}".lower()
        return any(term in text for term in query_terms)

    async def _dedupe_cached_shared_results(
        self,
        cached_results: list[MemoryResultDict],
        *,
        local_entries: list[MemoryEntry],
        embedder: EmbeddingProvider | None,
        dedup_threshold: float = 0.92,
    ) -> list[MemoryResultDict]:
        """Apply the same exact/semantic dedup rules to cached SSE results."""
        if not cached_results or not local_entries:
            return cached_results

        local_remote_ids = {str(entry.remote_id) for entry in local_entries if entry.remote_id}
        local_contents = {entry.content.lower().strip() for entry in local_entries}

        candidates: list[MemoryResultDict] = []
        candidate_texts: list[str] = []
        for result in cached_results:
            normalized_content = self._strip_shared_prefix(result["content"]).strip()
            if result["memory_id"] in local_remote_ids or normalized_content.lower() in local_contents:
                continue
            candidates.append(result)
            candidate_texts.append(f"{normalized_content} {result['detail']}".strip())

        if not candidates or embedder is None or not embedder.available():
            return candidates

        local_texts = [f"{entry.content} {entry.detail}".strip() for entry in local_entries]
        vectors = await asyncio.to_thread(embedder.embed_batch, [*local_texts, *candidate_texts])
        local_vectors = [vector for vector in vectors[: len(local_entries)] if vector is not None]
        remote_vectors = vectors[len(local_entries) :]
        if not local_vectors:
            return candidates

        deduped: list[MemoryResultDict] = []
        for candidate, remote_vector in zip(candidates, remote_vectors, strict=False):
            if remote_vector is None:
                deduped.append(candidate)
                continue
            if any(cosine_similarity(remote_vector, local_vector) > dedup_threshold for local_vector in local_vectors):
                continue
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _strip_shared_prefix(content: str) -> str:
        """Normalize cached shared content for dedup comparisons."""
        return content.removeprefix("[shared] ")

    async def _mark_fetch_retirements(self, shared_results: list[dict[str, object]]) -> None:
        """Record retirement markers returned from remote fetches."""
        remote_ids = {
            str(result.get("id", result.get("remote_id", "")))
            for result in shared_results
            if self._is_retired_shared_result(result)
        }
        if not remote_ids:
            return
        with self._pending_remote_retirements_lock:
            self._pending_remote_retirements.update(remote_id for remote_id in remote_ids if remote_id)
        await self._apply_pending_remote_retirements()

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

    def _make_tool_functions(self) -> dict[str, _ToolFn]:
        """Create the shared tool functions for agent registration."""
        client = self

        async def memory_store(
            content: str,
            tags: list[str] | None = None,
            importance: float = 0.5,
        ) -> StoreResultDict:
            return await client.store(content, tags=tags, importance=importance)

        async def memory_recall(
            query: str,
            limit: int = 10,
            include_org_memories: bool = True,
            include_shared: bool = False,
            include_distilled: bool = True,
            distilled_weight: float | None = None,
            include_source_kinds: list[str] | None = None,
            exclude_source_kinds: list[str] | None = None,
            source_weights: dict[str, float] | None = None,
            exclude_expired: bool = True,
        ) -> list[MemoryResultDict]:
            return await client.recall(
                query,
                limit=limit,
                include_org_memories=include_org_memories,
                include_shared=include_shared,
                include_distilled=include_distilled,
                distilled_weight=distilled_weight,
                include_source_kinds=include_source_kinds,
                exclude_source_kinds=exclude_source_kinds,
                source_weights=source_weights,
                exclude_expired=exclude_expired,
            )

        async def memory_forget(memory_id: str | None = None, actor: str | None = None) -> ForgetResultDict:
            return await client.forget(memory_id, actor=actor)

        async def memory_search(
            tags: list[str] | None = None,
            min_importance: float = 0.0,
            limit: int = 50,
            actor: str | None = None,
            status: str | None = None,
        ) -> list[MemoryResultDict]:
            return await client.search(
                tags=tags,
                min_importance=min_importance,
                limit=limit,
                actor=actor,
                status=status,
            )

        return {
            "memory_store": memory_store,
            "memory_recall": memory_recall,
            "memory_forget": memory_forget,
            "memory_search": memory_search,
        }

    def register_tools(self, agent: AgentWithRegisterTool | AgentWithToolDecorator) -> None:
        """Register memory tools with an agent framework.

        Attempts to register ``memory_store``, ``memory_recall``,
        ``memory_forget``, and ``memory_search`` tools on the agent object.

        Args:
            agent: An agent-like object with a ``tool()`` decorator method
                or ``register_tool()`` method.

        Raises:
            ToolAlreadyRegisteredError: If called more than once.
            TypeError: If *agent* does not have a compatible registration API.
        """
        if self._tools_registered:
            raise ToolAlreadyRegisteredError("register_tools() has already been called on this client")

        tools = self._make_tool_functions()

        # Detect registration API
        register_fn = getattr(agent, "register_tool", None)
        tool_decorator = getattr(agent, "tool", None)

        if register_fn is not None and callable(register_fn):
            for name, fn in tools.items():
                register_fn(name, fn)
        elif tool_decorator is not None and callable(tool_decorator):
            dec = tool_decorator()
            for fn in tools.values():
                dec(fn)
        else:
            raise TypeError("Agent must have a 'register_tool()' method or 'tool()' decorator")

        self._tools_registered = True

    # ------------------------------------------------------------------
    # auto_recall decorator (FR06)
    # ------------------------------------------------------------------

    def auto_recall(
        self,
        query_from: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Callable[[Callable[..., Coroutine[object, object, object]]], Callable[..., Coroutine[object, object, object]]]:
        """Decorator that injects recalled memories into a function.

        Before calling the decorated function, extracts a query string from
        the function's keyword argument named *query_from*, performs a recall,
        and injects the results as a ``recalled_memories`` keyword argument.

        Fail-open: if the backend is unreachable or the query_from key is
        absent, an empty list is injected.

        Args:
            query_from: Name of the kwarg to extract the query from.
            limit: Maximum number of recalled entries.
            min_score: Minimum score threshold.

        Returns:
            A decorator function.

        Raises:
            TypeError: If the decorated function has ``recalled_memories``
                as a positional parameter.
        """
        client = self

        def decorator(
            fn: Callable[..., Coroutine[object, object, object]],
        ) -> Callable[..., Coroutine[object, object, object]]:
            # Check if recalled_memories is a positional arg
            sig = inspect.signature(fn)
            for name, param in sig.parameters.items():
                if (
                    name == "recalled_memories"
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and param.default is inspect.Parameter.empty
                ):
                    raise TypeError(
                        "Decorated function must not have 'recalled_memories' as a required positional parameter"
                    )

            @functools.wraps(fn)
            async def wrapper(*args: object, **kwargs: object) -> object:
                memories: list[MemoryResultDict] = []
                try:
                    query = kwargs.get(query_from, "")
                    if query:
                        raw = await client.recall(str(query), limit=limit)
                        memories = [m for m in raw if float(m["score"]) >= min_score]
                except (OSError, ValueError, StorageError, MemoryConnectionError):  # fail-open: recall failures
                    logger.debug("auto_recall_failed", op="auto_recall", outcome="failure", exc_info=True)
                    memories = []

                kwargs["recalled_memories"] = memories
                return await fn(*args, **kwargs)

            return wrapper

        return decorator

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
