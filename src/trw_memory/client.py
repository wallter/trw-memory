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
from trw_memory.security.runtime import (
    append_audit_event,
    delete_quarantined_entries,
    list_quarantined_entries,
    prepare_entry_for_store,
    store_quarantined_entry,
)
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


# ---- Bulk store types (PRD-MEM-future, scaffolded 2026-04-19) -----
# Amortise per-item lock + audit + embed overhead for ingestion-heavy
# workloads (audits, distill batch ingestion). See learning L-ujVK for
# the per-item-overhead measurement that motivated this API.


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

        Trades per-item audit + remote-publish overhead for throughput.
        The per-item *security* checks (PII redaction, poisoning detection)
        STILL run on every record — those guard correctness invariants;
        only the batched bookkeeping is amortised.

        Args:
            requests: List of :class:`BulkStoreRequest` describing each record.
            skip_audit_per_item: When True (default), one batch-summary audit
                event covers all inserts. When False, each insert emits its
                own audit event (matches per-call ``store()`` semantics) at
                the cost of throughput.
            skip_remote_publish: When True (default), remote-publish tasks
                are NOT scheduled per-item. Audit / sync workloads that ingest
                from a single trusted source typically don't need per-record
                remote publishing; bulk consumers can manage replication
                separately.

        Returns:
            :class:`BulkStoreSummary` with aggregate counts and per-item
            results in input order.

        Raises:
            ValueError: If ``requests`` is empty.

        Performance notes:
            On a 100-record batch the lock is acquired once vs 100 times,
            embeddings are computed via ``embed_batch`` (single model call),
            and the audit log gets one event vs 100. Backend ``store()``
            still runs per-entry (sqlite has no native multi-INSERT path
            here) but the lock+audit savings dominate the wall-clock cost
            on workloads where embedding + lock acquisition were the
            bottleneck (the trw-distill audit case, see learning L-ujVK).
        """
        if not requests:
            raise ValueError("bulk_store requires at least one request")

        self._require_permission(Permission.WRITE, "bulk_store")
        self._maybe_start_retry_drain()

        start_ts = datetime.now(timezone.utc)

        # Pre-validate ALL inputs before acquiring the lock so a single
        # bad row doesn't leave the batch half-written. Failed rows are
        # tracked and skipped.
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
        embedder = self._get_embedder()
        accepted_indices: list[int] = []
        accepted_entries: list[MemoryEntry] = []

        async with self._lock:
            backend = self._get_backend()
            now = datetime.now(timezone.utc)

            # Pass 1 — build entries, run security gate, partition.
            # decisions parallels prepared; entries that get rejected/quarantined
            # leave None at their index. Accepted entries store the
            # PreparedStoreEntry decision so Pass 3 can use op + pii_matches.
            decisions: list[Any] = [None] * len(prepared)
            for i, (req, validation_error) in enumerate(prepared):
                if validation_error is not None:
                    items.append(BulkStoreItemResult(
                        memory_id=req.entry_id or "",
                        status="rejected",
                        skipped_reason=validation_error,
                    ))
                    continue

                memory_id = req.entry_id or _make_id()
                existing = backend.get(memory_id) if req.entry_id is not None else None
                entry_metadata = dict(existing.metadata) if existing is not None else {}
                entry_metadata.update(req.metadata or {})
                entry_metadata.setdefault("installation_id", self._installation_id)

                if existing is None:
                    entry = MemoryEntry(
                        id=memory_id,
                        content=req.content.strip(),
                        detail=req.detail,
                        tags=req.tags or [],
                        importance=req.importance,
                        namespace=self._namespace,
                        metadata=entry_metadata,
                        created_at=now,
                        updated_at=now,
                        source=req.source,
                        source_identity=req.source_identity,
                        vector_clock=init_clock(self._local_node_id),
                    )
                else:
                    entry = existing.model_copy(update={
                        "content": req.content.strip(),
                        "detail": req.detail,
                        "tags": req.tags or [],
                        "importance": req.importance,
                        "metadata": entry_metadata,
                        "updated_at": now,
                        "source": req.source,
                        "source_identity": req.source_identity or existing.source_identity,
                        "vector_clock": init_clock(self._local_node_id),
                    })

                # Per-item security gate. Catch poisoning / authorization
                # so a single bad row doesn't abort the whole batch — the
                # bulk semantic is "best-effort, report per-item".
                try:
                    decision = prepare_entry_for_store(
                        entry,
                        backend=backend,
                        config=self._config,
                        session_id=req.session_id,
                    )
                except Exception as exc:
                    items.append(BulkStoreItemResult(
                        memory_id=memory_id,
                        status="rejected",
                        skipped_reason=f"{type(exc).__name__}:{str(exc)[:80]}",
                    ))
                    continue

                if decision.quarantined:
                    store_quarantined_entry(self._config, decision.entry)
                    items.append(BulkStoreItemResult(
                        memory_id=decision.entry.id,
                        status="quarantined",
                        quarantined=True,
                        anomaly_dimension=decision.anomaly_dimension,
                        z_score=decision.anomaly_z_score,
                    ))
                    continue

                accepted_indices.append(i)
                accepted_entries.append(decision.entry)
                decisions[i] = decision

            # Pass 2 — single batched embed call for accepted entries only.
            embeddings: list[list[float] | None] = []
            if accepted_entries and embedder is not None:
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

            # Pass 3 — backend store + vector + tier registration per accepted.
            for j, (orig_i, entry) in enumerate(zip(accepted_indices, accepted_entries)):
                decision = decisions[orig_i]
                assert decision is not None  # mypy guard; partitioning ensures this
                embedding = embeddings[j] if j < len(embeddings) else None

                if self._namespace.startswith("team:"):
                    NamespaceManager(backend).ensure_team_namespace(
                        self._namespace, created_at=now
                    )

                backend.store(entry)
                if embedding is not None:
                    try:
                        backend.upsert_vector(entry.id, embedding)
                    except Exception as exc:
                        try:
                            backend.delete(entry.id)
                        except Exception:
                            logger.exception(
                                "bulk_store_vector_rollback_failed",
                                memory_id=entry.id,
                            )
                            raise StorageError(
                                f"failed to persist vector for {entry.id!r}; rollback did not complete cleanly"
                            ) from exc
                        raise StorageError(
                            f"failed to persist vector for {entry.id!r}; entry write was rolled back"
                        ) from exc

                try:
                    schedule_graph_update(
                        entry, backend, embedding=embedding, config=self._config
                    )
                except RuntimeError:
                    logger.warning(
                        "bulk_store_graph_schedule_failed",
                        memory_id=entry.id,
                        exc_info=True,
                    )

                remember_entry_in_tiers(
                    self._config, self._namespace, entry, embedding
                )

                if not skip_audit_per_item:
                    append_audit_event(
                        self._config,
                        decision.op,
                        entry_id=entry.id,
                        actor=entry.source_identity or entry.source,
                        namespace=self._namespace,
                        data={
                            "status": "updated" if decision.op == "update" else "stored",
                            "session_id": prepared[orig_i][0].session_id,
                            "pii_types": sorted({m.pii_type for m in decision.pii_matches}),
                            "quarantined": False,
                        },
                    )

                items.append(BulkStoreItemResult(
                    memory_id=entry.id,
                    status="updated" if decision.op == "update" else "stored",
                ))

                if not skip_remote_publish and self._should_attempt_remote_publish(entry):
                    self._schedule_background_task(self._publish_entry(entry, embedding))

        # Single batch-summary audit event covering the whole call.
        stored_count = sum(1 for it in items if it.status == "stored")
        updated_count = sum(1 for it in items if it.status == "updated")
        quarantined_count = sum(1 for it in items if it.status == "quarantined")
        rejected_count = sum(1 for it in items if it.status == "rejected")
        end_ts = datetime.now(timezone.utc)
        duration_ms = (end_ts - start_ts).total_seconds() * 1000.0

        if skip_audit_per_item:
            append_audit_event(
                self._config,
                "bulk_store",
                entry_id="",
                actor=requests[0].source_identity or requests[0].source if requests else "agent",
                namespace=self._namespace,
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
            namespace=self._namespace,
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

        Uses a two-tier strategy:

        1. **Hybrid search** (preferred): Fetches all namespace entries and runs
           them through the retrieval pipeline (``hybrid_search``) which combines
           BM25 sparse retrieval with dense vector similarity via Reciprocal Rank
           Fusion (RRF).  This produces substantially better ranking than simple
           keyword matching.

        2. **Fallback TF scoring**: When the hybrid pipeline is unavailable
           (missing optional deps, import errors, or empty results), falls back
           to the original LIKE-based ``backend.search()`` with a blended
           term-frequency + importance score.

        Both paths apply tag filtering, min_score thresholds, limit capping,
        and optional token budget fitting.

        Args:
            query: Free-text search term.
            limit: Maximum number of results (must be >= 1).
            tags: If provided, results must contain all listed tags.
            min_score: Minimum score threshold for results.
            token_budget: If provided, truncate results to fit within this
                token budget.  Must be a positive integer.  When the budget
                is too small for all results, at least one result is always
                returned (minimum-one guarantee).
            include_distilled: PRD-DIST-005 FR-6. When False, records from
                trw-distill (source="distilled:*" or tag "distill:*") are
                excluded from results. When True (default), distilled records
                are included but dampened — see ``distilled_weight``.
            distilled_weight: PRD-DIST-005 FR-6. Dampening factor applied
                to distilled record scores before final ranking. When None
                (default), resolves from TRW_MEMORY_DISTILLED_RECALL_WEIGHT
                env var, falling back to DEFAULT_DISTILLED_RECALL_WEIGHT
                (0.75). Set to 1.0 to disable dampening.

        Returns:
            List of result dicts ordered by score descending.

        Raises:
            ValueError: If *limit* < 1 or *token_budget* <= 0.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if token_budget is not None and token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        self._require_permission(Permission.READ, "recall")
        self._maybe_start_retry_drain()
        await self._apply_pending_remote_retirements()
        embedder = self._get_embedder() if query.strip() else None
        query_embedding: list[float] | None = None
        if embedder is not None:
            query_embedding = await asyncio.to_thread(embedder.embed, query)

        async with self._lock:
            backend = self._get_backend()
            if self._namespace.startswith("team:") and NamespaceManager(backend).team_namespace_expired(
                self._namespace
            ):
                logger.debug(
                    "memory_recall_team_namespace_expired",
                    op="recall",
                    namespace=self._namespace,
                )
                return []
            if tier_runtime_enabled(self._config):
                tier_local_results = self._tier_results(backend, query, tags, limit, query_embedding)
                self._tier_manager = get_tier_manager(self._config, self._namespace)
            else:
                tier_local_results = []

        # --- Tier 1: Hybrid retrieval pipeline (BM25 + dense + RRF) ----------
        hybrid_results = await self._try_hybrid_recall(query, limit, tags)
        if hybrid_results is not None:
            # Apply min_score filter and limit
            filtered = [r for r in hybrid_results if r["score"] >= min_score]
            final_pre_policy = self._merge_tier_results(
                filtered[:limit],
                tier_local_results,
                limit,
                query.lower().split(),
                self._config,
                query_embedding,
            )
            if min_score > 0.0:
                final_pre_policy = [result for result in final_pre_policy if result["score"] >= min_score]
            if include_org_memories:
                final_pre_policy = await self._merge_org_results(
                    query, final_pre_policy, limit, tags, min_score
                )
            if include_shared:
                final_pre_policy = await self._merge_shared_results(query, final_pre_policy, limit)
            final_scored = cast(
                list[MemoryResultDict],
                apply_source_policy(
                    final_pre_policy,
                    include_distilled=include_distilled,
                    distilled_weight=distilled_weight,
                    include_source_kinds=include_source_kinds,
                    exclude_source_kinds=exclude_source_kinds,
                    source_weights=source_weights,
                    exclude_expired=exclude_expired,
                ),
            )
            if min_score > 0.0:
                final_scored = [result for result in final_scored if result["score"] >= min_score]
            final = self._apply_budget(final_scored[:limit], token_budget)
            await self._record_recall_access(final)
            append_audit_event(
                self._config,
                "recall",
                actor="",
                namespace=self._namespace,
                data={"query": query[:80], "entries_returned": len(final)},
            )
            self._remember_results_in_tiers(final)
            logger.debug(
                "memory_recalled",
                op="recall",
                outcome="success",
                query=query[:80],
                namespace=self._namespace,
                result_count=len(final),
                search_path="hybrid",
            )
            return final

        # --- Tier 2: Fallback LIKE + TF scoring ------------------------------
        results = await self._fallback_recall(query, limit, tags, min_score)
        results = self._merge_tier_results(
            results,
            tier_local_results,
            limit,
            query.lower().split(),
            self._config,
            query_embedding,
        )
        if min_score > 0.0:
            results = [result for result in results if result["score"] >= min_score]
        if include_org_memories:
            results = await self._merge_org_results(query, results, limit, tags, min_score)
        if include_shared:
            results = await self._merge_shared_results(query, results, limit)
        filtered_results = cast(
            list[MemoryResultDict],
            apply_source_policy(
                results,
                include_distilled=include_distilled,
                distilled_weight=distilled_weight,
                include_source_kinds=include_source_kinds,
                exclude_source_kinds=exclude_source_kinds,
                source_weights=source_weights,
                exclude_expired=exclude_expired,
            ),
        )
        if min_score > 0.0:
            filtered_results = [result for result in filtered_results if result["score"] >= min_score]
        final = self._apply_budget(filtered_results[:limit], token_budget)
        await self._record_recall_access(final)
        append_audit_event(
            self._config,
            "recall",
            actor="",
            namespace=self._namespace,
            data={"query": query[:80], "entries_returned": len(final)},
        )
        self._remember_results_in_tiers(final)
        return final

    @staticmethod
    def _apply_budget(
        results: list[MemoryResultDict],
        token_budget: int | None,
    ) -> list[MemoryResultDict]:
        """Apply token budget filtering to recall results.

        When *token_budget* is ``None``, returns *results* unchanged.

        Args:
            results: Ordered list of result dicts.
            token_budget: Maximum token budget, or ``None`` to skip.

        Returns:
            Filtered list of results that fit within the budget.
        """
        if token_budget is None or not results:
            return results

        from trw_memory.retrieval.token_budget import apply_token_budget

        # MemoryResultDict is a TypedDict — cast to dict[str, object] for the
        # budget function, then cast back.  The underlying dicts are the same
        # objects so no copy overhead.
        raw: list[dict[str, object]] = list(results)  # type: ignore[arg-type]
        filtered, _used, _truncated = apply_token_budget(raw, token_budget)
        return filtered  # type: ignore[return-value]

    async def _merge_org_results(
        self,
        query: str,
        local_results: list[MemoryResultDict],
        limit: int,
        tags: list[str] | None,
        min_score: float,
    ) -> list[MemoryResultDict]:
        """Append cross-validated sibling-project memories after local results."""
        try:
            org_entries = await asyncio.to_thread(
                functools.partial(
                    list_org_shared_entries,
                    self._config,
                    self._namespace,
                    exclude_keys={(result["namespace"], result["memory_id"]) for result in local_results},
                    limit=max(limit, 25),
                )
            )
        except Exception:
            logger.debug(
                "memory_org_recall_failed",
                op="recall",
                outcome="failure",
                namespace=self._namespace,
                exc_info=True,
            )
            return local_results

        tag_set = set(tags or [])
        org_results: list[MemoryResultDict] = []
        for entry in org_entries:
            # Re-apply the FR11 gate here instead of trusting the helper
            # contract alone so patched/mocked callers cannot accidentally
            # widen org recall beyond cross-validated high-importance entries.
            if not entry.cross_validated or entry.importance < 0.8:
                continue
            if tag_set and not tag_set.issubset(set(entry.tags)):
                continue

            candidate = _entry_to_result(entry, score=entry.importance)
            candidate["source"] = "org"
            if query.strip() and not self._matches_query(candidate, query):
                continue
            if min_score > 0.0 and candidate["score"] < min_score:
                continue
            org_results.append(candidate)

        return self._merge_shared_candidates(local_results, org_results)

    async def _try_hybrid_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
    ) -> list[MemoryResultDict] | None:
        """Attempt hybrid retrieval; return None to signal fallback.

        Fetches all entries for the namespace, runs them through
        ``hybrid_search`` (BM25 + optional dense vectors + RRF fusion),
        applies tag filtering, and converts to result dicts with
        positional scoring.

        Returns:
            A list of result dicts on success, or ``None`` when the hybrid
            pipeline is unavailable or produces no candidates.
        """
        try:
            from trw_memory.retrieval.pipeline import hybrid_search
        except ImportError:
            return None

        # Fetch candidate entries under lock (consistent with forget/store pattern)
        async with self._lock:
            backend = self._get_backend()
            all_entries = backend.list_entries(
                namespace=self._namespace,
                limit=limit * 5,
            )
            # The hybrid pipeline only runs dense similarity when callers supply
            # the stored vectors explicitly; loading them here keeps recall
            # aligned with the vectors written during store().
            stored_embeddings = backend.get_stored_embeddings([entry.id for entry in all_entries])

        if not all_entries:
            return None

        # Optionally obtain an embedding provider for dense retrieval
        embedder = self._get_embedder()

        try:
            ranked = hybrid_search(
                query=query,
                entries=all_entries,
                embedder=embedder,
                stored_embeddings=stored_embeddings or None,
                top_k=limit * 3,
            )
        except Exception:
            logger.debug(
                "hybrid_search_failed",
                op="recall",
                outcome="failure",
                exc_info=True,
            )
            return None

        if not ranked:
            return None

        # Apply tag filter
        if tags:
            tag_set = set(tags)
            ranked = [e for e in ranked if tag_set.issubset(set(e.tags))]

        # Convert to result dicts with RRF-style positional scoring
        results: list[MemoryResultDict] = []
        for rank, entry in enumerate(ranked):
            score = round(1.0 / (1 + rank), 4)
            results.append(_entry_to_result(entry, score=score))

        return results

    async def _fallback_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
        min_score: float,
    ) -> list[MemoryResultDict]:
        """Original LIKE-based search with TF + importance scoring.

        Uses ``backend.search()`` for keyword matching and blends a
        term-frequency relevance score (weight ``_FALLBACK_TF_WEIGHT``)
        with the entry's stored importance (weight
        ``_FALLBACK_IMPORTANCE_WEIGHT``).
        """
        async with self._lock:
            entries = self._get_backend().search(
                query,
                top_k=limit * 3,
                tags=tags,
                namespace=self._namespace,
            )

        query_terms = set(query.lower().split())
        results: list[MemoryResultDict] = []
        for entry in entries:
            if not query_terms:
                tf_score = entry.importance
            else:
                text_tokens = f"{entry.content} {entry.detail} {' '.join(entry.tags)}".lower().split()
                matches = sum(1 for t in text_tokens if t in query_terms)
                tf_score = (
                    min(1.0, matches / max(len(text_tokens), 1) * _FALLBACK_TF_SCALE) * _FALLBACK_TF_WEIGHT
                    + entry.importance * _FALLBACK_IMPORTANCE_WEIGHT
                )
            if tf_score >= min_score:
                results.append(_entry_to_result(entry, score=round(tf_score, 4)))

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        final = results[:limit]
        logger.debug(
            "memory_recalled",
            op="recall",
            outcome="success",
            query=query[:80],
            namespace=self._namespace,
            result_count=len(final),
            search_path="fallback",
        )
        return final

    async def _record_recall_access(self, results: list[MemoryResultDict]) -> None:
        """Persist access metadata for the entries that were actually returned."""
        grouped: dict[str, list[str]] = {}
        for result in results:
            if result.get("source") == "shared":
                continue
            grouped.setdefault(result["namespace"], []).append(result["memory_id"])

        if not grouped:
            return

        async with self._lock:
            for namespace, entry_ids in grouped.items():
                if namespace == self._namespace:
                    record_recall_access(self._get_backend(), entry_ids)
                else:
                    with _create_local_backend(self._config, namespace) as backend:
                        record_recall_access(backend, entry_ids)
                append_audit_event(
                    self._config,
                    "access",
                    actor="",
                    namespace=namespace,
                    data={"entry_ids": entry_ids, "entries_accessed": len(entry_ids)},
                )

    def _tier_results(
        self,
        backend: StorageBackend,
        query: str,
        tags: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResultDict]:
        """Collect local tier-managed candidates for this namespace."""
        candidates = tier_candidates(
            self._config,
            self._namespace,
            backend,
            query=query,
            tags=tags,
            limit=limit,
            query_embedding=query_embedding,
        )
        return [self._tier_result_from_entry(candidate) for candidate in candidates]

    def _remember_results_in_tiers(self, results: list[MemoryResultDict]) -> None:
        """Keep the hot/warm tiers aligned with the entries callers actually saw."""
        recalled_at = datetime.now(timezone.utc).isoformat()
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
            remember_entry_data_in_tiers(self._config, payload)

    @staticmethod
    def _merge_tier_results(
        local_results: list[MemoryResultDict],
        tier_results: list[MemoryResultDict],
        limit: int,
        query_tokens: list[str],
        config: MemoryConfig,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResultDict]:
        """Merge tier-only candidates into the normal local recall results."""
        if not tier_results:
            return local_results[:limit]
        merged = list(local_results)
        seen_ids = {result["memory_id"] for result in local_results}
        seen_content = {result["content"] for result in local_results}
        for result in tier_results:
            if result["memory_id"] in seen_ids or result["content"] in seen_content:
                continue
            merged.append(result)
            seen_ids.add(result["memory_id"])
            seen_content.add(result["content"])
        if len(merged) == len(local_results):
            return local_results[:limit]
        for result in merged:
            relevance_hint = result.get("_relevance_hint")
            result["score"] = round(
                compute_importance_score(
                    cast("dict[str, object]", result),
                    query_tokens,
                    query_embedding=query_embedding,
                    config=config,
                    relevance_hint=float(relevance_hint) if relevance_hint is not None else None,
                ),
                4,
            )
        merged.sort(key=lambda result: float(result["score"]), reverse=True)
        return merged[:limit]

    @staticmethod
    def _tier_result_from_entry(entry: dict[str, object]) -> MemoryResultDict:
        """Convert a tier-managed entry dict into the client recall result shape."""
        raw_score = entry.get("score")
        score = float(str(raw_score)) if raw_score is not None else entry_utility(entry)
        raw_tags = entry.get("tags", [])
        tier_result: MemoryResultDict = {
            "memory_id": str(entry.get("id", entry.get("memory_id", ""))),
            "content": str(entry.get("content", "")),
            "detail": str(entry.get("detail", "")),
            "tags": [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else [],
            "importance": MemoryClient._coerce_float(entry.get("importance", 0.0)),
            "score": round(score, 4),
            "created_at": str(entry.get("created_at", "")),
            "updated_at": str(entry.get("updated_at", entry.get("created_at", ""))),
            "namespace": str(entry.get("namespace", "default")),
            "source": "local",
            "last_accessed_at": str(entry.get("last_accessed_at", "")),
            "q_value": MemoryClient._coerce_float(entry.get("q_value", 0.0)),
            "q_observations": int(str(entry.get("q_observations", 0))),
            "recurrence": int(str(entry.get("recurrence", 1))),
            "access_count": int(str(entry.get("access_count", 0))),
            "_relevance_hint": MemoryClient._coerce_float(entry.get("_tier_relevance", score)),
        }
        return tier_result

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

        Args:
            memory_id: ID of the memory to delete.
            actor: Optional actor identity for GDPR-style bulk erasure.

        Returns:
            Dict with ``memory_id``, ``status``, ``namespace``.

        Raises:
            MemoryNotFoundError: If *memory_id* does not exist or belongs
                to a different namespace.
        """
        self._require_permission(Permission.DELETE, "forget")
        self._maybe_start_retry_drain()
        if not memory_id and not actor:
            raise ValueError("memory_id or actor must be provided")
        async with self._lock:
            backend = self._get_backend()
            if actor:
                deleted_count = 0
                for candidate in backend.list_entries(
                    namespace=self._namespace,
                    limit=max(10_000, backend.count(namespace=self._namespace)),
                ):
                    if candidate.source_identity != actor:
                        continue
                    if backend.delete(candidate.id):
                        deleted_count += 1
                        remove_entry_from_tiers(self._config, self._namespace, candidate.id)
                deleted_count += delete_quarantined_entries(self._config, namespace=self._namespace, actor=actor)
                append_audit_event(
                    self._config,
                    "forget",
                    actor=actor,
                    namespace=self._namespace,
                    data={"entries_deleted": deleted_count, "selector": "actor"},
                )
                actor_forget_result: ForgetResultDict = {
                    "memory_id": "",
                    "status": "deleted",
                    "namespace": self._namespace,
                    "entries_deleted": deleted_count,
                }
                return actor_forget_result

            assert memory_id is not None
            existing = backend.get(memory_id)
            if existing is None:
                quarantined_deleted = delete_quarantined_entries(
                    self._config,
                    namespace=self._namespace,
                    memory_id=memory_id,
                )
                if quarantined_deleted == 0:
                    raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found")
                append_audit_event(
                    self._config,
                    "forget",
                    entry_id=memory_id,
                    actor="",
                    namespace=self._namespace,
                    data={"entries_deleted": quarantined_deleted, "quarantined": True},
                )
                quarantined_forget_result: ForgetResultDict = {
                    "memory_id": memory_id,
                    "status": "deleted",
                    "namespace": self._namespace,
                    "entries_deleted": quarantined_deleted,
                }
                return quarantined_forget_result
            if existing.namespace != self._namespace:
                raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found in namespace {self._namespace!r}")
            remote_id = existing.remote_id
            backend.delete(memory_id)
            remove_entry_from_tiers(self._config, self._namespace, memory_id)
            append_audit_event(
                self._config,
                "forget",
                entry_id=memory_id,
                actor=existing.source_identity,
                namespace=self._namespace,
                data={"entries_deleted": 1, "quarantined": False},
            )
        if remote_id:
            self._schedule_background_task(self._retire_remote_entry(memory_id, remote_id))

        logger.debug(
            "memory_forgotten",
            op="forget",
            outcome="success",
            memory_id=memory_id,
            namespace=self._namespace,
        )
        forget_result: ForgetResultDict = {
            "memory_id": memory_id,
            "status": "deleted",
            "namespace": self._namespace,
            "entries_deleted": 1,
        }
        return forget_result

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
        """Search memories with filters.

        Args:
            tags: If provided, results must contain all listed tags.
            min_importance: Lower bound on importance (inclusive, ``[0.0, 1.0]``).
            since: If provided, only return entries created after this time.
            limit: Maximum number of results (must be >= 1).

        Returns:
            List of result dicts ordered by importance descending.

        Raises:
            ValueError: If *limit* < 1 or *min_importance* out of range.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if not 0.0 <= min_importance <= 1.0:
            raise ValueError(f"min_importance must be in [0.0, 1.0], got {min_importance}")
        if status is not None and status not in {"active", "resolved", "obsolete", "archived", "quarantined"}:
            raise ValueError(f"status must be one of active/resolved/obsolete/archived/quarantined, got {status!r}")
        self._require_permission(Permission.READ, "search")
        self._maybe_start_retry_drain()
        await self._apply_pending_remote_retirements()

        if status == "quarantined":
            entries = list_quarantined_entries(
                self._config,
                namespace=self._namespace,
                actor=actor,
                limit=max(limit * 5, 10_000) if actor is not None else limit * 5,
            )
        else:
            async with self._lock:
                fetch_limit = limit * 5
                if actor is not None:
                    fetch_limit = max(fetch_limit, self._get_backend().count(namespace=self._namespace))
                entries = self._get_backend().list_entries(
                    namespace=self._namespace,
                    limit=fetch_limit,  # over-fetch to allow for post-filtering
                )

        # Post-filter
        tag_set: set[str] = set(tags) if tags else set()
        results: list[MemoryResultDict] = []
        for entry in entries:
            if actor is not None and entry.source_identity != actor:
                continue
            if status is not None and status != "quarantined" and entry.status.value != status:
                continue
            if entry.importance < min_importance:
                continue
            if tag_set and not tag_set.issubset(set(entry.tags)):
                continue
            if since is not None and entry.created_at < since:
                continue
            results.append(_entry_to_result(entry, score=entry.importance))

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        final = results[:limit]
        logger.debug(
            "memory_searched",
            op="search",
            outcome="success",
            namespace=self._namespace,
            tag_filter=tags,
            min_importance=min_importance,
            result_count=len(final),
        )
        append_audit_event(
            self._config,
            "access",
            actor=actor or "",
            namespace=self._namespace,
            data={
                "entries_returned": len(final),
                "status": status or "",
                "tag_filter": tags or [],
            },
        )
        return final

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
        ) -> list[MemoryResultDict]:
            return await client.recall(
                query,
                limit=limit,
                include_org_memories=include_org_memories,
                include_shared=include_shared,
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
