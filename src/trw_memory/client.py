# ruff: noqa: E402,F401,I001,RUF100
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
import threading
import uuid
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

import structlog
from typing_extensions import NotRequired

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import MemoryConnectionError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.rbac import Permission, require_namespace_permission as require_namespace_permission  # noqa: F401 — re-exported for downstream consumers
from trw_memory.security.runtime import (
    audit_entry,
    delete_quarantined_entries as delete_quarantined_entries,  # noqa: F401 — test-patched
    review_quarantined_entry,
)
from trw_memory.graph import list_org_shared_entries as list_org_shared_entries  # noqa: F401 — test-patched at module path
from trw_memory.lifecycle.tiers._runtime import remove_entry_from_tiers as remove_entry_from_tiers  # noqa: F401 — test-patched
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.remote import (
    _anonymize_entry as _anonymize_entry,  # noqa: F401 — test-patched
    drain_retry_queue as drain_retry_queue,  # noqa: F401 — test-patched
    fetch_shared_memories as fetch_shared_memories,  # noqa: F401 — test-patched
    publish_memory_result as publish_memory_result,  # noqa: F401 — test-patched
    retire_remote_memory as retire_remote_memory,  # noqa: F401 — test-patched
)
from trw_memory.sync.retry_queue import RetryQueue
from trw_memory.sync.subscriber import SSESubscriber as SSESubscriber  # noqa: F401 — test-patched

# Org-shared recall alias seam extracted to _client_org_shared_aliases.py
# (PRD-DIST-246 effective-LOC ratchet). MemoryClient mixes it in so the
# `self._X` / `MemoryClient._X` monkeypatch seam resolves via the MRO.
from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "SHARED_EVENT_CACHE_MAX",
    "BulkStoreItemResult",
    "BulkStoreRequest",
    "BulkStoreSummary",
    "ForgetResultDict",
    "MemoryClient",
    "MemoryResultDict",
    "StoreResultDict",
]


# Shared-event-cache cap extracted to _client_lifecycle.py (PRD-DIST-246
# batch 111). Re-export preserves the public API for downstream consumers
# and tests that import the constant from the facade.
from trw_memory._client_lifecycle import (  # noqa: E402
    SHARED_EVENT_CACHE_MAX as SHARED_EVENT_CACHE_MAX,
)

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


# TypedDict shapes + agent protocols extracted to _client_models.py
# (PRD-DIST-246 batch 113). Re-exports preserve the public API.
from trw_memory._client_models import (  # noqa: E402
    AgentWithRegisterTool as AgentWithRegisterTool,
    AgentWithToolDecorator as AgentWithToolDecorator,
    ForgetResultDict as ForgetResultDict,
    MemoryResultDict as MemoryResultDict,
    StoreResultDict as StoreResultDict,
    _ToolFn as _ToolFn,
)


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


class MemoryClient(OrgSharedAliasMixin):
    """High-level async client for the trw-memory system.

    Args:
        namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
        mode: Transport mode — ``"local"`` (SQLite/YAML), ``"mcp"`` (stdio),
            or ``"auto"`` (try local first).
        timeout: Timeout in seconds for remote operations.
    """

    # Instance attribute declarations — `__init__` body lives in
    # `_client_lifecycle.init_client` (PRD-DIST-246 batch 111), so mypy
    # needs explicit class-level type hints to see these attributes.
    _namespace: str
    _timeout: float
    _lock: asyncio.Lock
    _tools_registered: bool
    _backend: StorageBackend | None
    _resolved_mode: str
    _config: MemoryConfig
    _project_root: str
    _installation_id: str
    _local_node_id: str
    _background_tasks: set[asyncio.Task[None]]
    _retry_queue: RetryQueue
    _retry_drain_started: bool
    _shared_event_cache: list[MemoryResultDict]
    _shared_event_cache_lock: threading.Lock
    _pending_remote_retirements: set[str]
    _pending_remote_retirements_lock: threading.Lock
    _sse_subscriber: SSESubscriber | None
    _sse_subscriber_started: bool
    _tier_manager: object | None

    def __init__(self, namespace: str, mode: Literal["local", "mcp", "auto"] = "auto", timeout: float = 5.0) -> None:
        """Initialise a MemoryClient with namespace isolation and mode selection.

        Implementation lives in ``_client_lifecycle.init_client``
        (PRD-DIST-246 batch 111). Mode is one of ``"local"`` / ``"mcp"``
        / ``"auto"``. Sets up state, validates namespace, opens the
        backend, runs security defaults verification, seeds canaries,
        warms the tier manager, and starts the SSE subscription.
        """
        from trw_memory._client_lifecycle import init_client as _impl

        _impl(self, namespace, mode=mode, timeout=timeout)

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

        Implementation lives in ``_client_store.store_impl``
        (PRD-DIST-246 batch 110). Full per-entry write path: schema
        validation, security gate, vector upsert with rollback, graph
        schedule, tier register, audit, optional remote publish.
        """
        from trw_memory._client_store import store_impl as _impl

        return await _impl(
            self,
            content,
            tags=tags,
            importance=importance,
            detail=detail,
            metadata=metadata,
            expires=expires,
            source=source,
            source_identity=source_identity,
            session_id=session_id,
            entry_id=entry_id,
        )

    async def bulk_store(
        self, requests: list[BulkStoreRequest], *, skip_audit_per_item: bool = True, skip_remote_publish: bool = True
    ) -> BulkStoreSummary:
        """Store many records in one batched operation.

        Implementation lives in ``_client_bulk_store.bulk_store_impl``
        (PRD-DIST-246 batch 104). See that helper's docstring for full
        arg/return semantics. Trades per-item audit + remote-publish
        overhead for throughput; per-item security checks (PII /
        poisoning) still run on every record.
        """
        return await _bulk_store_impl(
            self, requests, skip_audit_per_item=skip_audit_per_item, skip_remote_publish=skip_remote_publish
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
        confidence_floor: float | None = None,
        exclude_historical_only: bool | None = None,
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
            confidence_floor=confidence_floor,
            exclude_historical_only=exclude_historical_only,
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
    def _apply_budget(results: list[MemoryResultDict], token_budget: int | None) -> list[MemoryResultDict]:
        from trw_memory._client_recall import apply_budget as _impl

        return _impl(results, token_budget)

    async def _merge_org_results(
        self, query: str, local_results: list[MemoryResultDict], limit: int, tags: list[str] | None, min_score: float
    ) -> list[MemoryResultDict]:
        from trw_memory._client_recall import merge_org_results as _impl

        return await _impl(self, query, local_results, limit, tags, min_score)

    async def _try_hybrid_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResultDict] | None:
        from trw_memory._client_recall import try_hybrid_recall as _impl

        return await _impl(self, query, limit, tags, query_embedding=query_embedding)

    async def _fallback_recall(
        self, query: str, limit: int, tags: list[str] | None, min_score: float
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

    # ---- Remote publish aliases (PRD-DIST-246 batch 111) -------------------

    def _should_attempt_remote_publish(self, entry: MemoryEntry) -> bool:
        from trw_memory._client_lifecycle import should_attempt_remote_publish as _impl

        return _impl(self, entry)

    def _schedule_background_task(self, coro: Coroutine[object, object, None]) -> None:
        from trw_memory._client_lifecycle import schedule_background_task as _impl

        _impl(self, coro)

    async def _publish_entry(self, entry: MemoryEntry, embedding: list[float] | None) -> None:
        from trw_memory._client_lifecycle import publish_entry as _impl

        await _impl(self, entry, embedding)

    # Org-shared helper aliases (PRD-DIST-246 batch 107) moved to the
    # ``OrgSharedAliasMixin`` base (`_client_org_shared_aliases.py`).

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
            self, tags=tags, min_importance=min_importance, since=since, limit=limit, actor=actor, status=status
        )

    async def audit_learning(self, learning_id: str) -> dict[str, object]:
        """Return SEC-001 audit data for an active or quarantined learning."""
        return audit_entry(self._config, learning_id=learning_id, active_backend=self._get_backend())

    async def review_quarantined(
        self, learning_id: str, *, decision: Literal["approve", "reject"], reviewer_id: str
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
        self, query_from: str, limit: int = 10, min_score: float = 0.0
    ) -> Callable[[Callable[..., Coroutine[object, object, object]]], Callable[..., Coroutine[object, object, object]]]:
        from trw_memory._client_tools_binding import auto_recall as _impl

        return _impl(self, query_from, limit=limit, min_score=min_score)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    # ---- Lifecycle aliases (PRD-DIST-246 batches 111+112) -----------------

    async def __aenter__(self) -> MemoryClient:
        from trw_memory._client_lifecycle import aenter as _impl

        return await _impl(self)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        from trw_memory._client_lifecycle import aexit as _impl

        await _impl(self, exc_type, exc_val, cast("Any", exc_tb))

    async def close(self) -> None:
        from trw_memory._client_lifecycle import close_client as _impl

        await _impl(self)

    def _should_start_retry_drain(self) -> bool:
        from trw_memory._client_lifecycle import should_start_retry_drain as _impl

        return _impl(self)

    def _should_start_sse_subscription(self) -> bool:
        from trw_memory._client_lifecycle import should_start_sse_subscription as _impl

        return _impl(self)

    def _maybe_start_sse_subscription(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_sse_subscription as _impl

        _impl(self)

    def _maybe_start_retry_drain(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_retry_drain as _impl

        _impl(self)

    def _handle_sse_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import handle_sse_event as _impl

        _impl(self, event)

    def _cache_shared_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import cache_shared_event as _impl

        _impl(self, event)

    async def _drain_retry_queue(self) -> None:
        from trw_memory._client_lifecycle import drain_retry_queue_impl as _impl

        await _impl(self)

    async def _retire_remote_entry(self, memory_id: str, remote_id: str) -> None:
        from trw_memory._client_lifecycle import retire_remote_entry as _impl

        await _impl(self, memory_id, remote_id)

    async def _apply_pending_remote_retirements(self) -> None:
        from trw_memory._client_lifecycle import apply_pending_remote_retirements as _impl

        await _impl(self)
