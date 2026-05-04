"""Client lifecycle — context-manager + remote-publish + sync + SSE.

Belongs to ``client.py``. Re-exported there for back-compat.

13 helpers + the ``init_client`` ctor-body covering:

- Context-manager protocol (``aenter`` / ``aexit`` / ``close_client``).
- Remote-publish (`should_attempt_remote_publish`, `publish_entry`,
  `schedule_background_task`).
- Retry queue drain (`should_start_retry_drain`,
  `maybe_start_retry_drain`, `drain_retry_queue_impl`).
- SSE subscription (`should_start_sse_subscription`,
  `maybe_start_sse_subscription`, `handle_sse_event`,
  `cache_shared_event`).
- Remote retirement (`retire_remote_entry`,
  `apply_pending_remote_retirements`).
- Constructor (`init_client`) — sets up state and starts SSE.

All helpers take ``client: MemoryClient`` as first arg
(backend-handle pattern). Logger lookup goes through the parent
module so test patches on ``trw_memory.client.logger`` propagate.

Extracted as PRD-DIST-246 batches 111 + 112.
"""

from __future__ import annotations

import asyncio
import functools
import socket
import threading
from collections.abc import Coroutine
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast

import structlog

from trw_memory.exceptions import MemoryConnectionError
from trw_memory.lifecycle.tiers._runtime import tier_runtime_enabled, warmup_tier_manager
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.pii import anonymize_installation_id
from trw_memory.security.runtime import initialize_canaries
from trw_memory.security.startup import verify_defaults
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.retry_queue import RetryQueue
from trw_memory.sync.subscriber import SSESubscriber

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)

SHARED_EVENT_CACHE_MAX = 256


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c
    return _c.logger


def _create_local_backend(config: MemoryConfig, namespace: str) -> StorageBackend:
    from trw_memory.client import _create_local_backend as _impl
    return _impl(config, namespace)


# ---------------------------------------------------------------------------
# Constructor body — extracted from MemoryClient.__init__
# ---------------------------------------------------------------------------


def init_client(
    client: "MemoryClient",
    namespace: str,
    mode: Literal["local", "mcp", "auto"] = "auto",
    timeout: float = 5.0,
) -> None:
    """Initialize a MemoryClient instance — extracted ctor body."""
    validate_namespace(namespace)
    client._namespace = namespace
    client._timeout = timeout
    client._lock = asyncio.Lock()
    client._tools_registered = False
    client._backend = None
    client._resolved_mode = ""
    client._config = MemoryConfig()
    client._project_root = str(Path.cwd())
    client._installation_id = f"{socket.gethostname()}:{Path(client._config.storage_path).resolve()}"
    client._local_node_id = anonymize_installation_id(client._installation_id)
    client._background_tasks = set()
    client._retry_queue = RetryQueue(Path(client._config.storage_path) / "sync_queue.jsonl")
    client._retry_drain_started = False
    client._shared_event_cache = []
    client._shared_event_cache_lock = threading.Lock()
    client._pending_remote_retirements = set()
    client._pending_remote_retirements_lock = threading.Lock()
    client._sse_subscriber = None
    client._sse_subscriber_started = False
    client._tier_manager = None

    if mode == "mcp":
        raise NotImplementedError("MCP mode is not yet implemented")

    if mode in ("local", "auto"):
        try:
            client._backend = _create_local_backend(client._config, namespace)
            verify_defaults(client._config)
            initialize_canaries(client._config, backend=client._backend)
            client._resolved_mode = "local"
            _client_logger().debug(
                "client_initialized",
                op="init",
                namespace=namespace,
                mode=client._resolved_mode,
                backend=client._config.storage_backend,
            )
            if tier_runtime_enabled(client._config):
                client._tier_manager = warmup_tier_manager(client._config, namespace, client._backend)
        except (OSError, ValueError, ImportError) as exc:
            if mode == "local":
                raise MemoryConnectionError(f"Failed to create local backend: {exc}") from exc

    if mode == "auto" and client._backend is None:
        raise MemoryConnectionError("No connection mode available. Tried: local.")

    maybe_start_sse_subscription(client)


# ---------------------------------------------------------------------------
# Remote publish
# ---------------------------------------------------------------------------


def should_attempt_remote_publish(client: "MemoryClient", entry: MemoryEntry) -> bool:
    return (
        not client._config.local_only
        and client._config.sync_enabled
        and bool(client._config.platform_url)
        and entry.importance >= client._config.sync_min_importance
    )


def schedule_background_task(client: "MemoryClient", coro: Coroutine[object, object, None]) -> None:
    task = asyncio.create_task(coro)
    client._background_tasks.add(task)
    task.add_done_callback(client._background_tasks.discard)


async def publish_entry(
    client: "MemoryClient",
    entry: MemoryEntry,
    embedding: list[float] | None,
) -> None:
    from trw_memory import client as _c
    publish_result = await asyncio.to_thread(
        functools.partial(
            _c.publish_memory_result,
            entry,
            client._config,
            embedding=embedding,
            project_root=client._project_root,
        )
    )
    if publish_result["success"]:
        async with client._lock:
            backend = client._get_backend()
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

    payload = await asyncio.to_thread(_c._anonymize_entry, entry, client._project_root)
    if embedding is not None:
        payload["embedding"] = embedding
    queue_payload = cast("dict[str, object]", payload)
    enqueued = await asyncio.to_thread(client._retry_queue.enqueue, entry.id, queue_payload)
    if not enqueued:
        _client_logger().warning(
            "memory_sync_queue_full",
            op="store",
            outcome="failure",
            memory_id=entry.id,
            namespace=client._namespace,
        )


# ---------------------------------------------------------------------------
# Context-manager + close
# ---------------------------------------------------------------------------


async def aenter(client: "MemoryClient") -> "MemoryClient":
    maybe_start_retry_drain(client)
    return client


async def aexit(
    client: "MemoryClient",
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
) -> None:
    await close_client(client)


async def close_client(client: "MemoryClient") -> None:
    if client._sse_subscriber is not None:
        client._sse_subscriber.stop()
        client._sse_subscriber = None
        client._sse_subscriber_started = False
    if client._background_tasks:
        await asyncio.gather(*list(client._background_tasks), return_exceptions=True)
    if client._backend is not None:
        client._backend.close()
        client._backend = None
        _client_logger().debug("client_closed", op="close", namespace=client._namespace)


# ---------------------------------------------------------------------------
# Retry queue drain
# ---------------------------------------------------------------------------


def should_start_retry_drain(client: "MemoryClient") -> bool:
    return (
        not client._retry_drain_started
        and not client._config.local_only
        and client._config.sync_enabled
        and bool(client._config.platform_url)
        and client._retry_queue.depth() > 0
    )


def maybe_start_retry_drain(client: "MemoryClient") -> None:
    if should_start_retry_drain(client):
        client._retry_drain_started = True
        schedule_background_task(client, drain_retry_queue_impl(client))


async def drain_retry_queue_impl(client: "MemoryClient") -> None:
    from trw_memory import client as _c
    queued_before = {record["entry_id"] for record in client._retry_queue.snapshot()}
    result = await asyncio.to_thread(_c.drain_retry_queue, client._retry_queue, client._config)
    queued_after = {record["entry_id"] for record in client._retry_queue.snapshot()}
    drained_ids = queued_before - queued_after
    if drained_ids:
        async with client._lock:
            backend = client._get_backend()
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
    _client_logger().debug(
        "memory_sync_queue_drained",
        op="session_start",
        outcome="success",
        namespace=client._namespace,
        drained=result["drained"],
        failed=result["failed"],
        skipped=result["skipped"],
    )


# ---------------------------------------------------------------------------
# SSE subscription
# ---------------------------------------------------------------------------


def should_start_sse_subscription(client: "MemoryClient") -> bool:
    return (
        not client._sse_subscriber_started
        and not client._config.local_only
        and client._config.sync_enabled
        and bool(client._config.platform_url)
        and bool(client._config.platform_api_key)
    )


def maybe_start_sse_subscription(client: "MemoryClient") -> None:
    if not should_start_sse_subscription(client):
        return
    from trw_memory import client as _c
    subscriber = _c.SSESubscriber(
        client._config,
        on_event=lambda event: handle_sse_event(client, event),
    )
    subscriber.start()
    client._sse_subscriber = subscriber
    client._sse_subscriber_started = True


def handle_sse_event(client: "MemoryClient", event: dict[str, object]) -> None:
    event_type = str(event.get("type", ""))
    if event_type in {"learning_published", "learning_updated"}:
        cache_shared_event(client, event)
        return
    if event_type == "learning_retired":
        remote_id = str(event.get("id", ""))
        if not remote_id:
            return
        with client._pending_remote_retirements_lock:
            client._pending_remote_retirements.add(remote_id)
        with client._shared_event_cache_lock:
            client._shared_event_cache = [
                cached for cached in client._shared_event_cache if cached["memory_id"] != remote_id
            ]


def cache_shared_event(client: "MemoryClient", event: dict[str, object]) -> None:
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
    with client._shared_event_cache_lock:
        client._shared_event_cache = [
            existing for existing in client._shared_event_cache if existing["memory_id"] != remote_id
        ]
        client._shared_event_cache.append(cached)
        if len(client._shared_event_cache) > SHARED_EVENT_CACHE_MAX:
            client._shared_event_cache = client._shared_event_cache[-SHARED_EVENT_CACHE_MAX:]


# ---------------------------------------------------------------------------
# Remote retirement
# ---------------------------------------------------------------------------


async def retire_remote_entry(
    client: "MemoryClient",
    memory_id: str,
    remote_id: str,
) -> None:
    from trw_memory import client as _c
    retired = await asyncio.to_thread(_c.retire_remote_memory, remote_id, client._config)
    if retired:
        _client_logger().debug(
            "memory_remote_retired",
            op="forget",
            outcome="success",
            memory_id=memory_id,
            remote_id=remote_id,
            namespace=client._namespace,
        )
        return
    _client_logger().warning(
        "memory_remote_retire_failed",
        op="forget",
        outcome="failure",
        memory_id=memory_id,
        remote_id=remote_id,
        namespace=client._namespace,
    )


async def apply_pending_remote_retirements(client: "MemoryClient") -> None:
    with client._pending_remote_retirements_lock:
        remote_ids = set(client._pending_remote_retirements)
        client._pending_remote_retirements.clear()
    if not remote_ids:
        return

    async with client._lock:
        backend = client._get_backend()
        limit = max(backend.count(namespace=client._namespace), 1)
        entries = backend.list_entries(namespace=client._namespace, limit=limit)
        unresolved = set(remote_ids)
        for entry in entries:
            if entry.remote_id not in unresolved:
                continue
            if entry.last_synced_at is None:
                continue
            updated = backend.update(entry.id, pending_delete=True)
            if updated is not None:
                unresolved.discard(str(entry.remote_id))
    if unresolved:
        with client._pending_remote_retirements_lock:
            client._pending_remote_retirements.update(unresolved)
