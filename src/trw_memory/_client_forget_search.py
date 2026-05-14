"""Forget + search async impls.

Belongs to ``client.py``. Re-exported there for back-compat.

Two methods extracted from ``MemoryClient``:

- ``forget_impl`` — async delete by memory_id OR actor; routes through
  the quarantine-aware delete path; emits audit event.
- ``search_impl`` — filtered search (tags + min_importance + since +
  status filter); not full-text — uses ``recall(...)`` for that.

Both helpers use the backend-handle pattern from PRD-DIST-245 batch 87
(take ``client: MemoryClient`` as first arg).

The parent-module ``logger`` lookup goes through ``_client_logger()``
so test patches on ``trw_memory.client.logger`` propagate.

Extracted as PRD-DIST-246 batch 106.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.exceptions import MemoryNotFoundError
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import (
    append_audit_event,
    list_quarantined_entries,
)

if TYPE_CHECKING:
    from trw_memory.client import ForgetResultDict, MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c
    return _c.logger


def _entry_to_result(entry: Any, score: float = 0.0) -> MemoryResultDict:
    from trw_memory._client_distilled_tiering import entry_to_result as _impl
    return _impl(entry, score=score)


async def forget_impl(
    client: MemoryClient,
    memory_id: str | None = None,
    *,
    actor: str | None = None,
) -> ForgetResultDict:
    """Async impl for :meth:`MemoryClient.forget`."""
    from trw_memory import client as _c
    client._require_permission(Permission.DELETE, "forget")
    client._maybe_start_retry_drain()
    if not memory_id and not actor:
        raise ValueError("memory_id or actor must be provided")
    async with client._lock:
        backend = client._get_backend()
        if actor:
            deleted_count = 0
            for candidate in backend.list_entries(
                namespace=client._namespace,
                limit=max(10_000, backend.count(namespace=client._namespace)),
            ):
                if candidate.source_identity != actor:
                    continue
                if backend.delete(candidate.id):
                    deleted_count += 1
                    _c.remove_entry_from_tiers(client._config, client._namespace, candidate.id)
            deleted_count += _c.delete_quarantined_entries(client._config, namespace=client._namespace, actor=actor)
            append_audit_event(
                client._config,
                "forget",
                actor=actor,
                namespace=client._namespace,
                data={"entries_deleted": deleted_count, "selector": "actor"},
            )
            actor_forget_result: ForgetResultDict = {
                "memory_id": "",
                "status": "deleted",
                "namespace": client._namespace,
                "entries_deleted": deleted_count,
            }
            return actor_forget_result

        assert memory_id is not None  # noqa: S101
        existing = backend.get(memory_id)
        if existing is None:
            quarantined_deleted = _c.delete_quarantined_entries(
                client._config,
                namespace=client._namespace,
                memory_id=memory_id,
            )
            if quarantined_deleted == 0:
                raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found")
            append_audit_event(
                client._config,
                "forget",
                entry_id=memory_id,
                actor="",
                namespace=client._namespace,
                data={"entries_deleted": quarantined_deleted, "quarantined": True},
            )
            quarantined_forget_result: ForgetResultDict = {
                "memory_id": memory_id,
                "status": "deleted",
                "namespace": client._namespace,
                "entries_deleted": quarantined_deleted,
            }
            return quarantined_forget_result
        if existing.namespace != client._namespace:
            raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found in namespace {client._namespace!r}")
        remote_id = existing.remote_id
        backend.delete(memory_id)
        _c.remove_entry_from_tiers(client._config, client._namespace, memory_id)
        append_audit_event(
            client._config,
            "forget",
            entry_id=memory_id,
            actor=existing.source_identity,
            namespace=client._namespace,
            data={"entries_deleted": 1, "quarantined": False},
        )
    if remote_id:
        client._schedule_background_task(client._retire_remote_entry(memory_id, remote_id))

    _client_logger().debug(
        "memory_forgotten",
        op="forget",
        outcome="success",
        memory_id=memory_id,
        namespace=client._namespace,
    )
    forget_result: ForgetResultDict = {
        "memory_id": memory_id,
        "status": "deleted",
        "namespace": client._namespace,
        "entries_deleted": 1,
    }
    return forget_result


async def search_impl(
    client: MemoryClient,
    tags: list[str] | None = None,
    min_importance: float = 0.0,
    since: datetime | None = None,
    limit: int = 50,
    *,
    actor: str | None = None,
    status: str | None = None,
) -> list[MemoryResultDict]:
    """Async impl for :meth:`MemoryClient.search`."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not 0.0 <= min_importance <= 1.0:
        raise ValueError(f"min_importance must be in [0.0, 1.0], got {min_importance}")
    if status is not None and status not in {"active", "resolved", "obsolete", "archived", "quarantined"}:
        raise ValueError(f"status must be one of active/resolved/obsolete/archived/quarantined, got {status!r}")
    client._require_permission(Permission.READ, "search")
    client._maybe_start_retry_drain()
    await client._apply_pending_remote_retirements()

    if status == "quarantined":
        entries = list_quarantined_entries(
            client._config,
            namespace=client._namespace,
            actor=actor,
            limit=max(limit * 5, 10_000) if actor is not None else limit * 5,
        )
    else:
        async with client._lock:
            fetch_limit = limit * 5
            if actor is not None:
                fetch_limit = max(fetch_limit, client._get_backend().count(namespace=client._namespace))
            entries = client._get_backend().list_entries(
                namespace=client._namespace,
                limit=fetch_limit,
            )

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
    _client_logger().debug(
        "memory_searched",
        op="search",
        outcome="success",
        namespace=client._namespace,
        tag_filter=tags,
        min_importance=min_importance,
        result_count=len(final),
    )
    append_audit_event(
        client._config,
        "access",
        actor=actor or "",
        namespace=client._namespace,
        data={
            "entries_returned": len(final),
            "status": status or "",
            "tag_filter": tags or [],
        },
    )
    return final
