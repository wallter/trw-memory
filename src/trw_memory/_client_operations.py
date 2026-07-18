"""Search, deletion, and security-review operations for ``MemoryClient``."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.client import ForgetResultDict, MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)


class ClientOperationsMixin:
    """Cohesive non-recall query and administration surface."""

    _lock: asyncio.Lock
    _config: MemoryConfig
    _namespace: str

    def _get_backend(self) -> StorageBackend:
        raise NotImplementedError

    def _require_permission(self, permission: Permission, operation: str) -> None:
        raise NotImplementedError

    async def forget(self, memory_id: str | None = None, *, actor: str | None = None) -> ForgetResultDict:
        from trw_memory._client_forget_search import forget_impl

        return await forget_impl(cast("MemoryClient", self), memory_id, actor=actor)

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
        from trw_memory._client_forget_search import search_impl

        return await search_impl(
            cast("MemoryClient", self),
            tags=tags,
            min_importance=min_importance,
            since=since,
            limit=limit,
            actor=actor,
            status=status,
        )

    async def search_fts(
        self, query: str, *, top_k: int = 25, min_importance: float = 0.0, status: str | None = None
    ) -> list[MemoryResultDict]:
        from trw_memory._client_distilled_tiering import entry_to_result
        from trw_memory.models.memory import MemoryStatus
        from trw_memory.security.runtime import append_audit_event

        self._require_permission(Permission.READ, "search_fts")
        try:
            parsed_status = MemoryStatus(status) if status is not None else None
        except ValueError:
            logger.warning("search_fts_invalid_status", invalid_status=status)
            return []
        async with self._lock:
            entries = self._get_backend().search_fts(
                query, top_k=top_k, min_importance=min_importance, namespace=self._namespace, status=parsed_status
            )
        results = [entry_to_result(entry, score=entry.importance) for entry in entries]
        append_audit_event(
            self._config,
            "access",
            actor="",
            namespace=self._namespace,
            data={"entries_returned": len(results), "operation": "search_fts"},
        )
        return results

    async def audit_learning(self, learning_id: str) -> dict[str, object]:
        from trw_memory.security.runtime import audit_entry

        self._require_permission(Permission.READ, "audit_learning")
        return audit_entry(
            self._config, learning_id=learning_id, active_backend=self._get_backend(), namespace=self._namespace
        )

    async def review_quarantined(
        self, learning_id: str, *, decision: Literal["approve", "reject"], reviewer_id: str
    ) -> dict[str, str]:
        from trw_memory.security.runtime import review_quarantined_entry

        self._require_permission(Permission.ADMIN, "review_quarantined")
        return review_quarantined_entry(
            self._config,
            active_backend=self._get_backend(),
            learning_id=learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
            namespace=self._namespace,
        )
