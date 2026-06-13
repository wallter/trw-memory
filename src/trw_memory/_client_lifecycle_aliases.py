"""Lifecycle alias mixin for MemoryClient facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient

__all__ = ["LifecycleAliasMixin"]


class LifecycleAliasMixin:
    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    # ---- Lifecycle aliases (PRD-DIST-246 batches 111+112) -----------------

    def _as_memory_client(self) -> MemoryClient:
        return cast("MemoryClient", self)

    async def __aenter__(self) -> MemoryClient:
        from trw_memory._client_lifecycle import aenter as _impl

        return await _impl(self._as_memory_client())

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        from trw_memory._client_lifecycle import aexit as _impl

        await _impl(self._as_memory_client(), exc_type, exc_val, cast("Any", exc_tb))

    async def close(self) -> None:
        from trw_memory._client_lifecycle import close_client as _impl

        await _impl(self._as_memory_client())

    def _should_start_retry_drain(self) -> bool:
        from trw_memory._client_lifecycle import should_start_retry_drain as _impl

        return _impl(self._as_memory_client())

    def _should_start_sse_subscription(self) -> bool:
        from trw_memory._client_lifecycle import should_start_sse_subscription as _impl

        return _impl(self._as_memory_client())

    def _maybe_start_sse_subscription(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_sse_subscription as _impl

        _impl(self._as_memory_client())

    def _maybe_start_retry_drain(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_retry_drain as _impl

        _impl(self._as_memory_client())

    def _handle_sse_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import handle_sse_event as _impl

        _impl(self._as_memory_client(), event)

    def _cache_shared_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import cache_shared_event as _impl

        _impl(self._as_memory_client(), event)

    async def _drain_retry_queue(self) -> None:
        from trw_memory._client_lifecycle import drain_retry_queue_impl as _impl

        await _impl(self._as_memory_client())

    async def _retire_remote_entry(self, memory_id: str, remote_id: str) -> None:
        from trw_memory._client_lifecycle import retire_remote_entry as _impl

        await _impl(self._as_memory_client(), memory_id, remote_id)

    async def _apply_pending_remote_retirements(self) -> None:
        from trw_memory._client_lifecycle import apply_pending_remote_retirements as _impl

        await _impl(self._as_memory_client())
