"""Context-manager and background lifecycle facade for ``MemoryClient``."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient


class ClientContextMixin:
    """Delegate lifecycle operations to their implementation module."""

    def _client(self) -> MemoryClient:
        return cast("MemoryClient", self)

    def _initialize_resource_state(self) -> None:
        client = self._client()
        client._sse_subscriber = None
        client._sse_subscriber_started = False
        client._tier_manager = None
        client._embedder = None
        client._embedder_initialized = False

    async def __aenter__(self) -> MemoryClient:
        from trw_memory._client_lifecycle import aenter

        return await aenter(self._client())

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        from trw_memory._client_lifecycle import aexit

        await aexit(self._client(), exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        from trw_memory._client_lifecycle import close_client

        await close_client(self._client())

    def _should_start_retry_drain(self) -> bool:
        from trw_memory._client_lifecycle import should_start_retry_drain

        return should_start_retry_drain(self._client())

    def _should_start_sse_subscription(self) -> bool:
        from trw_memory._client_lifecycle import should_start_sse_subscription

        return should_start_sse_subscription(self._client())

    def _maybe_start_sse_subscription(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_sse_subscription

        maybe_start_sse_subscription(self._client())

    def _maybe_start_retry_drain(self) -> None:
        from trw_memory._client_lifecycle import maybe_start_retry_drain

        maybe_start_retry_drain(self._client())

    def _handle_sse_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import handle_sse_event

        handle_sse_event(self._client(), event)

    def _cache_shared_event(self, event: dict[str, object]) -> None:
        from trw_memory._client_lifecycle import cache_shared_event

        cache_shared_event(self._client(), event)

    async def _drain_retry_queue(self) -> None:
        from trw_memory._client_lifecycle import drain_retry_queue_impl

        await drain_retry_queue_impl(self._client())

    async def _retire_remote_entry(self, memory_id: str, remote_id: str) -> None:
        from trw_memory._client_lifecycle import retire_remote_entry

        await retire_remote_entry(self._client(), memory_id, remote_id)

    async def _apply_pending_remote_retirements(self) -> None:
        from trw_memory._client_lifecycle import apply_pending_remote_retirements

        await apply_pending_remote_retirements(self._client())
