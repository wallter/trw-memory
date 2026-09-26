"""SSE subscription and the shared-event cache it feeds.

Belongs to the ``_client_lifecycle.py`` facade, which re-exports every name
here so ``trw_memory._client_lifecycle.handle_sse_event`` keeps working for
``_client_context.py`` and for the three test modules that import it directly.

Extracted for the 350 effective-LOC gate: ``_client_lifecycle.py`` measured
370, and this is the largest group in it that nothing else in the module
calls — the retry-drain and remote-retirement groups both go through
``schedule_background_task``, while the SSE group's only inbound edge is the
``MemoryClient`` facade.

``SHARED_EVENT_CACHE_MAX`` stays defined in ``_client_lifecycle`` rather than
moving here: ``client.py`` re-exports it under an explicit
``X as X`` binding that ``test_client_recall_sync.py`` imports, and moving a
public constant's definition across modules for a size gate is a distribution
change, not a structural one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient

__all__ = [
    "cache_shared_event",
    "handle_sse_event",
    "maybe_start_sse_subscription",
    "should_start_sse_subscription",
]


def should_start_sse_subscription(client: MemoryClient) -> bool:
    return (
        not client._sse_subscriber_started
        and client._config.sync_enabled
        and bool(client._config.platform_url)
        and bool(client._config.platform_api_key)
    )


def maybe_start_sse_subscription(client: MemoryClient) -> None:
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


def handle_sse_event(client: MemoryClient, event: dict[str, object]) -> None:
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


def cache_shared_event(client: MemoryClient, event: dict[str, object]) -> None:
    from trw_memory._client_lifecycle import SHARED_EVENT_CACHE_MAX
    from trw_memory._client_org_shared import shared_result_to_result

    remote_id = str(event.get("id", "")).strip()
    summary = str(event.get("summary", "")).strip()
    if not remote_id or not summary:
        return
    shared_content = summary if summary.startswith("[shared] ") else f"[shared] {summary}"
    payload: dict[str, object] = {
        **event,
        "memory_id": remote_id,
        "content": shared_content,
        "namespace": "shared",
        "source": "shared",
    }
    cached = shared_result_to_result(payload)
    with client._shared_event_cache_lock:
        client._shared_event_cache = [
            existing for existing in client._shared_event_cache if existing["memory_id"] != remote_id
        ]
        client._shared_event_cache.append(cached)
        if len(client._shared_event_cache) > SHARED_EVENT_CACHE_MAX:
            client._shared_event_cache = client._shared_event_cache[-SHARED_EVENT_CACHE_MAX:]
