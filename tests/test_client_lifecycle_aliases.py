"""Behavior tests for ``LifecycleAliasMixin`` delegation.

``src/trw_memory/_client_lifecycle_aliases.py`` is a thin mixin whose only job
is to forward each public/protected lifecycle method to the matching free
function in ``trw_memory._client_lifecycle``. The contract worth verifying is
the *delegation* itself:

* each alias imports and calls the right ``_client_lifecycle`` function,
* it passes ``self`` (cast to ``MemoryClient``) through ``_as_memory_client()``
  as the first positional argument,
* it forwards any extra arguments unchanged,
* it returns / awaits the impl result.

We compose the mixin onto a tiny stand-in object (so we do not need a real
``MemoryClient``) and monkeypatch the impl functions in
``trw_memory._client_lifecycle`` to record their call args. Because every alias
imports the impl lazily *inside the method body* (``from trw_memory._client_lifecycle import ... as _impl``),
patching the attribute on the module object is observed by the alias.
"""

from __future__ import annotations

import pytest

import trw_memory._client_lifecycle as lifecycle_impl
from trw_memory._client_lifecycle_aliases import LifecycleAliasMixin


class _FakeClient(LifecycleAliasMixin):
    """Minimal object carrying the mixin; stands in for ``MemoryClient``."""


@pytest.fixture
def client() -> _FakeClient:
    return _FakeClient()


# ---------------------------------------------------------------------------
# _as_memory_client — the identity cast every alias relies on
# ---------------------------------------------------------------------------
def test_as_memory_client_returns_self(client: _FakeClient) -> None:
    assert client._as_memory_client() is client


# ---------------------------------------------------------------------------
# Synchronous, value-returning aliases
# ---------------------------------------------------------------------------
def test_should_start_retry_drain_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def fake(c: object) -> bool:
        seen.append(c)
        return True

    monkeypatch.setattr(lifecycle_impl, "should_start_retry_drain", fake)
    result = client._should_start_retry_drain()
    assert result is True
    assert seen == [client]


def test_should_start_sse_subscription_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def fake(c: object) -> bool:
        seen.append(c)
        return False

    monkeypatch.setattr(lifecycle_impl, "should_start_sse_subscription", fake)
    result = client._should_start_sse_subscription()
    assert result is False
    assert seen == [client]


# ---------------------------------------------------------------------------
# Synchronous, side-effecting aliases (return None)
# ---------------------------------------------------------------------------
def test_maybe_start_sse_subscription_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        lifecycle_impl, "maybe_start_sse_subscription", lambda c: calls.append(c)
    )
    assert client._maybe_start_sse_subscription() is None
    assert calls == [client]


def test_maybe_start_retry_drain_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        lifecycle_impl, "maybe_start_retry_drain", lambda c: calls.append(c)
    )
    assert client._maybe_start_retry_drain() is None
    assert calls == [client]


def test_handle_sse_event_forwards_event(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        lifecycle_impl,
        "handle_sse_event",
        lambda c, event: captured.append((c, event)),
    )
    payload = {"type": "learning_published", "id": "abc"}
    client._handle_sse_event(payload)
    assert captured == [(client, payload)]


def test_cache_shared_event_forwards_event(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        lifecycle_impl,
        "cache_shared_event",
        lambda c, event: captured.append((c, event)),
    )
    payload = {"id": "xyz", "summary": "hello"}
    client._cache_shared_event(payload)
    assert captured == [(client, payload)]


# ---------------------------------------------------------------------------
# Async aliases
# ---------------------------------------------------------------------------
async def test_aenter_delegates_and_returns_impl_result(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    seen: list[object] = []

    async def fake(c: object) -> object:
        seen.append(c)
        return sentinel

    monkeypatch.setattr(lifecycle_impl, "aenter", fake)
    result = await client.__aenter__()
    assert result is sentinel
    assert seen == [client]


async def test_aexit_forwards_exception_triple(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[object, ...]] = []

    async def fake(c: object, exc_type: object, exc_val: object, exc_tb: object) -> None:
        captured.append((c, exc_type, exc_val, exc_tb))

    monkeypatch.setattr(lifecycle_impl, "aexit", fake)
    err = ValueError("boom")
    await client.__aexit__(ValueError, err, None)
    assert len(captured) == 1
    c, exc_type, exc_val, _exc_tb = captured[0]
    assert c is client
    assert exc_type is ValueError
    assert exc_val is err


async def test_aexit_passes_none_triple_on_clean_exit(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[object, ...]] = []

    async def fake(c: object, exc_type: object, exc_val: object, exc_tb: object) -> None:
        captured.append((c, exc_type, exc_val, exc_tb))

    monkeypatch.setattr(lifecycle_impl, "aexit", fake)
    await client.__aexit__(None, None, None)
    c, exc_type, exc_val, exc_tb = captured[0]
    assert c is client
    assert exc_type is None
    assert exc_val is None
    assert exc_tb is None


async def test_close_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    async def fake(c: object) -> None:
        seen.append(c)

    monkeypatch.setattr(lifecycle_impl, "close_client", fake)
    assert await client.close() is None
    assert seen == [client]


async def test_drain_retry_queue_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    async def fake(c: object) -> None:
        seen.append(c)

    monkeypatch.setattr(lifecycle_impl, "drain_retry_queue_impl", fake)
    await client._drain_retry_queue()
    assert seen == [client]


async def test_retire_remote_entry_forwards_ids(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[object, str, str]] = []

    async def fake(c: object, memory_id: str, remote_id: str) -> None:
        captured.append((c, memory_id, remote_id))

    monkeypatch.setattr(lifecycle_impl, "retire_remote_entry", fake)
    await client._retire_remote_entry("local-1", "remote-9")
    assert captured == [(client, "local-1", "remote-9")]


async def test_apply_pending_remote_retirements_delegates(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    async def fake(c: object) -> None:
        seen.append(c)

    monkeypatch.setattr(lifecycle_impl, "apply_pending_remote_retirements", fake)
    await client._apply_pending_remote_retirements()
    assert seen == [client]
