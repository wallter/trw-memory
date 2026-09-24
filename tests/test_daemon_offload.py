"""PRD-CORE-279 FR04/NFR02/NFR04: served tool bodies run off the event loop.

Every test here carries a deadline. Without one, a regression that puts the
work back on the event loop would not fail these tests -- it would hang them,
and a hung suite reads as "still running" rather than "broken".
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from trw_memory.daemon._offload import (
    OFFLOAD_MAX_WORKERS,
    run_offloaded,
    shutdown_offload_pool,
)

_DEADLINE = 15.0


@pytest.fixture(autouse=True)
def _fresh_pool():
    shutdown_offload_pool()
    yield
    shutdown_offload_pool()


async def test_offload_returns_the_callables_result():
    """NFR02: the helper is transport-neutral -- same callable, same payload."""
    payload = {"memories": [{"id": "a"}], "total_matches": 1}

    async def _call() -> dict[str, object]:
        return await run_offloaded(lambda: payload)

    assert await asyncio.wait_for(_call(), timeout=_DEADLINE) is payload


async def test_offload_propagates_exceptions():
    """A failure in the worker reaches the caller unchanged."""

    def _boom() -> None:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        await asyncio.wait_for(run_offloaded(_boom), timeout=_DEADLINE)


async def test_offload_runs_on_a_worker_thread():
    """FR04: the body does not execute on the loop's thread."""
    loop_thread = threading.current_thread().name

    name = await asyncio.wait_for(
        run_offloaded(lambda: threading.current_thread().name),
        timeout=_DEADLINE,
    )
    assert name != loop_thread
    assert name.startswith("trw-memory-tool")


async def test_recall_runs_off_the_event_loop():
    """FR04: two calls overlap. On the old serial path the second never starts."""
    started = threading.Event()
    release = threading.Event()

    def _blocking() -> str:
        started.set()
        assert release.wait(timeout=_DEADLINE), "the second call never reached the worker pool"
        return "first"

    first = asyncio.ensure_future(run_offloaded(_blocking))
    # Yield to the loop so the first call is submitted before the second.
    await asyncio.sleep(0)
    assert started.wait(timeout=_DEADLINE), "the first call never started: it is still on the loop"

    second = await asyncio.wait_for(run_offloaded(lambda: "second"), timeout=_DEADLINE)
    assert second == "second", "a second call could not run while the first was busy"
    release.set()
    assert await asyncio.wait_for(first, timeout=_DEADLINE) == "first"


async def test_offload_is_bounded_by_the_worker_count():
    """NFR04: never more than OFFLOAD_MAX_WORKERS callables run at once."""
    lock = threading.Lock()
    in_flight = 0
    peak = 0
    release = threading.Event()

    def _occupy() -> None:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        release.wait(timeout=_DEADLINE)
        with lock:
            in_flight -= 1

    tasks = [asyncio.ensure_future(run_offloaded(_occupy)) for _ in range(OFFLOAD_MAX_WORKERS + 3)]
    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        with lock:
            if in_flight >= OFFLOAD_MAX_WORKERS:
                break
    with lock:
        assert in_flight == OFFLOAD_MAX_WORKERS, f"expected the pool to be saturated, saw {in_flight}"
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=_DEADLINE)
    assert peak <= OFFLOAD_MAX_WORKERS, f"pool ran {peak} callables at once"


async def test_offload_carries_the_caller_context():
    """Log context bound by the caller must reach the worker, and not leak."""
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")

    marker.set("request-a")
    seen_a = await asyncio.wait_for(run_offloaded(marker.get), timeout=_DEADLINE)
    assert seen_a == "request-a"

    async def _other_request() -> str:
        marker.set("request-b")
        return await run_offloaded(marker.get)

    seen_b = await asyncio.wait_for(asyncio.ensure_future(_other_request()), timeout=_DEADLINE)
    assert seen_b == "request-b"
    # The worker that ran request-b must not still be carrying it.
    assert await asyncio.wait_for(run_offloaded(marker.get), timeout=_DEADLINE) == "request-a"


async def test_pool_is_recreated_after_shutdown():
    """Shutdown is not terminal: the next call gets a working pool."""
    assert await asyncio.wait_for(run_offloaded(lambda: 1), timeout=_DEADLINE) == 1
    shutdown_offload_pool()
    assert await asyncio.wait_for(run_offloaded(lambda: 2), timeout=_DEADLINE) == 2


async def test_registered_recall_tool_offloads(tmp_path, monkeypatch):
    """FR04 integration: the registered memory_recall body leaves the loop."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)

    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_recall")
    loop_thread = threading.current_thread().name
    observed: list[str] = []

    import trw_memory.tools.recall as recall_mod

    original = recall_mod.memory_recall_impl

    def _spy(*args, **kwargs):
        observed.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(recall_mod, "memory_recall_impl", _spy)

    await asyncio.wait_for(
        tool.run({"query": "anything", "namespace": "project:default", "include_org_memories": False}),
        timeout=60.0,
    )
    assert observed, "the impl never ran"
    assert observed[0] != loop_thread
    assert observed[0].startswith("trw-memory-tool")


async def test_registered_store_tool_offloads(tmp_path, monkeypatch):
    """FR04 integration: the registered memory_store body leaves the loop too."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)

    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_store")
    loop_thread = threading.current_thread().name
    observed: list[str] = []

    import trw_memory.tools.store as store_mod

    original = store_mod.memory_store_impl

    def _spy(*args, **kwargs):
        observed.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(store_mod, "memory_store_impl", _spy)

    await asyncio.wait_for(
        tool.run({"content": "a fact stored off the loop", "namespace": "project:default"}),
        timeout=60.0,
    )
    assert observed, "the impl never ran"
    assert observed[0] != loop_thread
    assert observed[0].startswith("trw-memory-tool")


async def test_shutdown_is_bounded_when_a_worker_will_not_stop():
    """NFR04: a stuck call must not hold shutdown open indefinitely."""
    release = threading.Event()
    started = threading.Event()

    def _stuck() -> None:
        started.set()
        release.wait(timeout=30)

    task = asyncio.ensure_future(run_offloaded(_stuck))
    await asyncio.sleep(0)
    assert started.wait(timeout=_DEADLINE)

    began = time.monotonic()
    drained = await asyncio.to_thread(shutdown_offload_pool, timeout=0.5)
    elapsed = time.monotonic() - began

    assert drained is False, "shutdown claimed a clean drain while a call was stuck"
    assert elapsed < 5.0, f"shutdown waited {elapsed:.1f}s despite a 0.5s grace"
    release.set()
    await asyncio.wait_for(task, timeout=_DEADLINE)
