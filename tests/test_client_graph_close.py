"""Owner-scoped graph teardown cannot race successful client close."""

import asyncio
import shutil
import threading
from pathlib import Path

import pytest

import trw_memory.graph as graph
from trw_memory._graph_threads import _GraphThreadRegistry
from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryConnectionError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def test_registered_but_unstarted_worker_is_not_reported_finished() -> None:
    registry = _GraphThreadRegistry()
    owner = object()
    thread = threading.Thread(target=lambda: None)
    registry.track(thread, owner)
    try:
        with pytest.raises(TimeoutError):
            registry.wait(timeout=0.001, owner=owner)
        registry.wait(timeout=0.001, owner=object())
    finally:
        registry.untrack(thread)


@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_drains_before_directory_removal(
    client: MemoryClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_close: bool,
) -> None:
    entered, release = threading.Event(), threading.Event()
    actual_worker = graph._run_scheduled_graph_update

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)
        actual_worker(entry, config, embedding)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    await client.__aenter__()
    close: asyncio.Task[None] | None = None
    concurrent_close: asyncio.Task[None] | None = None
    try:
        await client.store("A local graph lifecycle regression", tags=["lifecycle"])
        assert await asyncio.to_thread(entered.wait, 1)
        close = asyncio.create_task(client.close())
        await asyncio.sleep(0.03)
        assert not close.done()
        concurrent_close = asyncio.create_task(client.close())
        await asyncio.sleep(0.03)
        assert not concurrent_close.done()
        with pytest.raises(MemoryConnectionError, match="close is incomplete"):
            await client.__aenter__()
        if cancel_close:
            close.cancel()
            await asyncio.sleep(0.03)
            assert not close.done()
        release.set()
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                await close
        else:
            await close
        await concurrent_close
        root = tmp_path / "storage"
        shutil.rmtree(root)
        await asyncio.to_thread(graph.wait_for_graph_updates)
        assert not root.exists()
    finally:
        release.set()
        if close is not None and not close.done():
            await asyncio.gather(close, return_exceptions=True)
        if concurrent_close is not None and not concurrent_close.done():
            await asyncio.gather(concurrent_close, return_exceptions=True)
        await client.close()
        await asyncio.to_thread(graph.wait_for_graph_updates)


async def test_close_does_not_wait_for_other_backend(
    client: MemoryClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    await client.__aenter__()
    with SQLiteBackend(tmp_path / "other.sqlite") as other:
        try:
            assert graph.schedule_graph_update(MemoryEntry(id="M-other", content="other"), other)
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(client.close(), timeout=0.5)
            assert not release.is_set()
        finally:
            release.set()
            await asyncio.to_thread(graph.wait_for_graph_updates)
            await client.close()


async def test_owner_timeout_is_not_successful_close(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    original_wait = graph.wait_for_graph_updates

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)

    def short_wait(*, owner: object) -> None:
        original_wait(timeout=0.01, owner=owner)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    await client.__aenter__()
    try:
        await client.store("timeout lifecycle regression")
        assert await asyncio.to_thread(entered.wait, 1)
        monkeypatch.setattr(graph, "wait_for_graph_updates", short_wait)
        with pytest.raises(TimeoutError, match="background graph"):
            await client.close()
        with pytest.raises(MemoryConnectionError, match="closed"):
            await client.store("must not reuse a partially closed backend")
        with pytest.raises(TimeoutError, match="background graph"):
            await client.close()
    finally:
        release.set()
        await asyncio.to_thread(original_wait)
        await client.close()


async def test_backend_close_failure_keeps_retry_handle(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    await client.__aenter__()
    backend = client._get_backend()
    original_close = backend.close
    attempts = 0

    def fail_first_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected backend close failure")
        original_close()

    monkeypatch.setattr(backend, "close", fail_first_close)
    try:
        with pytest.raises(OSError, match="backend close"):
            await client.close()
        assert client._backend is None
        assert client._pending_close_backend is backend
        await client.close()
        assert client._pending_close_backend is None
        assert attempts == 2
    finally:
        original_close()
