"""Failure-atomicity regressions for MemoryClient lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory._client_lifecycle import (
    apply_pending_remote_retirements,
    close_client,
    drain_retry_queue_impl,
    init_client,
    schedule_background_task,
)
from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryConnectionError, SecurityDefaultUnresolvableError


@pytest.mark.parametrize("mode", ["local", "auto"])
def test_post_open_init_failure_closes_backend(tmp_path: Path, mode: str) -> None:
    client = MemoryClient.__new__(MemoryClient)
    backend = MagicMock()

    with (
        patch("trw_memory._client_lifecycle._create_local_backend", return_value=backend),
        patch("trw_memory._client_lifecycle.verify_defaults", side_effect=ValueError("invalid defaults")),
        pytest.raises(MemoryConnectionError),
    ):
        init_client(client, "default", mode=mode, db_path=tmp_path / "memory.db")

    backend.close.assert_called_once_with()
    assert client._backend is None


def test_security_init_failure_remains_fail_loud_after_cleanup(tmp_path: Path) -> None:
    client = MemoryClient.__new__(MemoryClient)
    backend = MagicMock()

    with (
        patch("trw_memory._client_lifecycle._create_local_backend", return_value=backend),
        patch(
            "trw_memory._client_lifecycle.verify_defaults",
            side_effect=SecurityDefaultUnresolvableError("missing security default"),
        ),
        pytest.raises(SecurityDefaultUnresolvableError, match="missing security default"),
    ):
        init_client(client, "default", mode="auto", db_path=tmp_path / "memory.db")

    backend.close.assert_called_once_with()
    assert client._backend is None


def test_backend_close_failure_does_not_mask_init_error(tmp_path: Path) -> None:
    client = MemoryClient.__new__(MemoryClient)
    backend = MagicMock()
    backend.close.side_effect = OSError("close failed")

    with (
        patch("trw_memory._client_lifecycle._create_local_backend", return_value=backend),
        patch("trw_memory._client_lifecycle.verify_defaults", side_effect=ValueError("invalid defaults")),
        pytest.raises(MemoryConnectionError, match="invalid defaults"),
    ):
        init_client(client, "default", mode="local", db_path=tmp_path / "memory.db")

    assert client._backend is None


def test_unexpected_auto_init_failure_is_not_masked(tmp_path: Path) -> None:
    client = MemoryClient.__new__(MemoryClient)
    backend = MagicMock()

    with (
        patch("trw_memory._client_lifecycle._create_local_backend", return_value=backend),
        patch("trw_memory._client_lifecycle.verify_defaults", side_effect=RuntimeError("programming defect")),
        pytest.raises(RuntimeError, match="programming defect"),
    ):
        init_client(client, "default", mode="auto", db_path=tmp_path / "memory.db")

    backend.close.assert_called_once_with()
    assert client._backend is None


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_backend_open(timeout: float) -> None:
    client = MemoryClient.__new__(MemoryClient)

    with (
        patch("trw_memory._client_lifecycle._create_local_backend") as create_backend,
        pytest.raises(ValueError, match="timeout"),
    ):
        init_client(client, "default", mode="local", timeout=timeout)

    create_backend.assert_not_called()


@pytest.mark.asyncio
async def test_retry_drain_failure_resets_restart_flag() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._retry_drain_started = True
    client._retry_queue = MagicMock()
    client._config = MagicMock()

    with (
        patch(
            "trw_memory.sync._remote_publish._drain_retry_queue_with_ids",
            side_effect=RuntimeError("remote unavailable"),
        ),
        pytest.raises(RuntimeError, match="remote unavailable"),
    ):
        await drain_retry_queue_impl(client)

    assert client._retry_drain_started is False


@pytest.mark.asyncio
async def test_retry_drain_cancellation_resets_restart_flag() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._retry_drain_started = True
    client._retry_queue = MagicMock()
    client._config = MagicMock()

    with (
        patch("trw_memory._client_lifecycle.asyncio.to_thread", side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await drain_retry_queue_impl(client)

    assert client._retry_drain_started is False


@pytest.mark.asyncio
async def test_skipped_retry_is_not_marked_published() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._retry_drain_started = True
    client._retry_queue = MagicMock()
    client._config = MagicMock()
    client._lock = asyncio.Lock()
    client._namespace = "default"
    backend = MagicMock()
    client._get_backend = MagicMock(return_value=backend)  # type: ignore[method-assign]
    result = {
        "drained": 0,
        "failed": 0,
        "skipped": 1,
        "remote_ids": {},
    }

    with patch("trw_memory.sync._remote_publish._drain_retry_queue_with_ids", return_value=(result, [])):
        await drain_retry_queue_impl(client)

    backend.update.assert_not_called()
    assert client._retry_drain_started is False


@pytest.mark.asyncio
async def test_close_releases_backend_when_cancelled_during_task_drain() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._sse_subscriber = None
    backend = MagicMock()
    client._backend = backend
    client._namespace = "default"
    pending = asyncio.create_task(asyncio.Event().wait())
    client._background_tasks = {pending}
    close_task = asyncio.create_task(close_client(client))
    await asyncio.sleep(0)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    backend.close.assert_called_once_with()
    assert client._backend is None


@pytest.mark.asyncio
async def test_close_preserves_cancellation_when_backend_close_fails() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._sse_subscriber = None
    backend = MagicMock()
    backend.close.side_effect = OSError("close failed")
    client._backend = backend
    client._namespace = "default"
    pending = asyncio.create_task(asyncio.Event().wait())
    client._background_tasks = {pending}
    close_task = asyncio.create_task(close_client(client))
    await asyncio.sleep(0)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    backend.close.assert_called_once_with()
    assert client._backend is None


@pytest.mark.asyncio
async def test_pending_retirements_are_requeued_after_backend_failure() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._pending_remote_retirements = {"remote-1"}
    client._pending_remote_retirements_lock = threading.Lock()
    client._lock = asyncio.Lock()
    client._namespace = "default"
    backend = MagicMock()
    backend.count.side_effect = RuntimeError("database unavailable")
    client._get_backend = MagicMock(return_value=backend)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="database unavailable"):
        await apply_pending_remote_retirements(client)

    assert client._pending_remote_retirements == {"remote-1"}


@pytest.mark.asyncio
async def test_completed_retirements_are_not_requeued_after_partial_failure() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._pending_remote_retirements = {"remote-1", "remote-2"}
    client._pending_remote_retirements_lock = threading.Lock()
    client._lock = asyncio.Lock()
    client._namespace = "default"
    first = MagicMock(id="local-1", remote_id="remote-1", last_synced_at=object())
    second = MagicMock(id="local-2", remote_id="remote-2", last_synced_at=object())
    backend = MagicMock()
    backend.count.return_value = 2
    backend.list_entries.return_value = [first, second]
    backend.update.side_effect = [first, RuntimeError("database unavailable")]
    client._get_backend = MagicMock(return_value=backend)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="database unavailable"):
        await apply_pending_remote_retirements(client)

    assert client._pending_remote_retirements == {"remote-2"}


@pytest.mark.asyncio
async def test_background_task_failure_is_observed() -> None:
    client = MemoryClient.__new__(MemoryClient)
    client._background_tasks = set()
    logger = MagicMock()

    async def fail() -> None:
        raise RuntimeError("background failed")

    with patch("trw_memory._client_lifecycle._client_logger", return_value=logger):
        schedule_background_task(client, fail())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert client._background_tasks == set()
    logger.warning.assert_called_once_with("memory_background_task_failed", exc_info=True)
