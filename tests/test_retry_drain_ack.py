"""The retry drain marks a row synced only while it is still the revision that was queued (B71-77).

A publish that fails is queued and re-sent later. The drain used to stamp ``last_synced_at`` on every row it
re-sent, so an edit made while the record waited in the queue was marked synced and never pushed. The record
now carries the row's ``(sync_seq, sync_hash)`` from enqueue time, and the drain acks through the same
conditional path as a direct publish (``sync.delta.ack_revision``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry
from trw_memory.sync.retry_queue import RetryQueue


def _sync_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ctx"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")


@contextmanager
def _platform_accepts() -> Iterator[None]:
    with patch("trw_memory.sync.remote.httpx.Client") as client_cls:
        client = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": "42"}
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client_cls.return_value = client
        yield


async def _queue_one(entry_id: str) -> None:
    """Store *entry_id* with the platform refusing, so its publish lands in the retry queue."""
    seed = MemoryClient(namespace="default", mode="local")
    with (
        patch(
            "trw_memory.client.publish_memory_result",
            return_value={"success": False, "remote_id": None, "retryable": True},
        ),
        patch("trw_memory.client._anonymize_entry", return_value={"summary": "queued", "source_learning_id": entry_id}),
    ):
        await seed.store("queue this entry", importance=0.9, entry_id=entry_id)
        await seed.close()


async def _drain() -> MemoryEntry:
    with _platform_accepts():
        async with MemoryClient(namespace="default", mode="local"):
            pass
    reopened = MemoryClient(namespace="default", mode="local")
    row = reopened._get_backend().list_entries(limit=10)[0]
    assert reopened._retry_queue.depth() == 0, "the drain did not publish the queued record"
    await reopened.close()
    return row


async def test_an_edit_made_while_queued_stays_dirty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("queued-entry")
    editor = MemoryClient(namespace="default", mode="local")
    backend = editor._get_backend()
    queued = backend.get("queued-entry", namespace="default")
    assert queued is not None
    backend.update(
        "queued-entry",
        namespace="default",
        content="edited while queued",
        sync_seq=queued.sync_seq + 1,
        sync_hash=f"{queued.sync_hash}-edited",
    )
    await editor.close()

    row = await _drain()

    assert row.published_to_platform is True and row.remote_id == "42"
    assert row.last_synced_at is None, "the drain marked an edit it never published as synced"


async def test_an_unedited_queued_row_is_marked_synced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("queued-entry")

    row = await _drain()

    assert row.published_to_platform is True and row.remote_id == "42"
    assert row.last_synced_at is not None


async def test_two_queued_revisions_mark_the_row_synced_at_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sol r1 P2: acking the stale record first renumbered the row, so the current record's ack failed."""
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("queued-entry")
    editor = MemoryClient(namespace="default", mode="local")
    backend = editor._get_backend()
    first = backend.get("queued-entry", namespace="default")
    assert first is not None
    backend.update("queued-entry", namespace="default", content="edited", sync_seq=first.sync_seq + 1, sync_hash="h2")
    current = backend.get("queued-entry", namespace="default")
    assert current is not None
    editor._retry_queue.enqueue(
        "queued-entry",
        {"summary": "edited", "source_learning_id": "queued-entry"},
        revision=(current.sync_seq, current.sync_hash),
    )
    await editor.close()

    row = await _drain()

    assert row.last_synced_at is not None, "the newest queued revision was published but the row stayed dirty"


async def test_a_later_success_without_an_id_keeps_the_earlier_remote_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sol r2 P2: coalescing per entry must keep the last remote id the platform returned."""
    from trw_memory import _client_lifecycle

    _sync_env(tmp_path, monkeypatch)
    client = MemoryClient(namespace="default", mode="local")
    await client.store("row", importance=0.9, entry_id="e")
    row = client._get_backend().get("e", namespace="default")
    assert row is not None
    published = [("e", (row.sync_seq - 1, "old"), "42"), ("e", (row.sync_seq, row.sync_hash), None)]
    monkeypatch.setattr(
        "trw_memory.sync._remote_publish._drain_retry_queue_with_ids",
        lambda _queue, _cfg: ({"drained": 2, "failed": 0, "skipped": 0, "remote_ids": {"e": "42"}}, published),
    )

    await _client_lifecycle._drain_retry_queue_once(client)

    acked = client._get_backend().get("e", namespace="default")
    assert acked is not None and acked.remote_id == "42" and acked.last_synced_at is not None
    await client.close()


def test_a_queued_record_carries_its_revision(tmp_path: Path) -> None:
    queue = RetryQueue(tmp_path / "sync_queue.jsonl")
    queue.enqueue("with", {"source_learning_id": "with"}, revision=(3, "abc"))
    queue.enqueue("without", {"source_learning_id": "without"})

    _, drained = queue._drain_with_ids(lambda _payload: True)

    assert drained == [("with", (3, "abc")), ("without", None)]


def test_a_record_queued_before_revisions_existed_never_stamps(tmp_path: Path) -> None:
    """An old queue file (no ``revision`` key) still drains, and its rows stay dirty for the next push."""
    path = tmp_path / "sync_queue.jsonl"
    old = {
        "entry_id": "e",
        "payload": {},
        "queued_at": "2026-01-01T00:00:00+00:00",
        "retry_count": 0,
        "last_error": None,
    }
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")

    _, drained = RetryQueue(path)._drain_with_ids(lambda _payload: True)

    assert drained == [("e", None)]


@pytest.mark.parametrize("backend_kind", ["sqlite", "yaml"])
def test_ack_revision_stamps_only_a_matching_row_on_a_transactional_backend(backend_kind: str, tmp_path: Path) -> None:
    from trw_memory.storage import SQLiteBackend, YAMLBackend
    from trw_memory.sync.delta import ack_revision

    backend = SQLiteBackend(tmp_path / "m.db") if backend_kind == "sqlite" else YAMLBackend(tmp_path / "yaml")
    backend.store(MemoryEntry(id="e", content="c", namespace="default"))
    stored = backend.get("e", namespace="default")
    assert stored is not None
    revision = (stored.sync_seq, stored.sync_hash)  # store() numbers the revision itself

    assert ack_revision(backend, "e", "default", (revision[0] + 1, revision[1]), published_to_platform=True) is False
    assert (
        ack_revision(backend, "e", "default", (revision[0], f"{revision[1]}-other"), published_to_platform=True)
        is False
    )
    assert ack_revision(backend, "e", "default", None, published_to_platform=True) is False
    # Each ack's update() numbers a new revision, so the matching case uses the row as it is now.
    current = backend.get("e", namespace="default")
    assert current is not None
    stamped = ack_revision(backend, "e", "default", (current.sync_seq, current.sync_hash), published_to_platform=True)

    row = backend.get("e", namespace="default")
    assert row is not None and row.published_to_platform is True
    # The YAML store has no transaction, so its compare-then-write is never trusted to stamp.
    assert stamped is (backend_kind == "sqlite")
    assert (row.last_synced_at is not None) is (backend_kind == "sqlite")
