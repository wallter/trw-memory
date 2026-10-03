"""SYNC-RETRY-LABEL-RECHECK: a queued retry asks the confidentiality label again immediately before its POST.

``publish_memory_result`` refuses a row labelled above team (PRD-SEC-023 FR05), but a publish that failed is queued
as an anonymized payload and the retry drain re-sent that payload with no label check: a row relabelled
``sensitive`` (or deleted) while its record waited still left the host. The drain now reads each queued row as it is
now -- by its namespace-qualified identity, immediately before its POST -- and asks the label policy in force then; a
row it cannot read, or a payload with no consistent identity, is withheld.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.egress_gate import identity_keys
from trw_memory.sync import _remote_publish
from trw_memory.sync.retry_queue import MAX_RETRIES, RetryQueue


def _sync_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "ctx"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")


@contextmanager
def _platform_accepts() -> Iterator[MagicMock]:
    with patch("trw_memory.sync.remote.httpx.Client") as client_cls:
        client = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": "42"}
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client_cls.return_value = client
        yield client.post


async def _queue_one(entry_id: str) -> None:
    """Store *entry_id* with the platform refusing, so its publish lands in the retry queue."""
    seed = MemoryClient(namespace="default", mode="local")
    with patch(
        "trw_memory.client.publish_memory_result",
        return_value={"success": False, "remote_id": None, "retryable": True},
    ):
        await seed.store("queue this entry", importance=0.9, entry_id=entry_id)
    await seed.close()
    assert seed._retry_queue.depth() == 1, "the failed publish was not queued"


async def _drain_posts() -> MagicMock:
    with _platform_accepts() as post:
        async with MemoryClient(namespace="default", mode="local"):
            pass
    return post


async def test_a_row_relabelled_sensitive_while_queued_is_never_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("L-relabel")
    editor = MemoryClient(namespace="default", mode="local")
    backend = editor._get_backend()
    row = backend.get("L-relabel", namespace="default")
    assert row is not None
    backend.update("L-relabel", namespace="default", metadata={**row.metadata, "trw_label": "sensitive"})
    await editor.close()

    post = await _drain_posts()

    post.assert_not_called()
    reopened = MemoryClient(namespace="default", mode="local")
    after = reopened._get_backend().get("L-relabel", namespace="default")
    assert after is not None and not after.published_to_platform
    await reopened.close()


async def test_a_row_deleted_while_queued_is_never_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("L-gone")
    editor = MemoryClient(namespace="default", mode="local")
    assert editor._get_backend().delete("L-gone", namespace="default")
    await editor.close()

    post = await _drain_posts()

    post.assert_not_called()


async def test_a_team_row_still_admitted_is_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_env(tmp_path, monkeypatch)
    await _queue_one("L-team")

    post = await _drain_posts()

    post.assert_called_once()
    assert post.call_args.kwargs["json"]["source_learning_id"] == "L-team"


async def test_another_namespaces_queued_row_is_left_untouched_and_its_own_drain_sends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue is shared by every namespace of a storage root, but each namespace has its own store: a drain started
    by another namespace's client leaves the record as it was (no retry spent), and the owner's drain sends it (codex r1)."""
    _sync_env(tmp_path, monkeypatch)
    seed = MemoryClient(namespace="project:alpha", mode="local")
    with patch(
        "trw_memory.client.publish_memory_result",
        return_value={"success": False, "remote_id": None, "retryable": True},
    ):
        await seed.store("queued by alpha", importance=0.9, entry_id="L-alpha")
    await seed.close()

    queue_file = tmp_path / "ctx" / "sync_queue.jsonl"
    before = queue_file.read_text()

    with _platform_accepts() as post:
        async with MemoryClient(namespace="project:beta", mode="local"):
            pass
    post.assert_not_called()
    assert queue_file.read_text() == before, "another namespace's drain spent a retry on alpha's record"

    with _platform_accepts() as post:
        async with MemoryClient(namespace="project:alpha", mode="local"):
            pass
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["source_learning_id"] == "L-alpha"


def _queued(entry: MemoryEntry, queue: RetryQueue, *, retry_count: int = 0) -> None:
    payload = dict(_remote_publish._anonymize_entry(entry))
    payload[_remote_publish.LEDGER_KEYS_FIELD] = identity_keys(entry)
    assert queue.enqueue(entry.id, payload)
    if retry_count:
        records = [json.loads(line) for line in queue._path.read_text().splitlines()]
        records[-1]["retry_count"] = retry_count
        queue._path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    monkeypatch.setattr(_remote_publish, "platform_contact_blocked", lambda *_a, **_k: None)
    with patch("trw_memory.sync._remote_publish.httpx.Client") as client_cls:
        response = MagicMock(status_code=201)
        response.json.return_value = {"id": "R-1"}
        client_cls.return_value.__enter__.return_value.post.return_value = response
        yield client_cls.return_value.__enter__.return_value.post


def _cfg(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "memory"), platform_url="https://platform.invalid")


def test_a_same_id_team_row_in_another_namespace_never_admits_a_sensitive_payload(
    tmp_path: Path, posts: MagicMock
) -> None:
    """Admission is bound to the queued record's namespace-qualified identity, not its bare id (codex r1)."""
    queued = MemoryEntry(id="shared", namespace="project:a", content="a's text", importance=0.9)
    queue = RetryQueue(tmp_path / "queue.jsonl")
    _queued(queued, queue)
    rows = {
        ("project:a", "shared"): queued.model_copy(update={"metadata": {"trw_label": "sensitive"}}),
        ("project:b", "shared"): MemoryEntry(id="shared", namespace="project:b", content="b's text", importance=0.9),
    }

    _remote_publish._drain_retry_queue_with_ids(queue, _cfg(tmp_path), current_row=lambda ns, i: rows.get((ns, i)))

    posts.assert_not_called()


def test_a_row_relabelled_during_the_backoff_is_never_sent(
    tmp_path: Path, posts: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is read immediately before its POST, after the retry backoff, not once before the drain (codex r1)."""
    entry = MemoryEntry(id="L-1", namespace="project:a", content="a learning", importance=0.9)
    queue = RetryQueue(tmp_path / "queue.jsonl")
    _queued(entry, queue, retry_count=1)
    rows = {(entry.namespace, entry.id): entry}

    def relabel_while_waiting(_seconds: float) -> None:
        rows[(entry.namespace, entry.id)] = entry.model_copy(update={"metadata": {"trw_label": "sensitive"}})

    monkeypatch.setattr("trw_memory.sync.retry_queue.time.sleep", relabel_while_waiting)

    _remote_publish._drain_retry_queue_with_ids(queue, _cfg(tmp_path), current_row=lambda ns, i: rows.get((ns, i)))

    posts.assert_not_called()


def test_a_payload_whose_identity_names_another_id_is_refused(tmp_path: Path, posts: MagicMock) -> None:
    entry = MemoryEntry(id="L-1", namespace="project:a", content="a learning", importance=0.9)
    queue = RetryQueue(tmp_path / "queue.jsonl")
    payload = dict(_remote_publish._anonymize_entry(entry))
    payload[_remote_publish.LEDGER_KEYS_FIELD] = [["id", "project:a", "L-other"]]
    assert queue.enqueue(entry.id, payload)

    _remote_publish._drain_retry_queue_with_ids(queue, _cfg(tmp_path), current_row=lambda _ns, _i: entry)

    posts.assert_not_called()


def test_an_exhausted_record_of_another_namespace_is_still_evicted(tmp_path: Path, posts: MagicMock) -> None:
    """Exhaustion is checked before ownership: a retired namespace's dead records never hold the shared queue's
    capacity (codex r2)."""
    foreign = MemoryEntry(id="L-old", namespace="project:old", content="old", importance=0.9)
    queue = RetryQueue(tmp_path / "queue.jsonl")
    _queued(foreign, queue, retry_count=MAX_RETRIES)

    result, _ = _remote_publish._drain_retry_queue_with_ids(
        queue, _cfg(tmp_path), current_row=lambda _ns, _i: None, namespace="project:new"
    )

    assert result["skipped"] == 1 and queue.depth() == 0
    posts.assert_not_called()


def test_a_row_read_failure_withholds_that_record_and_keeps_the_others_accounting(
    tmp_path: Path, posts: MagicMock
) -> None:
    """A resolver error is that record's refusal, not an abort that re-sends the records already published (codex r2)."""
    ok = MemoryEntry(id="L-ok", namespace="project:a", content="ok", importance=0.9)
    broken = MemoryEntry(id="L-broken", namespace="project:a", content="broken", importance=0.9)
    queue = RetryQueue(tmp_path / "queue.jsonl")
    _queued(ok, queue)
    _queued(broken, queue)

    def resolve(namespace: str, entry_id: str) -> MemoryEntry | None:
        if entry_id == "L-broken":
            raise StorageError("database is locked")
        return ok

    result, published = _remote_publish._drain_retry_queue_with_ids(
        queue, _cfg(tmp_path), current_row=resolve, namespace="project:a"
    )

    assert (result["drained"], result["failed"]) == (1, 1)
    assert [entry_id for entry_id, _, _ in published] == ["L-ok"]
    assert [r["entry_id"] for r in queue.snapshot()] == ["L-broken"]
    posts.assert_called_once()
