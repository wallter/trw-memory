"""PRD-CORE-333: the platform publish asks the quarantine ledger immediately before every POST (CORE-333-PUBLISHER-BYPASS).

A row can be quarantined after it was published-on-store failed and queued; the retry
drain then sent it with no recheck. The publish now carries the row's full ledger
identity (in the queued record too) and refuses the POST when the ledger blocks it, when
the ledger cannot be read, or when a queued record carries no identity at all.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.quarantine_ledger import LedgerIdentity, QuarantineLedger, ledger_for_config
from trw_memory.sync import _remote_publish
from trw_memory.sync.retry_queue import RetryQueue


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryConfig:
    monkeypatch.setattr(_remote_publish, "platform_contact_blocked", lambda *_a, **_k: None)
    return MemoryConfig(storage_path=str(tmp_path / "memory"), platform_url="https://platform.invalid")


@pytest.fixture
def posts() -> object:
    with patch("trw_memory.sync._remote_publish.httpx.Client") as client_cls:
        response = MagicMock(status_code=201)
        response.json.return_value = {"id": "R-1"}
        client_cls.return_value.__enter__.return_value.post.return_value = response
        yield client_cls.return_value.__enter__.return_value.post


def _entry(entry_id: str = "L-1", *, source: str = "") -> MemoryEntry:
    metadata = {"source_learning_id": source} if source else {}
    return MemoryEntry(id=entry_id, namespace="proj", content="a learning", importance=0.9, metadata=metadata)


def test_publish_refuses_an_entry_quarantined_by_its_source_id(cfg: MemoryConfig, posts: MagicMock) -> None:
    ledger_for_config(cfg).append(
        LedgerIdentity(namespace="x", entry_id="x", source_learning_id="S-9"), "quarantined", actor="system"
    )

    result = _remote_publish.publish_memory_result(_entry(source="S-9"), cfg)

    posts.assert_not_called()
    assert result == {"success": False, "remote_id": None, "retryable": False}


def test_a_row_quarantined_after_it_was_queued_is_never_sent(
    cfg: MemoryConfig, posts: MagicMock, tmp_path: Path
) -> None:
    from trw_memory.security.egress_gate import identity_keys

    entry = _entry()
    queue = RetryQueue(tmp_path / "queue.jsonl")
    payload = dict(_remote_publish._anonymize_entry(entry))
    payload[_remote_publish.LEDGER_KEYS_FIELD] = identity_keys(entry)
    assert queue.enqueue(entry.id, payload)
    ledger_for_config(cfg).append(LedgerIdentity.of(entry), "rejected", actor="reviewer")

    result, published = _remote_publish._drain_retry_queue_with_ids(queue, cfg)

    posts.assert_not_called()
    assert published == [] and result["drained"] == 0
    kept = [json.loads(line) for line in (tmp_path / "queue.jsonl").read_text().splitlines()]
    assert kept and kept[0]["payload"][_remote_publish.LEDGER_KEYS_FIELD], (
        "the record keeps its identity for later drains"
    )


def test_a_queued_record_without_an_identity_is_never_sent(cfg: MemoryConfig, posts: MagicMock, tmp_path: Path) -> None:
    queue = RetryQueue(tmp_path / "queue.jsonl")
    assert queue.enqueue("L-legacy", dict(_remote_publish._anonymize_entry(_entry("L-legacy"))))

    _remote_publish._drain_retry_queue_with_ids(queue, cfg)

    posts.assert_not_called()


def test_a_clean_queued_record_is_sent_without_its_ledger_keys(
    cfg: MemoryConfig, posts: MagicMock, tmp_path: Path
) -> None:
    from trw_memory.security.egress_gate import identity_keys

    entry = _entry()
    queue = RetryQueue(tmp_path / "queue.jsonl")
    payload = dict(_remote_publish._anonymize_entry(entry))
    payload[_remote_publish.LEDGER_KEYS_FIELD] = identity_keys(entry)
    assert queue.enqueue(entry.id, payload)

    result, _ = _remote_publish._drain_retry_queue_with_ids(queue, cfg)

    assert result["drained"] == 1
    sent = posts.call_args.kwargs["json"]
    assert _remote_publish.LEDGER_KEYS_FIELD not in sent


def test_an_unreadable_ledger_refuses_and_keeps_the_send_retryable(
    cfg: MemoryConfig, posts: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(self: QuarantineLedger) -> object:
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(QuarantineLedger, "view", _broken)

    result = _remote_publish.publish_memory_result(_entry(), cfg)

    posts.assert_not_called()
    assert result == {"success": False, "remote_id": None, "retryable": True}
