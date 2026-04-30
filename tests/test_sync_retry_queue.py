"""Tests for trw_memory.sync.retry_queue — PRD-CORE-047."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from trw_memory.sync.remote import clear_retry_queue, drain_retry_queue
from trw_memory.sync.retry_queue import MAX_QUEUE_BYTES, MAX_QUEUE_DEPTH, MAX_RETRIES, RetryQueue

from ._test_sync_support import make_sync_config as _make_config, mock_httpx_client as _mock_httpx_client


class TestRetryQueue:
    """FR06: RetryQueue provides JSONL persistence with depth cap."""

    def test_enqueue_appends_to_file(self, tmp_path: Path) -> None:
        """Enqueue writes a JSONL record to the queue file."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        assert queue.enqueue("M-001", {"summary": "test"})
        assert queue.depth() == 1

    def test_enqueue_returns_false_at_max_depth(self, tmp_path: Path) -> None:
        """Returns False when queue is at MAX_QUEUE_DEPTH capacity."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        for i in range(MAX_QUEUE_DEPTH):
            queue.enqueue(f"M-{i}", {"summary": f"entry-{i}"})
        assert not queue.enqueue("M-overflow", {"summary": "overflow"})
        assert queue.depth() == MAX_QUEUE_DEPTH

    def test_enqueue_returns_false_when_size_limit_exceeded(self, tmp_path: Path) -> None:
        """Queue enforces the 5MB aggregate size cap before writing a new record."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        oversized_detail = "x" * ((MAX_QUEUE_BYTES // 2) + 1024)

        assert queue.enqueue("M-001", {"summary": "test", "detail": oversized_detail})
        assert not queue.enqueue("M-002", {"summary": "test", "detail": oversized_detail})

    def test_drain_publishes_and_removes_successful(self, tmp_path: Path) -> None:
        """Drain removes entries that publish successfully."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test-1"})
        queue.enqueue("M-002", {"summary": "test-2"})

        result = queue.drain(lambda _: True)
        assert result == {"drained": 2, "failed": 0, "skipped": 0}
        assert queue.depth() == 0

    def test_drain_increments_retry_count_on_failure(self, tmp_path: Path) -> None:
        """Drain increments retry_count when publish returns False."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = queue.drain(lambda _: False)
        assert result["failed"] == 1
        assert queue.depth() == 1

        lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["retry_count"] == 1
        assert record["last_error"] == "publish returned False"

    def test_drain_applies_exponential_backoff_after_failures(self, tmp_path: Path) -> None:
        """Retries after prior failures sleep using the documented backoff curve."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            json.dumps(
                {
                    "entry_id": "M-001",
                    "payload": {"summary": "test"},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "retry_count": 3,
                    "last_error": "previous failure",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        with patch("trw_memory.sync.retry_queue.time.sleep") as sleep_mock:
            result = queue.drain(lambda _: False)

        assert result["failed"] == 1
        sleep_mock.assert_called_once_with(4.0)

    def test_drain_skips_entries_at_max_retries(self, tmp_path: Path) -> None:
        """Entries with retry_count >= MAX_RETRIES are skipped."""
        queue_path = tmp_path / "queue.jsonl"
        record = {
            "entry_id": "M-001",
            "payload": {"summary": "exhausted"},
            "queued_at": "2026-01-01T00:00:00Z",
            "retry_count": MAX_RETRIES,
            "last_error": "previous error",
        }
        queue_path.write_text(json.dumps(record) + "\n")

        queue = RetryQueue(queue_path)
        result = queue.drain(lambda _: True)
        assert result == {"drained": 0, "failed": 0, "skipped": 1}
        assert queue.depth() == 1

    def test_drain_handles_publish_exception(self, tmp_path: Path) -> None:
        """Drain catches exceptions from publish_fn and increments retry_count."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        def failing_publish(_payload: Any) -> bool:
            raise ConnectionError("network down")

        result = queue.drain(failing_publish)
        assert result["failed"] == 1
        assert queue.depth() == 1

        lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["retry_count"] == 1
        assert "network down" in record["last_error"]

    def test_clear_empties_queue(self, tmp_path: Path) -> None:
        """Clear truncates the queue file to empty."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})
        assert queue.depth() == 1

        queue.clear()
        assert queue.depth() == 0

    def test_depth_returns_correct_count(self, tmp_path: Path) -> None:
        """Depth returns the number of entries in the queue."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        assert queue.depth() == 0
        queue.enqueue("M-001", {"summary": "first"})
        assert queue.depth() == 1
        queue.enqueue("M-002", {"summary": "second"})
        assert queue.depth() == 2

    def test_depth_on_nonexistent_file(self, tmp_path: Path) -> None:
        """Depth returns 0 when the queue file doesn't exist."""
        queue = RetryQueue(tmp_path / "nonexistent.jsonl")
        assert queue.depth() == 0

    def test_enqueue_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Enqueue creates parent directories if needed."""
        queue_path = tmp_path / "subdir" / "queue.jsonl"
        queue = RetryQueue(queue_path)
        assert queue.enqueue("M-001", {"summary": "test"})
        assert queue_path.exists()

    def test_corrupt_jsonl_lines_are_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSONL lines should be logged and skipped, not crash dequeue."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            '{"entry_id": "M-001", "data": {"summary": "valid"}, "retries": 0}\n'
            "not valid json at all\n"
            '{"entry_id": "M-002", "data": {"summary": "also valid"}, "retries": 0}\n',
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        assert queue.depth() == 2

    def test_empty_lines_in_queue_are_harmless(self, tmp_path: Path) -> None:
        """Empty lines in the JSONL file should not cause errors."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            '{"entry_id": "M-001", "data": {"summary": "test"}, "retries": 0}\n\n\n',
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        assert queue.depth() == 1

    def test_drain_retry_queue_republishes_payloads(self, tmp_path: Path) -> None:
        """The remote helper drains queued payloads through the publish transport."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test", "source_learning_id": "M-001"})

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            _mock_httpx_client(mock_client_cls, status_code=200, json_data={"id": "42"})
            result = drain_retry_queue(queue, _make_config())

        assert result == {"drained": 1, "failed": 0, "skipped": 0, "remote_ids": {"M-001": "42"}}
        assert queue.depth() == 0

    def test_drain_retry_queue_skips_when_sync_disabled(self, tmp_path: Path) -> None:
        """Drain helper leaves the queue intact when sync is disabled."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = drain_retry_queue(queue, _make_config(sync_enabled=False))

        assert result == {"drained": 0, "failed": 0, "skipped": 1, "remote_ids": {}}
        assert queue.depth() == 1

    def test_drain_retry_queue_skips_invalid_platform_url(self, tmp_path: Path) -> None:
        """Drain helper leaves queued payloads untouched for unsafe URL schemes."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = drain_retry_queue(queue, _make_config(platform_url="file:///etc/passwd"))

        assert result == {"drained": 0, "failed": 0, "skipped": 1, "remote_ids": {}}
        assert queue.depth() == 1

    def test_clear_retry_queue_helper_empties_file(self, tmp_path: Path) -> None:
        """The remote helper delegates to the queue clear operation."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        clear_retry_queue(queue)

        assert queue.depth() == 0


class TestModuleConstants:
    """Verify module-level retry queue constants are set correctly."""

    def test_max_queue_depth(self) -> None:
        assert MAX_QUEUE_DEPTH == 500

    def test_max_retries(self) -> None:
        assert MAX_RETRIES == 5
