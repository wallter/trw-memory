"""Tests for trw_memory.sync.retry_queue — PRD-CORE-047."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

from structlog.testing import capture_logs

from trw_memory.sync.remote import clear_retry_queue, drain_retry_queue
from trw_memory.sync.retry_queue import MAX_QUEUE_BYTES, MAX_QUEUE_DEPTH, MAX_RETRIES, RetryQueue

from ._test_sync_support import make_sync_config as _make_config
from ._test_sync_support import mock_httpx_client as _mock_httpx_client


def _record_line(
    entry_id: str,
    payload: dict[str, Any],
    *,
    retry_count: int = 0,
    last_error: str | None = None,
) -> str:
    """Serialise a well-formed ``QueueRecord`` as one JSONL line (with newline)."""
    return (
        json.dumps(
            {
                "entry_id": entry_id,
                "payload": payload,
                "queued_at": "2026-01-01T00:00:00Z",
                "retry_count": retry_count,
                "last_error": last_error,
            }
        )
        + "\n"
    )


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
        # Exhausted records are evicted (dead-letter drain) so that new failures
        # can be enqueued once the 500-entry cap would otherwise be full.
        assert queue.depth() == 0

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
            _record_line("M-001", {"summary": "valid"})
            + "not valid json at all\n"
            + _record_line("M-002", {"summary": "also valid"}),
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        assert queue.depth() == 2

    def test_corrupt_record_log_omits_payload_contents(self, tmp_path: Path) -> None:
        """Corrupt-row observability must not leak raw payload text (privacy).

        Regression: ``_read_all`` previously logged ``line_preview=line[:100]``,
        which could surface sensitive memory content or metadata. The dropped-row
        log must carry only structural locators (path, line number, error class).
        """
        secret = "SENTINEL-SECRET-cAfEbAbE-do-not-log"
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            '{"entry_id": "M-001", "payload": {"summary": "valid"}, '
            '"queued_at": "2026-01-01T00:00:00Z", "retry_count": 0, "last_error": null}\n'
            f'{{"payload": "{secret}" not valid json\n',
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        with capture_logs() as logs:
            depth = queue.depth()

        # Fail-open: valid record preserved, corrupt row skipped.
        assert depth == 1

        dropped = [e for e in logs if e["event"] == "retry_queue_corrupt_record_dropped"]
        assert len(dropped) == 1
        record = dropped[0]
        assert record["path"] == str(queue_path)
        assert record["line_number"] == 2
        assert record["error_class"] == "JSONDecodeError"
        assert "line_preview" not in record
        # The sentinel must never appear in any field of the emitted log event.
        assert secret not in json.dumps(record)

    def test_valid_json_non_object_rows_are_skipped(self, tmp_path: Path) -> None:
        """Valid JSON that is not a QueueRecord object (scalar/list) is dropped.

        Regression: ``_read_all`` only guarded ``json.JSONDecodeError``, so a
        valid-JSON scalar/list/legacy-dict slipped into ``list[QueueRecord]``
        and crashed ``drain``/``depth`` on ``record["retry_count"]``.
        """
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            _record_line("M-001", {"summary": "valid"})
            + "42\n"  # JSON scalar
            + '"just a string"\n'  # JSON string scalar
            + "[1, 2, 3]\n"  # JSON list
            + _record_line("M-002", {"summary": "also valid"}),
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        with capture_logs() as logs:
            depth = queue.depth()

        # Fail-open: the two well-formed records survive; the three
        # non-object rows are dropped without crashing.
        assert depth == 2
        dropped = [e for e in logs if e["event"] == "retry_queue_corrupt_record_dropped"]
        assert len(dropped) == 3
        assert {e["error_class"] for e in dropped} == {"NonObjectRow"}
        assert {e["line_number"] for e in dropped} == {2, 3, 4}

    def test_dict_missing_required_field_is_skipped_without_crashing(self, tmp_path: Path) -> None:
        """A dict missing retry_count or payload is dropped; drain/depth survive.

        These wrong-shape dicts are exactly what would have crashed
        ``drain`` at ``record["retry_count"]`` / ``record["payload"]``.
        """
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            # Missing retry_count
            json.dumps(
                {
                    "entry_id": "M-001",
                    "payload": {"summary": "no-retry-count"},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "last_error": None,
                }
            )
            + "\n"
            # Missing payload
            + json.dumps(
                {
                    "entry_id": "M-002",
                    "queued_at": "2026-01-01T00:00:00Z",
                    "retry_count": 0,
                    "last_error": None,
                }
            )
            + "\n"
            # Wrong type: retry_count is a string
            + json.dumps(
                {
                    "entry_id": "M-003",
                    "payload": {"summary": "bad-type"},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "retry_count": "0",
                    "last_error": None,
                }
            )
            + "\n"
            + _record_line("M-OK", {"summary": "valid"}),
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        # depth() must not raise on the malformed rows.
        assert queue.depth() == 1

        # drain() must not raise (it indexes retry_count/payload) and must
        # publish the one surviving valid record.
        published: list[dict[str, object]] = []

        def _capture(payload: dict[str, object]) -> bool:
            published.append(payload)
            return True

        result = queue.drain(_capture)
        assert result == {"drained": 1, "failed": 0, "skipped": 0}
        assert published == [{"summary": "valid"}]
        assert queue.depth() == 0

    def test_schema_mismatch_log_omits_payload_contents(self, tmp_path: Path) -> None:
        """A wrong-shape (but valid-JSON) row must not leak payload text to logs."""
        secret = "SENTINEL-SECRET-dEadBeeF-do-not-log"
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            _record_line("M-001", {"summary": "valid"})
            # Wrong shape: a dict carrying sensitive text but missing retry_count.
            + json.dumps(
                {
                    "entry_id": "M-002",
                    "payload": {"summary": secret},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "last_error": secret,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        with capture_logs() as logs:
            depth = queue.depth()

        assert depth == 1
        dropped = [e for e in logs if e["event"] == "retry_queue_corrupt_record_dropped"]
        assert len(dropped) == 1
        record = dropped[0]
        assert record["path"] == str(queue_path)
        assert record["line_number"] == 2
        assert record["error_class"] == "SchemaMismatch"
        # The sentinel must never appear in any field of the emitted log event.
        assert secret not in json.dumps(record)

    def test_non_utf8_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A single non-UTF-8 row is isolated; valid adjacent records survive.

        Regression: ``_read_all`` decoded the whole file via ``read_text()``
        before per-row parsing, so one torn/non-UTF-8 byte raised
        ``UnicodeDecodeError`` and bricked ``depth``/``snapshot``/``drain``,
        contrary to the queue's fail-open corrupt-row contract.
        """
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_bytes(
            _record_line("M-001", {"summary": "valid"}).encode("utf-8")
            # 0xFF is never a valid UTF-8 byte — decoding this line must fail.
            + b"\xff\xfe torn non-utf-8 row\n"
            + _record_line("M-002", {"summary": "also valid"}).encode("utf-8")
        )
        queue = RetryQueue(queue_path)

        # depth/snapshot must not raise and must preserve both valid records.
        assert queue.depth() == 2
        snapshot = queue.snapshot()
        assert [record["entry_id"] for record in snapshot] == ["M-001", "M-002"]

        # drain must not raise and must publish the two surviving records.
        published: list[dict[str, object]] = []

        def _capture(payload: dict[str, object]) -> bool:
            published.append(payload)
            return True

        result = queue.drain(_capture)
        assert result == {"drained": 2, "failed": 0, "skipped": 0}
        assert published == [{"summary": "valid"}, {"summary": "also valid"}]
        assert queue.depth() == 0

    def test_non_utf8_line_log_is_content_free(self, tmp_path: Path) -> None:
        """The non-UTF-8 drop log carries only structural locators (privacy).

        A torn row may embed sensitive memory text; the dropped-row event must
        surface only path + line number + error class — never raw bytes, a line
        preview, byte offsets, or exception text.
        """
        secret = "SENTINEL-NONUTF8-do-not-log"
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_bytes(
            _record_line("M-001", {"summary": "valid"}).encode("utf-8")
            + b"\xff "
            + secret.encode("utf-8")
            + b" \xfe\n"
            + _record_line("M-002", {"summary": "also valid"}).encode("utf-8")
        )
        queue = RetryQueue(queue_path)

        with capture_logs() as logs:
            depth = queue.depth()

        # Fail-open: the two well-formed records survive the torn middle row.
        assert depth == 2
        dropped = [e for e in logs if e["event"] == "retry_queue_corrupt_record_dropped"]
        assert len(dropped) == 1
        record = dropped[0]
        assert record["path"] == str(queue_path)
        assert record["line_number"] == 2
        assert record["error_class"] == "UnicodeDecodeError"
        assert "line_preview" not in record
        # The sentinel must never appear in any field of the emitted log event.
        assert secret not in json.dumps(record)

    def test_crlf_line_endings_are_preserved(self, tmp_path: Path) -> None:
        """Byte-level line splitting still handles \\r\\n records (no regression)."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_bytes(
            _record_line("M-001", {"summary": "valid"}).replace("\n", "\r\n").encode("utf-8")
            + _record_line("M-002", {"summary": "also valid"}).replace("\n", "\r\n").encode("utf-8")
        )
        queue = RetryQueue(queue_path)
        assert queue.depth() == 2

    def test_empty_lines_in_queue_are_harmless(self, tmp_path: Path) -> None:
        """Empty lines in the JSONL file should not cause errors."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            _record_line("M-001", {"summary": "test"}) + "\n\n",
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


class TestDrainLockNotHeldDuringSleep:
    """P1 regression: drain must NOT hold self._lock while sleeping.

    Original bug: drain held self._lock across time.sleep(backoff_seconds)
    inside the loop, starving enqueue/depth/snapshot for up to ~30s.
    Fix: collect work under lock, release, sleep+publish, reacquire only
    for write-back.
    """

    def test_enqueue_succeeds_concurrently_during_drain_backoff(self, tmp_path: Path) -> None:
        """A concurrent enqueue can complete while drain is sleeping (backoff).

        Seeds the queue with a record that has retry_count=2 (backoff=2s),
        patches time.sleep to inject a concurrent enqueue in the sleep window,
        and asserts the enqueue completes without deadlock.
        """
        queue_path = tmp_path / "queue.jsonl"
        # Seed one record that requires backoff (retry_count > 0).
        queue_path.write_text(
            json.dumps(
                {
                    "entry_id": "M-backoff",
                    "payload": {"summary": "retry me"},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "retry_count": 2,
                    "last_error": "previous failure",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        concurrent_enqueue_completed = threading.Event()

        def fake_sleep(seconds: float) -> None:
            # While "sleeping" (lock must NOT be held), enqueue a new record.
            result = queue.enqueue("M-concurrent", {"summary": "concurrent"})
            if result:
                concurrent_enqueue_completed.set()
            # Do not actually sleep to keep tests fast.

        with patch("trw_memory.sync.retry_queue.time.sleep", side_effect=fake_sleep):
            queue.drain(lambda _: False)

        # The key assertion: enqueue must have completed without deadlock.
        # (Note: the drain's final _write_all may overwrite the concurrent enqueue's
        # file write — that is an acknowledged limitation of the JSONL design and
        # orthogonal to the lock-starve fix. The bug we are guarding against is
        # the lock being held during sleep, causing enqueue to block indefinitely.)
        assert concurrent_enqueue_completed.is_set(), (
            "Concurrent enqueue during drain backoff must complete without deadlock — "
            "drain must release self._lock before sleeping"
        )

    def test_depth_not_blocked_during_drain_publish(self, tmp_path: Path) -> None:
        """depth() completes promptly while drain is publishing (lock released).

        Seeds one record with no backoff needed (retry_count=0), uses a
        publish_fn that checks depth() from a background thread to confirm
        the lock is not held during publish.
        """
        queue_path = tmp_path / "queue.jsonl"
        queue = RetryQueue(queue_path)
        queue.enqueue("M-001", {"summary": "test"})

        depth_during_publish: list[int] = []
        depth_unblocked = threading.Event()

        def publish_fn(payload: dict[str, Any]) -> bool:
            # Check depth from a background thread; should not deadlock.
            result: list[int] = []

            def check_depth() -> None:
                result.append(queue.depth())
                depth_unblocked.set()

            t = threading.Thread(target=check_depth, daemon=True)
            t.start()
            t.join(timeout=2.0)
            depth_during_publish.extend(result)
            return True

        queue.drain(publish_fn)

        assert depth_unblocked.is_set(), (
            "depth() must not be blocked while drain is inside publish_fn — "
            "lock must be released before calling publish_fn"
        )
        assert depth_during_publish, "depth() must return a value during drain"
