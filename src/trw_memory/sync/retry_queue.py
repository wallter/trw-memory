"""JSONL retry queue for failed sync publishes.

Implements FR06 from PRD-CORE-047.  Failed publish operations are appended
to a persistent JSONL file and re-attempted on the next drain cycle.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import structlog
from typing_extensions import TypedDict

from trw_memory.storage.persistence import lock_for_rmw

logger = structlog.get_logger(__name__)

MAX_QUEUE_DEPTH = 500
MAX_QUEUE_BYTES = 5 * 1024 * 1024
MAX_RETRIES = 5


class QueueRecord(TypedDict):
    """Typed structure for a single JSONL retry queue record."""

    entry_id: str
    payload: dict[str, object]
    queued_at: str
    retry_count: int
    last_error: str | None


class RetryQueue:
    """Persistent JSONL retry queue for failed publish operations.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(self, queue_path: Path) -> None:
        self._path = queue_path
        self._drain_lock_path = queue_path.with_name(f"{queue_path.name}.drain")
        self._lock = threading.Lock()

    def enqueue(self, entry_id: str, payload: dict[str, object]) -> bool:
        """Append a failed publish to the retry queue.

        Returns ``False`` if the queue is at capacity (500 entries).
        """
        with self._lock, lock_for_rmw(self._path):
            entries = self._read_all()
            if len(entries) >= MAX_QUEUE_DEPTH:
                logger.warning(
                    "retry_queue_full",
                    entry_id=entry_id,
                    depth=len(entries),
                )
                return False

            record: QueueRecord = {
                "entry_id": entry_id,
                "payload": payload,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0,
                "last_error": None,
            }
            queued_bytes = sum(self._serialized_size(entry) for entry in entries)
            record_bytes = self._serialized_size(record)
            if queued_bytes + record_bytes > MAX_QUEUE_BYTES:
                logger.warning(
                    "retry_queue_size_limit_exceeded",
                    entry_id=entry_id,
                    queued_bytes=queued_bytes,
                    record_bytes=record_bytes,
                    max_bytes=MAX_QUEUE_BYTES,
                )
                return False
            self._write_all([*entries, record])
            return True

    def drain(self, publish_fn: Callable[[dict[str, object]], bool]) -> dict[str, int]:
        """Attempt to drain the queue by re-publishing all entries.

        Args:
            publish_fn: ``Callable(payload) -> bool``.  Returns ``True`` on
                success.

        Returns:
            ``{"drained": int, "failed": int, "skipped": int}``
        """
        result, _ = self._drain_with_ids(publish_fn)
        return result

    def _drain_with_ids(
        self,
        publish_fn: Callable[[dict[str, object]], bool],
    ) -> tuple[dict[str, int], list[str]]:
        """Drain queued records and return canonical IDs for successful records."""
        with lock_for_rmw(self._drain_lock_path):
            return self._drain_serialized(publish_fn)

    def _drain_serialized(
        self,
        publish_fn: Callable[[dict[str, object]], bool],
    ) -> tuple[dict[str, int], list[str]]:
        """Process one drain while the cross-instance drain lock is held."""
        # Collect work under lock, then release before sleeping/publishing so
        # enqueue/depth/snapshot are not starved for up to ~30s per drain cycle.
        # (Bug: original held self._lock across time.sleep(backoff_seconds) inside
        # the loop — up to 30s of lock contention per retry.)
        with self._lock, lock_for_rmw(self._path):
            entries = self._read_all()
        if not entries:
            return {"drained": 0, "failed": 0, "skipped": 0}, []

        remaining: list[QueueRecord] = []
        drained = 0
        failed = 0
        skipped = 0
        drained_entry_ids: list[str] = []

        for record in entries:
            if record["retry_count"] >= MAX_RETRIES:
                # Exhausted records are evicted from the queue (dead-letter drain)
                # rather than re-appended.  Silently keeping them blocks new
                # failures from being enqueued once the 500-entry cap is reached.
                logger.warning(
                    "retry_queue_record_exhausted",
                    entry_id=record.get("entry_id"),
                    retry_count=record["retry_count"],
                    last_error=record.get("last_error"),
                )
                skipped += 1
                continue

            retry_count = int(record["retry_count"])
            if retry_count > 0:
                backoff_seconds = min(1.0 * (2 ** (retry_count - 1)), 30.0)
                # Sleep OUTSIDE the lock so concurrent enqueue/depth/snapshot
                # are not blocked during backoff.
                time.sleep(backoff_seconds)

            try:
                success = publish_fn(record["payload"])
            except (OSError, ConnectionError, ValueError) as exc:
                record["retry_count"] += 1
                record["last_error"] = str(exc)
                remaining.append(record)
                failed += 1
                continue

            if success:
                drained += 1
                drained_entry_ids.append(record["entry_id"])
            else:
                record["retry_count"] += 1
                record["last_error"] = "publish returned False"
                remaining.append(record)
                failed += 1

        # Write results back under lock. Re-read first and re-append any records
        # enqueued during the lock-free processing window above: drain snapshots
        # the queue under the lock, releases it to publish/sleep, then re-acquires
        # to write. Overwriting with `remaining` alone would silently drop a
        # record that enqueue() appended in that window (TOCTOU data loss).
        with self._lock, lock_for_rmw(self._path):
            processed_keys = {self._record_key(r) for r in entries}
            new_arrivals = [r for r in self._read_all() if self._record_key(r) not in processed_keys]
            self._write_all(remaining + new_arrivals)
        return {"drained": drained, "failed": failed, "skipped": skipped}, drained_entry_ids

    def clear(self) -> None:
        """Clear the entire retry queue."""
        with self._lock, lock_for_rmw(self._path):
            self._write_all([])

    def depth(self) -> int:
        """Return the number of entries in the queue."""
        with self._lock, lock_for_rmw(self._path):
            return len(self._read_all())

    def snapshot(self) -> list[QueueRecord]:
        """Return a copy of the current queue records for reconciliation logic."""
        with self._lock, lock_for_rmw(self._path):
            return list(self._read_all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[QueueRecord]:
        if not self._path.exists():
            return []
        entries: list[QueueRecord] = []
        # Read raw bytes and decode one line at a time so a single non-UTF-8
        # row is isolated like any other corrupt row, rather than aborting the
        # whole-file decode and bricking depth/snapshot/drain. ``bytes`` splits
        # on the same ASCII line boundaries (\n, \r, \r\n) that the writer uses.
        for line_number, raw_line in enumerate(self._path.read_bytes().splitlines(), start=1):
            line = self._decode_line(raw_line, line_number)
            if line is None or not line.strip():
                continue
            record = self._parse_record(line, line_number)
            if record is not None:
                entries.append(record)
        return entries

    def _decode_line(self, raw_line: bytes, line_number: int) -> str | None:
        """Decode one raw JSONL byte-line as UTF-8, isolating non-UTF-8 rows.

        Fail open: a torn or non-UTF-8 line yields ``None`` (dropped with a
        content-free diagnostic) instead of raising ``UnicodeDecodeError`` and
        bricking ``depth``/``snapshot``/``drain`` for every adjacent valid
        record. Records are written UTF-8 by :meth:`_write_all`, so a decode
        failure marks a row corrupted at the byte layer — below the JSON
        well-formedness that :meth:`_parse_record` guards.

        Like :meth:`_parse_record`, never logs raw row bytes, decoded text, or
        exception/offset detail — only structural locators (path, line number,
        error class) so sensitive memory text cannot leak.
        """
        try:
            return raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            self._log_dropped_record(line_number, type(exc).__name__)
            return None

    def _parse_record(self, line: str, line_number: int) -> QueueRecord | None:
        """Parse and validate a single JSONL row into a :class:`QueueRecord`.

        Fail open: returns ``None`` for any row that is not a well-formed
        ``QueueRecord`` — invalid JSON, valid-JSON-but-not-an-object (scalars,
        lists), or a dict whose fields are missing or the wrong type. Dropping
        such rows keeps one bad line from crashing ``drain``/``snapshot``, which
        index ``record["retry_count"]`` and ``record["payload"]`` directly.

        Validated shape (the fields ``enqueue``/``drain``/``snapshot`` rely on):
        ``entry_id`` str, ``payload`` object, ``queued_at`` str,
        ``retry_count`` int (not bool), ``last_error`` str-or-null.

        Never logs raw row or payload contents — only structural locators
        (path, line number, error class) so sensitive memory text cannot leak.
        """
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            self._log_dropped_record(line_number, type(exc).__name__)
            return None

        if not isinstance(raw, dict):
            self._log_dropped_record(line_number, "NonObjectRow")
            return None

        entry_id = raw.get("entry_id")
        payload = raw.get("payload")
        queued_at = raw.get("queued_at")
        retry_count = raw.get("retry_count")
        last_error = raw.get("last_error")

        if (
            not isinstance(entry_id, str)
            or not isinstance(payload, dict)
            or not isinstance(queued_at, str)
            or not isinstance(retry_count, int)
            or isinstance(retry_count, bool)
            or not (last_error is None or isinstance(last_error, str))
        ):
            self._log_dropped_record(line_number, "SchemaMismatch")
            return None

        return QueueRecord(
            entry_id=entry_id,
            payload=payload,
            queued_at=queued_at,
            retry_count=retry_count,
            last_error=last_error,
        )

    def _log_dropped_record(self, line_number: int, error_class: str) -> None:
        """Emit the corrupt-row drop event with structural locators only.

        Never include raw row or payload contents — the payload may carry
        sensitive memory text or metadata.
        """
        logger.warning(
            "retry_queue_corrupt_record_dropped",
            path=str(self._path),
            line_number=line_number,
            error_class=error_class,
        )

    def _write_all(self, entries: list[QueueRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._path.parent),
            prefix=f"{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.writelines(json.dumps(entry) + "\n" for entry in entries)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self._path)

    @staticmethod
    def _record_key(record: QueueRecord) -> tuple[str, str]:
        """Stable identity for a record across a drain read/write cycle.

        ``(entry_id, queued_at)`` survives the ``retry_count`` mutation drain
        applies, so a record processed this cycle is still matched after its
        counter changes — while a record enqueued during the lock-free window
        (a fresh ``queued_at``) is recognised as a new arrival and preserved.
        """
        return (record["entry_id"], record["queued_at"])

    @staticmethod
    def _serialized_size(entry: QueueRecord) -> int:
        """Return the encoded JSONL byte size for one record."""
        return len((json.dumps(entry) + "\n").encode("utf-8"))
