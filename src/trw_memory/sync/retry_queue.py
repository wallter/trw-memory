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
from typing import TypedDict

import structlog

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
        self._lock = threading.Lock()

    def enqueue(self, entry_id: str, payload: dict[str, object]) -> bool:
        """Append a failed publish to the retry queue.

        Returns ``False`` if the queue is at capacity (500 entries).
        """
        with self._lock:
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
        with self._lock:
            entries = self._read_all()
            if not entries:
                return {"drained": 0, "failed": 0, "skipped": 0}

            remaining: list[QueueRecord] = []
            drained = 0
            failed = 0
            skipped = 0

            for record in entries:
                if record["retry_count"] >= MAX_RETRIES:
                    remaining.append(record)
                    skipped += 1
                    continue

                retry_count = int(record["retry_count"])
                if retry_count > 0:
                    backoff_seconds = min(1.0 * (2 ** (retry_count - 1)), 30.0)
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
                else:
                    record["retry_count"] += 1
                    record["last_error"] = "publish returned False"
                    remaining.append(record)
                    failed += 1

            self._write_all(remaining)
            return {"drained": drained, "failed": failed, "skipped": skipped}

    def clear(self) -> None:
        """Clear the entire retry queue."""
        with self._lock:
            self._path.write_text("")

    def depth(self) -> int:
        """Return the number of entries in the queue."""
        with self._lock:
            return len(self._read_all())

    def snapshot(self) -> list[QueueRecord]:
        """Return a copy of the current queue records for reconciliation logic."""
        with self._lock:
            return list(self._read_all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[QueueRecord]:
        if not self._path.exists():
            return []
        entries: list[QueueRecord] = []
        for line in self._path.read_text().strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("retry_queue_corrupt_record_dropped", line_preview=line[:100])
        return entries

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
    def _serialized_size(entry: QueueRecord) -> int:
        """Return the encoded JSONL byte size for one record."""
        return len((json.dumps(entry) + "\n").encode("utf-8"))
