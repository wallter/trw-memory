"""JSONL retry queue for failed sync publishes.

Implements FR06 from PRD-CORE-047.  Failed publish operations are appended
to a persistent JSONL file and re-attempted on the next drain cycle.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

MAX_QUEUE_DEPTH = 500
MAX_RETRIES = 5


class RetryQueue:
    """Persistent JSONL retry queue for failed publish operations.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(self, queue_path: Path) -> None:
        self._path = queue_path
        self._lock = threading.Lock()

    def enqueue(self, entry_id: str, payload: dict[str, Any]) -> bool:
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

            record: dict[str, Any] = {
                "entry_id": entry_id,
                "payload": payload,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0,
                "last_error": None,
            }
            self._append(record)
            return True

    def drain(self, publish_fn: Callable[[dict[str, Any]], bool]) -> dict[str, int]:
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

            remaining: list[dict[str, Any]] = []
            drained = 0
            failed = 0
            skipped = 0

            for record in entries:
                if record.get("retry_count", 0) >= MAX_RETRIES:
                    remaining.append(record)
                    skipped += 1
                    continue

                try:
                    success = publish_fn(record["payload"])
                except (OSError, ConnectionError, ValueError) as exc:
                    record["retry_count"] = record.get("retry_count", 0) + 1
                    record["last_error"] = str(exc)
                    remaining.append(record)
                    failed += 1
                    continue

                if success:
                    drained += 1
                else:
                    record["retry_count"] = record.get("retry_count", 0) + 1
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self._path.read_text().strip().splitlines():
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError):
                    entries.append(json.loads(line))
        return entries

    def _append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_all(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
