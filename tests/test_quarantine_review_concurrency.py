"""Concurrency regression coverage for quarantine review decisions."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_quarantine import review_quarantined_entry


def test_reviews_are_serialized_per_quarantine_store(tmp_path: Path) -> None:
    entry = MemoryEntry(id="M-review", content="quarantined", namespace="default")
    active_backend = MagicMock()
    active_backend.get.return_value = None
    quarantine_backend = MagicMock()
    active_calls = 0
    max_active_calls = 0
    counter_lock = threading.Lock()

    def get_entry(_learning_id: str) -> MemoryEntry:
        nonlocal active_calls, max_active_calls
        with counter_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.01)
        with counter_lock:
            active_calls -= 1
        return entry

    quarantine_backend.get.side_effect = get_entry

    @contextmanager
    def open_backend(_config: MemoryConfig):
        yield quarantine_backend

    config = MemoryConfig(quarantine_db_path=str(tmp_path / "quarantine.db"))
    with (
        patch("trw_memory.security._runtime_quarantine.open_quarantine_backend", open_backend),
        patch("trw_memory.security._runtime_quarantine.get_status_history", return_value=[]),
        patch("trw_memory.security._runtime_quarantine.append_review_log"),
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        results = list(
            pool.map(
                lambda _: review_quarantined_entry(
                    config,
                    active_backend=active_backend,
                    learning_id=entry.id,
                    decision="reject",
                    reviewer_id="reviewer",
                ),
                range(8),
            )
        )

    assert all(result["status"] == "rejected" for result in results)
    assert max_active_calls == 1
