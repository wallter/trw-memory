"""Concurrency regression coverage for quarantine review decisions."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security import _runtime_quarantine as quarantine
from trw_memory.security._runtime_quarantine import review_quarantined_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def test_reviews_are_serialized_per_quarantine_store(tmp_path: Path) -> None:
    entry = MemoryEntry(id="M-review", content="quarantined", namespace="default")
    active_backend = MagicMock()
    active_backend.get.return_value = None
    quarantine_backend = MagicMock()
    active_calls = 0
    max_active_calls = 0
    counter_lock = threading.Lock()

    def get_entry(_learning_id: str, **_kwargs) -> MemoryEntry:
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


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("racer", ["intake", "delete"])
def test_a_write_landing_mid_review_is_not_clobbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str, racer: str
) -> None:
    """B71-91: intake B (or a forget) racing a review of A under the same id is applied after it, never lost.

    The racer starts after the review has read A and before it deletes (approve) or
    re-stores (reject) that row. Unlocked, the review then deletes B or overwrites it
    with A (or resurrects a forgotten row); with one lock the racer waits for the review.
    """
    config = MemoryConfig(
        storage_path=str(tmp_path / "mem"),
        quarantine_db_path=str(tmp_path / "quarantine.db"),
        audit_enabled=False,
    )
    ns = "default"
    quarantine.store_quarantined_entry(config, MemoryEntry(id="M-q", content="submission A", namespace=ns))
    history = quarantine.get_status_history
    racers: list[threading.Thread] = []

    def race(*args: object, **kwargs: object) -> list[dict[str, str]]:
        if not racers:
            if racer == "intake":
                entry_b = MemoryEntry(id="M-q", content="submission B", namespace=ns)
                target = lambda: quarantine.store_quarantined_entry(config, entry_b)  # noqa: E731
            else:
                target = lambda: quarantine.delete_quarantined_entries(config, namespace=ns, memory_id="M-q")  # noqa: E731
            racers.append(threading.Thread(target=target))
            racers[0].start()
            racers[0].join(timeout=0.5)  # unlocked it lands now; locked it waits for the review
        return history(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(quarantine, "get_status_history", race)
    with SQLiteBackend(tmp_path / "active.db") as active:
        result = review_quarantined_entry(
            config, active_backend=active, learning_id="M-q", decision=decision, reviewer_id="r", namespace=ns
        )
    racers[0].join()

    assert result["status"] == ("approved" if decision == "approve" else "rejected")
    with quarantine.open_quarantine_backend(config) as backend:
        row = backend.get("M-q", namespace=ns)
    if racer == "delete":
        assert row is None
    else:
        assert row is not None
        assert (row.content, row.metadata.get("quarantined"), row.metadata.get("review_decision")) == (
            "submission B",
            "true",
            None,
        )
