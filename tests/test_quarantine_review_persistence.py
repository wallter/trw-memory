"""Quarantine review decisions become terminal only after persistence."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_quarantine import review_quarantined_entry


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_failed_review_persistence_does_not_record_terminal_status(decision: str) -> None:
    entry = MemoryEntry(id="M-review", content="quarantined", namespace="default")
    active_backend = MagicMock()
    active_backend.get.return_value = None
    quarantine_backend = MagicMock()
    quarantine_backend.get.return_value = entry
    failing_store = active_backend.store if decision == "approve" else quarantine_backend.store
    failing_store.side_effect = RuntimeError("write failed")

    @contextmanager
    def open_backend(_config: MemoryConfig):
        yield quarantine_backend

    with (
        patch("trw_memory.security._runtime_quarantine.open_quarantine_backend", open_backend),
        patch("trw_memory.security._runtime_quarantine.get_status_history", return_value=[]),
        patch("trw_memory.security._runtime_quarantine.append_review_log") as append_log,
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            review_quarantined_entry(
                MemoryConfig(),
                active_backend=active_backend,
                learning_id=entry.id,
                decision=decision,
                reviewer_id="reviewer",
            )
        append_log.assert_not_called()

        failing_store.side_effect = None
        result = review_quarantined_entry(
            MemoryConfig(),
            active_backend=active_backend,
            learning_id=entry.id,
            decision=decision,
            reviewer_id="reviewer",
        )

    assert result["status"] == ("approved" if decision == "approve" else "rejected")
    append_log.assert_called_once()
