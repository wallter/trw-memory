"""PRD-CORE-333: a row the quarantine ledger blocks never leaves the host through sync push (CORE-333-PUBLISHER-BYPASS).

The read layer filters every StorageBackend method, but the push pages dirty rows with
``DeltaTracker.get_dirty_entries``, a raw query outside the backend class, so a
quarantined learning was still paged and POSTed. Filtering after the LIMIT alone would
stall the push: a page that held only blocked rows would come back empty for ever, since
blocked rows stay dirty and always sort first. The page is filled from the rows behind them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.quarantine_ledger import LedgerIdentity, ledger_for_config
from trw_memory.sync.delta import DeltaTracker

NS = "default"


@pytest.mark.parametrize("kind", ["sqlite", "yaml"])
def test_a_quarantined_dirty_row_is_never_paged_for_push(tmp_path: Path, kind: str) -> None:
    config = MemoryConfig(storage_path=str(tmp_path / "memory"), storage_backend=kind)
    with create_backend_from_config(config, NS) as backend:
        for entry_id in ("Q-1", "Q-2", "V-1", "V-2"):  # the blocked rows are the oldest dirty rows
            backend.store(MemoryEntry(id=entry_id, content=f"learning {entry_id}", namespace=NS))
        ledger = ledger_for_config(config)
        for entry_id in ("Q-1", "Q-2"):
            ledger.append(LedgerIdentity(namespace=NS, entry_id=entry_id), "quarantined", actor="system")

        everything = DeltaTracker.get_dirty_entries(backend, namespace=NS)
        page = DeltaTracker.get_dirty_entries(backend, namespace=NS, limit=2)

    assert sorted(e.id for e in everything) == ["V-1", "V-2"]
    assert sorted(e.id for e in page) == ["V-1", "V-2"], "a page of blocked rows must not stall the push"
