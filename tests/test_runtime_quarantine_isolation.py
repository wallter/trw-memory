"""Tests for SEC-001 quarantine namespace isolation + audit completeness.

Covers closure re-audit findings:
- #1: delete_quarantined_entries memory_id branch must verify namespace.
- #6: ...and must verify the entry is actually quarantined.
- #2: list_quarantined_entries must not silently truncate at limit*5.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.runtime import (
    delete_quarantined_entries,
    list_quarantined_entries,
    store_quarantined_entry,
)


def _cfg(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "mem"))


def _entry(entry_id: str, namespace: str, *, actor: str = "agent-a") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        namespace=namespace,
        source_identity=actor,
    )


class TestQuarantineNamespaceIsolation:
    def test_delete_by_id_rejects_cross_namespace(self, tmp_path: Path) -> None:
        """#1: deleting an ns-b quarantined row from ns-a returns 0 + row survives."""
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("Q-1", "project:b"))

        deleted = delete_quarantined_entries(cfg, namespace="project:a", memory_id="Q-1")

        assert deleted == 0
        # Row must still be visible in its own namespace.
        survivors = list_quarantined_entries(cfg, namespace="project:b")
        assert [e.id for e in survivors] == ["Q-1"]

    def test_delete_by_id_same_namespace_succeeds(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("Q-2", "project:a"))

        deleted = delete_quarantined_entries(cfg, namespace="project:a", memory_id="Q-2")

        assert deleted == 1
        assert list_quarantined_entries(cfg, namespace="project:a") == []

    def test_delete_by_id_missing_returns_zero(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        deleted = delete_quarantined_entries(cfg, namespace="project:a", memory_id="nope")
        assert deleted == 0


class TestQuarantineRequiresQuarantinedFlag:
    def test_delete_by_id_rejects_non_quarantined_row(self, tmp_path: Path) -> None:
        """#6: a row in the quarantine DB without quarantined=true is not deletable by id."""
        cfg = _cfg(tmp_path)
        # Write a non-quarantined row directly into the quarantine DB.
        from trw_memory.security._runtime_quarantine import open_quarantine_backend

        with open_quarantine_backend(cfg) as backend:
            backend.store(_entry("N-1", "project:a"))  # no quarantined metadata

        deleted = delete_quarantined_entries(cfg, namespace="project:a", memory_id="N-1")

        assert deleted == 0
        # Row survives — it was never quarantined.
        from trw_memory.security._runtime_quarantine import open_quarantine_backend

        with open_quarantine_backend(cfg) as backend:
            assert backend.get("N-1") is not None


class TestQuarantineListNoTruncation:
    def test_list_returns_actor_entry_past_window(self, tmp_path: Path) -> None:
        """#2: an actor-tagged quarantined entry beyond limit*5 is still listed."""
        cfg = _cfg(tmp_path)
        # Seed many entries for a noise actor, then one for the target actor.
        # limit=2 -> window of 10 in the old buggy path; put target at position 30.
        for i in range(30):
            store_quarantined_entry(cfg, _entry(f"NOISE-{i:02d}", "project:a", actor="noise"))
        store_quarantined_entry(cfg, _entry("TARGET", "project:a", actor="target"))

        found = list_quarantined_entries(cfg, namespace="project:a", actor="target", limit=2)

        assert [e.id for e in found] == ["TARGET"]
