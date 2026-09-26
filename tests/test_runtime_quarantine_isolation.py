"""Tests for SEC-001 quarantine namespace isolation + audit completeness.

Covers closure re-audit findings:
- #1: delete_quarantined_entries memory_id branch must verify namespace.
- #6: ...and must verify the entry is actually quarantined.
- #2: list_quarantined_entries must not silently truncate at limit*5.

Also covers the 2026-09-24 security review (Q3): review state must be scoped
per (namespace, learning_id), not learning_id alone, and (Q1's approval
follow-on) an approval must never silently clobber a pre-existing active row.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.runtime import (
    delete_quarantined_entries,
    get_status_history,
    list_quarantined_entries,
    review_quarantined_entry,
    store_quarantined_entry,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend


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
            assert backend.get("N-1", namespace="project:a") is not None


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


class TestQuarantineReviewNamespaceIsolation:
    """Q3: review state is keyed by (namespace, learning_id), not learning_id alone."""

    def test_a_namespace_a_review_does_not_resolve_namespace_bs_own_id(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("x", "project:a"))
        store_quarantined_entry(cfg, _entry("x", "project:b"))
        active_a = SQLiteBackend(tmp_path / "active_a.db")
        try:
            resolved = review_quarantined_entry(
                cfg,
                active_backend=active_a,
                learning_id="x",
                decision="reject",
                reviewer_id="rev-a",
                namespace="project:a",
            )
        finally:
            active_a.close()
        assert resolved["status"] == "rejected"

        # project:b's OWN "x" must still be reviewable -- not "already_resolved".
        active_b = SQLiteBackend(tmp_path / "active_b.db")
        try:
            outcome_b = review_quarantined_entry(
                cfg,
                active_backend=active_b,
                learning_id="x",
                decision="approve",
                reviewer_id="rev-b",
                namespace="project:b",
            )
        finally:
            active_b.close()
        assert outcome_b["status"] == "approved"

    def test_status_history_does_not_leak_another_namespaces_reviewer(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("x", "project:a"))
        store_quarantined_entry(cfg, _entry("x", "project:b"))
        active_a = SQLiteBackend(tmp_path / "active_a.db")
        try:
            review_quarantined_entry(
                cfg,
                active_backend=active_a,
                learning_id="x",
                decision="reject",
                reviewer_id="secret-reviewer-a",
                namespace="project:a",
            )
        finally:
            active_a.close()

        history_b = get_status_history(cfg, "x", namespace="project:b")
        assert history_b == [{"status": "quarantined", "reviewer_id": "system", "ts": history_b[0]["ts"]}]
        assert "secret-reviewer-a" not in [item["reviewer_id"] for item in history_b]

        history_a = get_status_history(cfg, "x", namespace="project:a")
        assert [item["status"] for item in history_a] == ["quarantined", "obsolete_poisoned"]
        assert history_a[-1]["reviewer_id"] == "secret-reviewer-a"


class TestApprovalNeverOverwritesALiveActiveRow:
    """Adversarial audit follow-on: approve must refuse an active-row conflict."""

    def test_approve_refuses_when_an_active_row_already_exists_at_that_id(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("M-collide", "project:a"))
        active = SQLiteBackend(tmp_path / "active.db")
        try:
            active.store(MemoryEntry(id="M-collide", content="a legitimate later write", namespace="project:a"))

            outcome = review_quarantined_entry(
                cfg,
                active_backend=active,
                learning_id="M-collide",
                decision="approve",
                reviewer_id="rev",
                namespace="project:a",
            )
            assert outcome["status"] == "conflict"
            # The legitimate row must be untouched.
            survivor = active.get("M-collide", namespace="project:a")
            assert survivor is not None
            assert survivor.content == "a legitimate later write"
        finally:
            active.close()

        # Not terminal: a caller can still resolve it (e.g. reject, or retry
        # approve once the conflicting id is freed up).
        history = get_status_history(cfg, "M-collide", namespace="project:a")
        assert [item["status"] for item in history] == ["quarantined", "approve_conflict"]

    def test_a_write_racing_the_approval_is_never_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C12 (7.0 freeze): a legitimate write between the conflict check and the store must survive.

        The racing writer is its own connection to the same store, started right
        after approve's conflict check. Approve's check and store share one write
        transaction, so that writer waits and lands after the approval, never under it.
        """
        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("M-race", "project:a"))
        active = SQLiteBackend(tmp_path / "active.db")
        racer = SQLiteBackend(tmp_path / "active.db")
        legit = MemoryEntry(id="M-race", content="a legitimate racing write", namespace="project:a")
        writer = threading.Thread(target=racer.store, args=(legit,))
        real_get = SQLiteBackend.get

        def get_then_race(self: SQLiteBackend, entry_id: str, *, namespace: str) -> MemoryEntry | None:
            found = real_get(self, entry_id, namespace=namespace)
            if self is active and not writer.is_alive() and writer.ident is None:
                writer.start()
                writer.join(timeout=1.0)  # lands now, unless the approval holds the write lock
            return found

        monkeypatch.setattr(SQLiteBackend, "get", get_then_race)
        try:
            review_quarantined_entry(
                cfg,
                active_backend=active,
                learning_id="M-race",
                decision="approve",
                reviewer_id="rev",
                namespace="project:a",
            )
            writer.join(timeout=30)
            survivor = real_get(active, "M-race", namespace="project:a")
        finally:
            active.close()
            racer.close()

        assert not writer.is_alive()
        assert survivor is not None and survivor.content == "a legitimate racing write"

    def test_a_backend_without_an_atomic_transaction_refuses_approval_and_writes_nothing(self, tmp_path: Path) -> None:
        """C12 sol review: the YAML backend's transaction is a no-op, so check-then-store could race."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        cfg = _cfg(tmp_path)
        store_quarantined_entry(cfg, _entry("M-yaml", "project:a"))
        active = YAMLBackend(tmp_path / "yaml-store")

        outcome = review_quarantined_entry(
            cfg,
            active_backend=active,
            learning_id="M-yaml",
            decision="approve",
            reviewer_id="rev",
            namespace="project:a",
        )

        assert outcome["status"] == "unsupported_backend"
        assert active.get("M-yaml", namespace="project:a") is None
        history = get_status_history(cfg, "M-yaml", namespace="project:a")
        assert [item["status"] for item in history] == ["quarantined"]
