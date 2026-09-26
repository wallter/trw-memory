"""Q1 (2026-09-24 security review): a caller-set ``quarantined`` flag must never
bypass the audited store gates, and an approval must re-validate before promotion.

Prior behaviour: ``prepare_entry_for_store`` read ``metadata.get("quarantined")
== "true"`` straight off caller-supplied metadata to decide "skip rate-limit /
PII / anomaly, this is a held trust-score entry" -- so ``memory_store(content=
<a PII payload>, metadata={"quarantined": "true"})`` took that branch, skipped
every gate, and landed in the quarantine DB unchecked; approving it then
promoted the unscanned content straight to the active store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_quarantine import open_quarantine_backend
from trw_memory.security.poisoning import RESERVED_SYSTEM_METADATA_KEYS
from trw_memory.security.runtime import prepare_entry_for_store, review_quarantined_entry, store_quarantined_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _cfg(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "mem"), pii_enabled=True)


class TestCallerCannotForgeTheQuarantineShortCircuit:
    def test_a_forged_quarantined_flag_does_not_skip_pii_blocking(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        entry = MemoryEntry(
            id="M-forged",
            content="sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD",  # looks like an API key
            namespace="project:default",
            metadata={"quarantined": "true"},
        )
        with SQLiteBackend(tmp_path / "active.db") as backend, pytest.raises(PIIBlockError):
            prepare_entry_for_store(entry, backend=backend, config=cfg, session_id="s1")

    @pytest.mark.parametrize("key", RESERVED_SYSTEM_METADATA_KEYS)
    def test_every_reserved_key_is_stripped_before_the_quarantine_decision(self, tmp_path: Path, key: str) -> None:
        cfg = _cfg(tmp_path)
        entry = MemoryEntry(
            id=f"M-{key}",
            content="benign content, nothing to detect",
            namespace="project:default",
            metadata={key: "true" if key != "reviewed_by" else "someone"},
        )
        with SQLiteBackend(tmp_path / "active.db") as backend:
            prepared = prepare_entry_for_store(entry, backend=backend, config=cfg, session_id="s1")
        assert key not in prepared.entry.metadata
        assert prepared.quarantined is False

    def test_a_genuine_trust_quarantine_still_quarantines(self, tmp_path: Path) -> None:
        """Non-vacuity: stripping must not also break the LEGITIMATE short-circuit."""
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            enable_trust_scoring=True,
            trust_scoring_mode="enforce",
            trust_score_threshold=0.99,  # force below-threshold on any content
        )
        entry = MemoryEntry(id="M-held", content="ordinary content", namespace="project:default")
        with SQLiteBackend(tmp_path / "active.db") as backend:
            prepared = prepare_entry_for_store(entry, backend=backend, config=cfg, session_id="s1")
        assert prepared.quarantined is True
        assert prepared.entry.metadata.get("quarantined") == "true"


class TestApprovalRevalidatesBeforePromotion:
    def test_approving_a_pii_bearing_quarantined_entry_is_blocked_not_promoted(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        # Seed the quarantine DB directly -- however an entry got there (a
        # legitimate anomaly hold, a forged flag closed above, or anything
        # else), approval must re-check it.
        store_quarantined_entry(
            cfg,
            MemoryEntry(
                id="M-pii-held",
                content="sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD",
                namespace="project:default",
            ),
        )
        active = SQLiteBackend(tmp_path / "active.db")
        try:
            outcome = review_quarantined_entry(
                cfg,
                active_backend=active,
                learning_id="M-pii-held",
                decision="approve",
                reviewer_id="maintainer",
                namespace="project:default",
            )
            assert outcome["status"] == "blocked"
            assert active.get("M-pii-held", namespace="project:default") is None
        finally:
            active.close()
        # Still sitting in quarantine, not silently dropped.
        with open_quarantine_backend(cfg) as quarantine_backend:
            assert quarantine_backend.get("M-pii-held", namespace="project:default") is not None

    def test_approving_a_clean_quarantined_entry_still_succeeds(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        store_quarantined_entry(
            cfg, MemoryEntry(id="M-clean-held", content="nothing sensitive here", namespace="project:default")
        )
        active = SQLiteBackend(tmp_path / "active.db")
        try:
            outcome = review_quarantined_entry(
                cfg,
                active_backend=active,
                learning_id="M-clean-held",
                decision="approve",
                reviewer_id="maintainer",
                namespace="project:default",
            )
            assert outcome["status"] == "approved"
            promoted = active.get("M-clean-held", namespace="project:default")
            assert promoted is not None
            assert promoted.metadata.get("quarantined") == "false"
        finally:
            active.close()

    def test_a_credential_in_nudge_line_also_blocks_approval(self, tmp_path: Path) -> None:
        """Closes the adversarial-review nudge_line PII-scan gap alongside Q1."""
        cfg = _cfg(tmp_path)
        store_quarantined_entry(
            cfg,
            MemoryEntry(
                id="M-nudge-pii",
                content="benign",
                namespace="project:default",
                nudge_line="use sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD to auth",
            ),
        )
        active = SQLiteBackend(tmp_path / "active.db")
        try:
            outcome = review_quarantined_entry(
                cfg,
                active_backend=active,
                learning_id="M-nudge-pii",
                decision="approve",
                reviewer_id="maintainer",
                namespace="project:default",
            )
            assert outcome["status"] == "blocked"
        finally:
            active.close()
