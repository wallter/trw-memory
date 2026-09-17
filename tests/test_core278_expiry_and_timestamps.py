"""PRD-CORE-278 FR05/FR06 — one expiry contract, and two different timestamps.

Both requirements exist because a fact was being answered in more than one place:
"is this entry expired" had two predicates that disagreed, and "when did this
entry last change" was written by code that had not changed anything.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import patch

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle._recall import _expires_in_past, drop_expired_entries, rank_by_utility
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.validity_prior import expiry_has_passed
from trw_memory.tools.recall import memory_recall_impl

NAMESPACE = "project:default"


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


class TestOneExpiryContract:
    def test_instant_expiry_takes_effect_at_the_instant(self) -> None:
        """sub_4-nL1paSXxQx41fH: a checkpoint that expired an hour ago was still
        returned, because the only predicate reduced it to a calendar date."""
        assert expiry_has_passed(_iso(timedelta(hours=-1))) is True
        assert expiry_has_passed(_iso(timedelta(hours=1))) is False

    def test_date_only_expiry_stays_day_exclusive(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        assert expiry_has_passed(today) is False
        assert expiry_has_passed(yesterday) is True

    def test_offset_instants_are_compared_in_utc(self) -> None:
        # "2026-07-01T18:30:00-04:00" is 22:30 UTC — already past at 23:00 UTC,
        # even though its own calendar date is still the reference date. Reducing
        # the value to a date (the old parser) called both of these unexpired.
        reference = datetime(2026, 7, 1, 23, 0, tzinfo=timezone.utc)
        assert expiry_has_passed("2026-07-01T18:30:00-04:00", reference_time=reference) is True
        assert expiry_has_passed("2026-07-01T19:30:00-04:00", reference_time=reference) is False

    def test_malformed_expiry_never_expires(self) -> None:
        for value in ("", "   ", "when the migration ships", "never", "2025-13-99"):
            assert expiry_has_passed(value) is False
            assert _expires_in_past({"expires": value}) is False

    def test_metadata_expiry_is_resolved_like_the_top_level_field(self) -> None:
        assert _expires_in_past({"metadata": {"expires": _iso(timedelta(hours=-1))}}) is True

    def test_the_ranker_no_longer_removes_expired_rows(self) -> None:
        rows: list[dict[str, object]] = [
            {"id": "expired", "content": "stale", "expires": _iso(timedelta(hours=-1))},
            {"id": "live", "content": "stale"},
        ]
        assert {str(row["id"]) for row in rank_by_utility(rows, ["stale"])} == {"expired", "live"}
        assert {str(row["id"]) for row in drop_expired_entries(rows)} == {"live"}

    def _recall(self, exclude_expired: bool, *, with_tier_entry: bool = False) -> list[str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with create_backend_from_config(cfg, NAMESPACE) as backend:
                backend.store(
                    MemoryEntry(
                        id="ckpt-past",
                        content="checkpoint ckpt-past: continue the migration",
                        namespace=NAMESPACE,
                        expires=_iso(timedelta(hours=-1)),
                    )
                )
                backend.store(
                    MemoryEntry(
                        id="ckpt-future",
                        content="checkpoint ckpt-future: continue the migration",
                        namespace=NAMESPACE,
                        expires=_iso(timedelta(days=14)),
                    )
                )
                if with_tier_entry:
                    from trw_memory.lifecycle.tiers._runtime import get_tier_manager

                    manager = get_tier_manager(cfg, NAMESPACE)
                    manager.warm_add(
                        "ckpt-tier",
                        MemoryEntry(
                            id="ckpt-tier",
                            content="checkpoint ckpt-tier: continue the migration",
                            namespace=NAMESPACE,
                            expires=_iso(timedelta(hours=-1)),
                            importance=0.95,
                        ).model_dump(mode="json"),
                        [1.0, 0.0],
                    )
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    result = memory_recall_impl(
                        "checkpoint continue the migration",
                        NAMESPACE,
                        backend=backend,
                        config=cfg,
                        limit=5,
                        include_org_memories=False,
                        exclude_expired=exclude_expired,
                    )
        return [str(row["id"]) for row in cast("list[dict[str, object]]", result["memories"])]

    def test_expired_entry_is_absent_with_exclude_expired_true(self) -> None:
        ids = self._recall(exclude_expired=True)
        assert "ckpt-future" in ids
        assert "ckpt-past" not in ids

    def test_expired_entry_stays_absent_with_exclude_expired_false(self) -> None:
        """The contract this PRD decides: expiry is INELIGIBILITY.

        ``exclude_expired=False`` cannot re-admit an expired record, because the
        validity prior inside retrieval already excluded it. Asserting the flag
        resurrects the row would be asserting a behaviour the retrieval gate
        makes unreachable — so the contract is stated, and tested, as it is.
        """
        ids = self._recall(exclude_expired=False)
        assert "ckpt-past" not in ids

    def test_expired_entry_cannot_re_enter_through_the_tier_merge(self) -> None:
        ids = self._recall(exclude_expired=True, with_tier_entry=True)
        assert "ckpt-tier" not in ids
        assert "ckpt-past" not in ids


class TestMaintenanceDoesNotStampContentTime:
    def _backend(self, td: str, backend_kind: str) -> Any:
        cfg = MemoryConfig(storage_backend=backend_kind, storage_path=td)
        return create_backend_from_config(cfg, NAMESPACE)

    def test_recall_access_leaves_updated_at_alone(self) -> None:
        with tempfile.TemporaryDirectory() as td, self._backend(td, "sqlite") as backend:
            backend.store(MemoryEntry(id="a", content="first", namespace=NAMESPACE))
            before = backend.get("a", namespace=NAMESPACE)
            time.sleep(0.01)
            backend.increment_recall_access(["a"])
            after = backend.get("a", namespace=NAMESPACE)
            assert after.updated_at == before.updated_at
            assert after.last_accessed_at != before.last_accessed_at
            assert after.access_count == before.access_count + 1
            # Replication still sees the row: selection is on sync_seq, not on
            # updated_at (sync/delta.py).
            assert after.sync_seq > before.sync_seq

    def test_a_content_update_still_moves_updated_at(self) -> None:
        with tempfile.TemporaryDirectory() as td, self._backend(td, "sqlite") as backend:
            backend.store(MemoryEntry(id="a", content="first", namespace=NAMESPACE))
            before = backend.get("a", namespace=NAMESPACE)
            time.sleep(0.01)
            backend.update("a", namespace=NAMESPACE, content="edited")
            assert backend.get("a", namespace=NAMESPACE).updated_at > before.updated_at

    def test_bookkeeping_only_update_does_not_move_updated_at(self) -> None:
        with tempfile.TemporaryDirectory() as td, self._backend(td, "sqlite") as backend:
            backend.store(MemoryEntry(id="a", content="first", namespace=NAMESPACE))
            before = backend.get("a", namespace=NAMESPACE)
            time.sleep(0.01)
            backend.update("a", namespace=NAMESPACE, sync_seq=99, last_synced_at=None)
            after = backend.get("a", namespace=NAMESPACE)
            assert after.updated_at == before.updated_at
            assert after.sync_seq == 99

    def test_yaml_backend_applies_the_same_rule(self) -> None:
        with tempfile.TemporaryDirectory() as td, self._backend(td, "yaml") as backend:
            backend.store(MemoryEntry(id="a", content="first", namespace=NAMESPACE))
            before = backend.get("a", namespace=NAMESPACE)
            time.sleep(0.01)
            # The YAML backend's update() has no namespace routing parameter, so
            # passing one would land in the field set as a content field.
            backend.update("a", access_count=7)
            after = backend.get("a", namespace=NAMESPACE)
            assert after.updated_at == before.updated_at
            assert after.access_count == 7

    def test_newest_first_listing_follows_content_time(self) -> None:
        """sub_ruTTqov1kvAvpbiJ: the first entry written sorted above the sixty
        written after it, because a maintenance pass restamped them all."""
        with tempfile.TemporaryDirectory() as td, self._backend(td, "sqlite") as backend:
            backend.store(MemoryEntry(id="kv-first", content="kubernetes_cluster: cx1-prod", namespace=NAMESPACE))
            time.sleep(0.01)
            for index in range(3):
                backend.store(MemoryEntry(id=f"filler-{index}", content=f"filler {index}", namespace=NAMESPACE))
                time.sleep(0.005)
            backend.increment_recall_access(["kv-first"])
            listed = [entry.id for entry in backend.list_entries(namespace=NAMESPACE, limit=10)]
            assert listed[0] != "kv-first"
            assert listed[-1] == "kv-first"
