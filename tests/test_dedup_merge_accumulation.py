"""Behavior tests: merge_entries preserves accumulation fields.

Before the fix, merge_entries dropped q_value, q_observations, access_count,
recall_count, helpful_count, sessions_surfaced, assertions, and protection_tier
(consolidation._create_consolidated_entry handled them correctly, but the dedup
merge path did not). These tests assert the cumulative/max/union semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.lifecycle.dedup import merge_entries
from trw_memory.models.memory import (
    Assertion,
    AssertionType,
    MemoryEntry,
    ProtectionTier,
)


def _entry(entry_id: str, **kwargs: object) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    base: dict[str, object] = {
        "id": entry_id,
        "content": "shared content",
        "detail": "",
        "created_at": now,
        "updated_at": now,
    }
    base.update(kwargs)
    return MemoryEntry(**base)  # type: ignore[arg-type]


class TestMergeAccumulationFields:
    def test_q_value_takes_max(self) -> None:
        existing = _entry("e1", q_value=0.4)
        new_entry = _entry("e2", q_value=0.9)
        updated = merge_entries(existing, new_entry)
        assert updated.q_value == 0.9

    def test_q_value_existing_wins_when_higher(self) -> None:
        existing = _entry("e1", q_value=0.9)
        new_entry = _entry("e2", q_value=0.4)
        updated = merge_entries(existing, new_entry)
        assert updated.q_value == 0.9

    def test_counters_are_summed(self) -> None:
        existing = _entry(
            "e1",
            q_observations=3,
            access_count=5,
            recall_count=7,
            helpful_count=2,
            sessions_surfaced=4,
        )
        new_entry = _entry(
            "e2",
            q_observations=2,
            access_count=1,
            recall_count=3,
            helpful_count=6,
            sessions_surfaced=1,
        )
        updated = merge_entries(existing, new_entry)
        assert updated.q_observations == 5
        assert updated.access_count == 6
        assert updated.recall_count == 10
        assert updated.helpful_count == 8
        assert updated.sessions_surfaced == 5

    def test_protection_tier_takes_stronger(self) -> None:
        existing = _entry("e1", protection_tier=ProtectionTier.NORMAL)
        new_entry = _entry("e2", protection_tier=ProtectionTier.CRITICAL)
        updated = merge_entries(existing, new_entry)
        # use_enum_values=True stores the string form.
        assert updated.protection_tier == ProtectionTier.CRITICAL.value

    def test_protection_tier_existing_wins_when_stronger(self) -> None:
        existing = _entry("e1", protection_tier=ProtectionTier.PERMANENT)
        new_entry = _entry("e2", protection_tier=ProtectionTier.LOW)
        updated = merge_entries(existing, new_entry)
        assert updated.protection_tier == ProtectionTier.PERMANENT.value

    def test_assertions_are_unioned(self) -> None:
        a1 = Assertion(type=AssertionType.GREP_PRESENT, pattern="foo", target="src/a.py")
        a2 = Assertion(type=AssertionType.GREP_PRESENT, pattern="bar", target="src/b.py")
        # a3 is a duplicate of a1 by (type, pattern, target) → must be deduped.
        a3 = Assertion(type=AssertionType.GREP_PRESENT, pattern="foo", target="src/a.py")

        existing = _entry("e1", assertions=[a1])
        new_entry = _entry("e2", assertions=[a2, a3])
        updated = merge_entries(existing, new_entry)

        keys = {(a.type, a.pattern, a.target) for a in updated.assertions}
        assert keys == {
            (AssertionType.GREP_PRESENT.value, "foo", "src/a.py"),
            (AssertionType.GREP_PRESENT.value, "bar", "src/b.py"),
        }
        assert len(updated.assertions) == 2
