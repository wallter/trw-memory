"""PRD-SEC-016 round-5 -- discovery_entries' identity check around the read-only vector-index connect.

``WarmTierStore.discovery_entries``'s vector-scoring branch opens the warm
sidecar's vector index read-only through ``connect_registered``, which pins the
file's inode across the connect and refuses a swap (``StorageError``); the
branch degrades that refusal to "vectors unavailable".
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers._warm import WarmTierStore

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NS = "project:t"


def _payload(i: int) -> dict[str, object]:
    from trw_memory.models.memory import MemoryEntry

    entry = MemoryEntry(id=f"M-{i:04d}", content=f"content {i}", namespace=NS, created_at=T0, updated_at=T0)
    return entry.model_dump(mode="json")


def _store_with_a_vector(tmp_path: Path) -> WarmTierStore:
    """A warm store with ONE real vector, so ``warm.db`` exists with a populated vec_index."""
    store = WarmTierStore(tmp_path)
    store.warm_add("M-0001", _payload(1), [0.1, 0.2, 0.3, 0.4])
    db_path = tmp_path / "memory" / "warm.db"
    if not db_path.exists():
        pytest.skip("sqlite-vec unavailable: warm.db was never created")
    return store


def test_a_db_identity_change_between_stat_and_connect_degrades_to_unscored_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A swap between the pre-connect stat and the vector-index open must never silently score a
    caller against the wrong file -- it degrades to 'vectors unavailable', the same fail-open
    contract every other failure in this branch already gets."""
    store = _store_with_a_vector(tmp_path)
    try:
        from tests._swap_after_pin import swap_after_pin

        swap_after_pin(monkeypatch, tmp_path / "memory" / "warm.db")

        import structlog.testing

        with structlog.testing.capture_logs() as logs:
            rows = store.discovery_entries([0.1, 0.2, 0.3, 0.4], covered_ids=frozenset(), namespace=NS)

        assert [row["id"] for row in rows] == ["M-0001"]
        assert "_tier_relevance" not in rows[0], "a detected identity mismatch must not score the row"
        assert any(entry.get("event") == "warm_tier_db_identity_changed_during_open" for entry in logs)
    finally:
        store.close()


def test_an_unchanged_db_through_discovery_does_not_trip_the_identity_check(tmp_path: Path) -> None:
    """Regression control: the identity check must not false-positive on the ordinary, unraced path.

    (Vector admission itself needs a registered embedding space this test
    does not set up -- that is orthogonal to the identity check, which is
    what this test exists to prove did not misfire.)
    """
    import structlog.testing

    store = _store_with_a_vector(tmp_path)
    try:
        with structlog.testing.capture_logs() as logs:
            rows = store.discovery_entries([0.1, 0.2, 0.3, 0.4], covered_ids=frozenset(), namespace=NS)

        assert [row["id"] for row in rows] == ["M-0001"]
        assert not any(entry.get("event") == "warm_tier_db_identity_changed_during_open" for entry in logs)
    finally:
        store.close()
