"""Warm tier discovery returns only rows the caller has not already ranked.

Hybrid recall passes the ids its primary-store pool covered; tier discovery
drops those rows anyway. Copying every sidecar row to then drop almost all of
them made each recall O(sidecar rows) (~12 ms at 5,000, ~60 ms at 20,000). The
covered rows are now never materialised, while the namespace containment check
discovery applies to every row still covers them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.lifecycle.tiers._warm_sidecar_cache import parse_sidecar
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.namespace_scope import NamespaceScopeError

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NS = "project:t"


def _payload(i: int, namespace: str = NS, content: str = "") -> dict[str, object]:
    entry = MemoryEntry(
        id=f"M-{i:04d}", content=content or f"content {i}", namespace=namespace, created_at=T0, updated_at=T0
    )
    return entry.model_dump(mode="json")


def _store(tmp_path: Path, n: int = 12) -> WarmTierStore:
    store = WarmTierStore(tmp_path)
    store.warm_add_many([(f"M-{i:04d}", _payload(i), None) for i in range(n)])
    return store


def test_covered_rows_are_omitted_and_the_rest_keep_file_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Supersede one uncovered row so file order differs from id order.
    store.warm_add(f"M-{3:04d}", _payload(3, content="changed"), None)
    covered = frozenset(f"M-{i:04d}" for i in range(12) if i not in (1, 3, 7))

    rows = store.discovery_entries(None, covered_ids=covered, namespace=NS)

    assert [row["id"] for row in rows] == ["M-0001", "M-0007", "M-0003"]
    assert rows[2]["content"] == "changed"
    # Without covered ids every live row is returned, as before.
    assert len(store.discovery_entries(None)) == 12
    store.close()


def test_returned_rows_are_copies(tmp_path: Path) -> None:
    store = _store(tmp_path, n=3)
    rows = store.discovery_entries(None, covered_ids=frozenset({"M-0000"}), namespace=NS)
    rows[0]["_tier_relevance"] = 0.9

    again = store.discovery_entries(None, covered_ids=frozenset({"M-0000"}), namespace=NS)
    assert "_tier_relevance" not in again[0]
    store.close()


def test_a_covered_row_from_another_namespace_still_fails_the_containment_check(tmp_path: Path) -> None:
    store = _store(tmp_path, n=4)
    store.warm_add("M-9999", _payload(9999, namespace="project:other"), None)

    with pytest.raises(NamespaceScopeError):
        store.discovery_entries(None, covered_ids=frozenset({"M-9999"}), namespace=NS)
    # Superseding the foreign row with an in-namespace one clears the check,
    # through the append path and through a fresh parse of the same bytes.
    store.warm_add("M-9999", _payload(9999), None)
    assert len(store.discovery_entries(None, covered_ids=frozenset({"M-9999"}), namespace=NS)) == 4
    parsed = parse_sidecar(tmp_path / "memory" / "warm.jsonl")
    assert not parsed.names_other_namespace(NS)
    store.close()


def test_the_containment_check_sees_a_fresh_parse_of_a_foreign_row(tmp_path: Path) -> None:
    store = _store(tmp_path, n=2)
    store.warm_add("M-0100", _payload(100, namespace="project:other"), None)
    fresh = WarmTierStore(tmp_path)  # another process: parses the file from disk

    with pytest.raises(NamespaceScopeError):
        fresh.discovery_entries(None, covered_ids=frozenset({"M-0100"}), namespace=NS)
    store.close()
    fresh.close()
