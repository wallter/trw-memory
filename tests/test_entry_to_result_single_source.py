"""Guards the DRY collapse of the four local ``_entry_to_result`` shims.

Historically ``_client_recall``, ``_client_recall_hybrid``,
``_client_recall_helpers`` and ``_client_forget_search`` each carried an
identical three-line ``_entry_to_result`` wrapper that lazily re-imported the
canonical ``_client_distilled_tiering.entry_to_result``. Those wrappers were
deleted in favour of a single module-top import alias.

These tests pin the invariant that all four call sites resolve to the ONE
canonical implementation and therefore produce byte-identical result dicts —
so the shims can never silently diverge again.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.conftest import make_entry
from trw_memory._client_distilled_tiering import entry_to_result as canonical
from trw_memory._client_forget_search import _entry_to_result as forget_variant
from trw_memory._client_recall import _entry_to_result as recall_variant
from trw_memory._client_recall_helpers import _entry_to_result as helpers_variant
from trw_memory._client_recall_hybrid import _entry_to_result as hybrid_variant
from trw_memory.models.memory import MemoryEntry

_VARIANTS = {
    "_client_recall": recall_variant,
    "_client_recall_hybrid": hybrid_variant,
    "_client_recall_helpers": helpers_variant,
    "_client_forget_search": forget_variant,
}


def _entries() -> list[MemoryEntry]:
    """Representative entries exercising every branch of ``entry_to_result``."""
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    plain = make_entry(
        entry_id="M-plain",
        content="use absolute paths",
        detail="d",
        tags=["gotcha", "path"],
        importance=0.9,
        created_at=now,
        last_accessed_at=now,
    )
    # None last_accessed_at -> empty-string branch (make_entry coerces None to
    # now, so force the None via model_copy to genuinely exercise the branch).
    never_accessed = make_entry(entry_id="M-fresh", created_at=now).model_copy(update={"last_accessed_at": None})
    # metadata with anomaly_dimension + parseable z_score -> extra keys added.
    anomalous = make_entry(
        entry_id="M-anom",
        created_at=now,
        metadata={"anomaly_dimension": "size", "z_score": "2.5", "extra": "keep"},
    )
    # metadata with an UNparseable z_score -> ValueError swallowed, key absent.
    bad_z = make_entry(
        entry_id="M-badz",
        created_at=now,
        metadata={"z_score": "not-a-float"},
    )
    # expires set -> expires branch.
    expiring = make_entry(entry_id="M-exp", created_at=now).model_copy(update={"expires": "2027-12-31T00:00:00+00:00"})
    return [plain, never_accessed, anomalous, bad_z, expiring]


def test_all_variants_are_the_canonical_function() -> None:
    """Each module's ``_entry_to_result`` is the exact canonical object (single source)."""
    for name, variant in _VARIANTS.items():
        assert variant is canonical, f"{name}._entry_to_result diverged from canonical"


@pytest.mark.parametrize("score", [0.0, 0.4321, 1.0])
def test_all_call_sites_produce_identical_results(score: float) -> None:
    """Every call site yields the SAME dict the canonical fn would for the same input."""
    for entry in _entries():
        expected = canonical(entry, score=score)
        for name, variant in _VARIANTS.items():
            got = variant(entry, score=score)
            assert got == expected, f"{name} produced a divergent result for {entry.id}"


def test_canonical_result_field_values() -> None:
    """Pin the actual field values so the parity assertion is non-vacuous."""
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    entry = make_entry(
        entry_id="M-anom",
        content="c",
        detail="det",
        tags=["t"],
        importance=0.75,
        created_at=now,
        metadata={"anomaly_dimension": "size", "z_score": "2.5"},
    ).model_copy(update={"last_accessed_at": None})
    result = recall_variant(entry, score=0.5)
    assert result["memory_id"] == "M-anom"
    assert result["score"] == 0.5
    assert result["_relevance_hint"] == 0.5
    assert result["source"] == "local"
    assert result["last_accessed_at"] == ""  # None -> empty string
    assert result["anomaly_dimension"] == "size"
    assert result["z_score"] == 2.5  # parsed to float
    assert result["created_at"] == now.isoformat()
