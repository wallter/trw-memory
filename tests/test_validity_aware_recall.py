"""PRD-CORE-194 FR02/FR03 — honest-state distinction + validity-aware recall.

FR02: TTL/decay downgrades confidence/importance and NEVER sets invalid_from;
supersession sets invalid_from/invalidated_by and never changes confidence.

FR03: superseded records are excluded by default; include_superseded re-includes
them ranked strictly below open records (positional append, OQ2); as_of re-scopes
the window test; an age-decay term gives newer valid_from a non-negative
advantage without disturbing fusion order.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trw_memory._graph_decay import apply_importance_decay
from trw_memory.models.memory import Confidence, MemoryEntry
from trw_memory.retrieval.validity_prior import apply_validity_prior
from trw_memory.storage.sqlite_backend import SQLiteBackend

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 3, tzinfo=timezone.utc)


def _open(entry_id: str, *, valid_from: datetime = T0) -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=f"c {entry_id}", created_at=valid_from, valid_from=valid_from)


def _closed(entry_id: str, *, valid_from: datetime, invalid_from: datetime, by: str) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"c {entry_id}",
        created_at=valid_from,
        valid_from=valid_from,
        invalid_from=invalid_from,
        invalidated_by=by,
    )


# ---------------------------------------------------------------------------
# FR02 — honest-state discrimination
# ---------------------------------------------------------------------------


def test_ttl_decay_does_not_set_invalid_from(tmp_path: Path) -> None:
    """A decay sweep downgrades importance but leaves the validity window open."""
    entry = MemoryEntry(id="M-d", content="c", importance=0.8, cross_validated=True)
    assert entry.invalid_from is None

    decayed = apply_importance_decay(entry)

    assert decayed.importance < entry.importance  # confidence/importance downgraded
    assert decayed.invalid_from is None  # window NOT closed by decay
    assert decayed.invalidated_by is None
    assert decayed.validity_state() == "open"


def test_supersession_preserves_confidence() -> None:
    """Closing a window does not change the record's confidence (a truth statement)."""
    entry = MemoryEntry(id="M-s", content="c", confidence=Confidence.HIGH)
    closed = entry.model_copy(update={"invalid_from": T1, "invalidated_by": "M-new"})
    # use_enum_values=True stores the string value.
    assert closed.confidence == Confidence.HIGH.value
    assert closed.validity_state() == "superseded"


def test_derived_state_orthogonal_to_unverified_stale() -> None:
    """validity_state reflects only the window; confidence is the orthogonal signal."""
    superseded_high = MemoryEntry(
        id="M-x", content="c", confidence=Confidence.HIGH, created_at=T0, invalid_from=T1, invalidated_by="M-y"
    )
    open_stale = MemoryEntry(id="M-z", content="c", confidence=Confidence.UNVERIFIED)
    assert superseded_high.validity_state() == "superseded"
    assert open_stale.validity_state() == "open"


# ---------------------------------------------------------------------------
# FR03 — validity prior
# ---------------------------------------------------------------------------


def test_superseded_excluded_by_default_and_as_of_reincludes() -> None:
    a = _closed("A", valid_from=T0, invalid_from=T2, by="B")
    b = _open("B", valid_from=T2)
    ordered = [a, b]

    # Default: A (superseded) excluded; B returned.
    default = apply_validity_prior(ordered)
    assert [e.id for e in default] == ["B"]

    # as_of=T1 (inside A's [T0,T2) window): A eligible; B (opens at T2) excluded.
    as_of = apply_validity_prior(ordered, as_of=T1)
    assert [e.id for e in as_of] == ["A"]


def test_superseded_ranks_below_open() -> None:
    """include_superseded=True appends superseded AFTER every open record (OQ2)."""
    # A is superseded but fused FIRST (highest relevance); B is open, fused second.
    a = _closed("A", valid_from=T0, invalid_from=T1, by="B")
    b = _open("B", valid_from=T1)
    ordered = [a, b]

    result = apply_validity_prior(ordered, include_superseded=True)
    # Positional: open B first despite A's higher fused position; A appended last.
    assert [e.id for e in result] == ["B", "A"]


def test_age_decay_without_scores_preserves_fusion_order() -> None:
    """Without fused scores, age decay cannot prove ties and preserves order."""
    older = _open("OLD", valid_from=T0)
    newer = _open("NEW", valid_from=T2)

    result = apply_validity_prior([older, newer], age_decay=True)

    assert [e.id for e in result] == ["OLD", "NEW"]


def test_age_decay_breaks_fused_score_ties_only() -> None:
    """Among score-tied open records the newer valid_from gets the advantage."""
    newer = _open("NEW", valid_from=T2)
    older = _open("OLD", valid_from=T0)

    result = apply_validity_prior(
        [older, newer],
        age_decay=True,
        fusion_scores={"OLD": 1.0, "NEW": 1.0},
    )

    assert [e.id for e in result] == ["NEW", "OLD"]


def test_age_decay_does_not_reorder_non_ties() -> None:
    """An older higher-scored result stays above a newer lower-scored result."""
    older = _open("OLD", valid_from=T0)
    newer = _open("NEW", valid_from=T2)

    result = apply_validity_prior(
        [older, newer],
        age_decay=True,
        fusion_scores={"OLD": 2.0, "NEW": 1.0},
    )

    assert [e.id for e in result] == ["OLD", "NEW"]


def test_as_of_open_outranks_closed_at_as_of() -> None:
    """as_of composes with include_superseded: open-at-as_of outranks closed-at."""
    # At as_of=T1: A is open (window [T0, +inf)), C closed at T1 (window [T0,T1)).
    a = _open("A", valid_from=T0)
    c = _closed("C", valid_from=T0, invalid_from=T1, by="A")
    # Fused order puts C first; the prior must float open-A above closed-C.
    result = apply_validity_prior([c, a], as_of=T1, include_superseded=True)
    assert [e.id for e in result] == ["A", "C"]


def test_store_round_trip_then_recall_filter(tmp_path: Path) -> None:
    """Integration: superseded entry persisted, then excluded by the prior."""
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(_closed("A", valid_from=T0, invalid_from=T2, by="B"))
    backend.store(_open("B", valid_from=T2))
    entries = backend.list_entries(limit=100)
    kept = apply_validity_prior(entries)
    assert {e.id for e in kept} == {"B"}
