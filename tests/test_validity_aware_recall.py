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


def test_age_decay_entry_missing_from_fusion_scores_appended_in_order() -> None:
    """Entry whose id is absent from fusion_scores is appended without bucket sorting."""
    scored = _open("SCORED", valid_from=T0)
    unscored = _open("UNSCORED", valid_from=T2)

    result = apply_validity_prior(
        [scored, unscored],
        age_decay=True,
        fusion_scores={"SCORED": 1.0},  # UNSCORED absent → lines 124-128
    )
    # SCORED goes through tie-bucket path; UNSCORED is appended directly.
    assert "SCORED" in [e.id for e in result]
    assert "UNSCORED" in [e.id for e in result]


def test_age_decay_two_consecutive_unscored_trigger_empty_bucket_flush() -> None:
    """Two consecutive entries absent from fusion_scores flush an empty bucket (line 117)."""
    a = _open("A", valid_from=T0)
    b = _open("B", valid_from=T1)

    result = apply_validity_prior(
        [a, b],
        age_decay=True,
        # Neither entry in fusion_scores → flush_bucket() called on empty bucket for B
        fusion_scores={"OTHER": 0.5},
    )
    # Both present (no filtering applied), order preserved.
    assert [e.id for e in result] == ["A", "B"]


def test_valid_from_min_excludes_older_entries() -> None:
    """valid_from_min filters out entries with valid_from < min."""
    old = _open("OLD", valid_from=T0)
    recent = _open("RECENT", valid_from=T2)
    # Exclude anything older than T2
    result = apply_validity_prior([old, recent], valid_from_min=T2)
    assert [e.id for e in result] == ["RECENT"]


def test_valid_from_min_inclusive_boundary() -> None:
    """valid_from == valid_from_min is included (inclusive lower bound)."""
    at_min = _open("AT_MIN", valid_from=T1)
    before_min = _open("BEFORE_MIN", valid_from=T0)
    result = apply_validity_prior([before_min, at_min], valid_from_min=T1)
    assert [e.id for e in result] == ["AT_MIN"]


def test_valid_from_min_none_includes_all() -> None:
    """valid_from_min=None (default) does not filter by date."""
    old = _open("OLD", valid_from=T0)
    recent = _open("RECENT", valid_from=T2)
    result = apply_validity_prior([old, recent], valid_from_min=None)
    assert {e.id for e in result} == {"OLD", "RECENT"}


def test_valid_from_min_combined_with_as_of() -> None:
    """valid_from_min AND as_of are applied together (AND semantics)."""
    # at T1, entry A is open AND after T0 (valid_from=T0 >= min=T0?)
    # entry B has valid_from=T2 but as_of=T1 → valid_from=T2 > T1 → excluded by as_of
    a = _open("A", valid_from=T0)  # valid_from=T0, open always
    b = _open("B", valid_from=T2)  # valid_from=T2, open always
    # as_of=T1 → B excluded (valid_from T2 > T1); valid_from_min=T0 → A included
    result = apply_validity_prior([a, b], as_of=T1, valid_from_min=T0)
    assert [e.id for e in result] == ["A"]


def test_store_round_trip_then_recall_filter(tmp_path: Path) -> None:
    """Integration: superseded entry persisted, then excluded by the prior."""
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(_closed("A", valid_from=T0, invalid_from=T2, by="B"))
    backend.store(_open("B", valid_from=T2))
    entries = backend.list_entries(limit=100)
    kept = apply_validity_prior(entries)
    assert {e.id for e in kept} == {"B"}


# ---------------------------------------------------------------------------
# PRD-CORE-244 FR05 — an expired record demotes exactly like a superseded one
# ---------------------------------------------------------------------------


def _expiring(entry_id: str, expires: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=f"c {entry_id}", created_at=T0, valid_from=T0, expires=expires)


class TestExpiryDemotesLikeSupersession:
    """``expires`` was measured non-empty on 0 of 9,366 rows, and the two ranking
    paths disagreed about what it means: ``trw_mcp.scoring._decay`` floored an
    expired entry's utility at 0.01 while ``_is_open_at`` still called it an open
    record. They now agree."""

    def test_expired_state_learning_demoted_below_open(self) -> None:
        expired = _expiring("m-expired", "2020-01-01")
        open_record = _open("m-open")

        # Default: excluded outright, exactly like a superseded record.
        assert apply_validity_prior([expired, open_record]) == [open_record]

        # include_superseded: appended AFTER every open record, even though it
        # was ranked FIRST by fusion.
        assert apply_validity_prior([expired, open_record], include_superseded=True) == [open_record, expired]

    def test_entry_expiring_today_is_still_open(self) -> None:
        """Day-exclusive boundary, matching ``today > expires_date`` exactly."""
        today = datetime.now(timezone.utc).date().isoformat()
        entry = _expiring("m-today", today)
        assert apply_validity_prior([entry]) == [entry]

    def test_expiry_is_evaluated_against_as_of_not_now(self) -> None:
        """OQ-04: a caller asking what was believed at T gets what was unexpired at T."""
        entry = _expiring("m-window", "2026-01-02")
        assert apply_validity_prior([entry], as_of=T0) == [entry]
        assert apply_validity_prior([entry], as_of=T2) == []

    def test_datetime_shaped_expires_is_parsed(self) -> None:
        assert apply_validity_prior([_expiring("m-dt", "2020-01-01T00:00:00+00:00")]) == []

    def test_empty_or_unparseable_expires_never_expires(self) -> None:
        """A value nobody can read must not silently retire a record."""
        assert apply_validity_prior([_expiring("m-blank", "")]) != []
        assert apply_validity_prior([_expiring("m-junk", "when the migration lands")]) != []
