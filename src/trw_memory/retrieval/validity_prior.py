"""PRD-CORE-194 FR03 — validity prior over an already-fused candidate list.

This is a POST-FUSION pass: it operates on the ordered ``list[MemoryEntry]``
``hybrid_search`` returns, so RRF / CombMAX fusion is never disturbed (NFR04 —
an in-memory field compare, no extra DB query). It:

1. Determines eligibility per entry against ``as_of`` (default: open-only;
   ``as_of=T``: window contains T, half-open ``[valid_from, invalid_from)``).
2. Drops ineligible (superseded / out-of-window) entries unless
   ``include_superseded=True``, in which case they are APPENDED after every
   eligible entry (positional, OQ2 — so a superseded record can never outrank an
   open one regardless of its fused score).
3. Optionally applies a small, bounded, monotone ``valid_from`` age-decay
   adjustment among the eligible entries that only ever breaks ties in favour of
   the newer record — fusion order is otherwise preserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Protocol

from trw_memory.models.memory import MemoryEntry


class ValidityFields(Protocol):
    """Read-only window fields shared by full entries and preselection views."""

    @property
    def valid_from(self) -> datetime: ...

    @property
    def invalid_from(self) -> datetime | None: ...

    @property
    def expires(self) -> str: ...


def _parse_expires_date(raw: str) -> date | None:
    """Parse an ``expires`` field to a UTC date, or ``None`` if it never expires.

    Accepts the two shapes the field is written in — a bare ISO date
    (``"2026-07-01"``) and a full ISO datetime (``"2026-07-01T00:00:00+00:00"``).
    An empty or unparseable value returns ``None``, matching the existing
    fallback in ``trw_mcp.scoring._decay`` rather than inventing a second
    convention: a value nobody can read must not silently retire a record.
    """
    if not raw:
        return None
    try:
        # UTC-normalise BEFORE taking the calendar date: an offset-bearing value
        # near a day boundary otherwise yields the wrong day (PRD-CORE-278 FR05).
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.date()
    except ValueError:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None


def _is_expired_at(entry: ValidityFields, as_of: datetime | None, *, reference_time: datetime | None = None) -> bool:
    """PRD-CORE-244 FR05 — has *entry*'s author-set validity window closed?

    Boundary semantics are day-exclusive and chosen to match
    ``trw_mcp.scoring._decay`` exactly (``today > expires_date``): an entry whose
    ``expires`` date IS the evaluation date is still valid, and expiry takes
    effect at the start of the following UTC day.

    Under ``as_of`` time travel the predicate is evaluated against the ``as_of``
    instant, so a caller asking what was believed at time T gets a record that
    was unexpired at T (resolves OQ-04).
    """
    return expiry_has_passed(entry.expires, reference_time=as_of or reference_time)


def _parse_expires_instant(raw: str) -> datetime | None:
    """Parse an ``expires`` value that carries a TIME, normalised to UTC.

    Returns ``None`` for a bare date (the day-exclusive branch owns those) and
    for anything unparseable. A naive datetime is read as UTC, matching how the
    date branch already interprets naive references.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:  # trw-fail-silent-allow: None IS the answer — "this value is not an instant" is this function's contract, and the only caller then tries the date branch and finally treats an unreadable value as never-expiring, which is the documented fail-open for expiry
        return None
    # A bare date parses as midnight; it is a DATE, and the day-exclusive rule
    # applies to it. Only a value that actually spelled a time is an instant.
    if len(raw.strip()) <= 10:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def expiry_has_passed(expires: str, *, reference_time: datetime | None = None) -> bool:
    """The ONE expiry predicate for every recall gate (PRD-CORE-278 FR05).

    Two shapes, one rule each, because the field is written in both:

    - **A value carrying a time** (``2026-09-16T20:38:51Z``) expires AT that
      instant, compared in UTC. This is the shape a checkpoint or handoff uses,
      and reducing it to a calendar date is what let an entry that expired an
      hour ago keep its slot for the rest of the day (sub_4-nL1paSXxQx41fH).
    - **A bare date** (``2026-07-01``) keeps the established day-exclusive
      convention: valid through that whole UTC day, expired from the next.

    Missing or malformed expiry never silently retires a record. Callers supply
    the invocation's evaluation instant (historical ``as_of`` when present).
    """
    instant = reference_time or datetime.now(timezone.utc)
    # Naive references retain the existing calendar-date interpretation as UTC;
    # never let astimezone infer the machine's local timezone for them.
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    expires_at = _parse_expires_instant(expires)
    if expires_at is not None:
        return expires_at < instant
    expires_date = _parse_expires_date(expires)
    if expires_date is None:
        return False
    return instant.date() > expires_date


def _is_open_at(entry: ValidityFields, as_of: datetime | None, *, reference_time: datetime | None = None) -> bool:
    """Eligibility test for a single entry.

    ``as_of is None`` (default): a record is eligible iff its window is OPEN now
    (``invalid_from is None``). ``as_of=T``: eligible iff its window contained T
    — half-open ``valid_from <= T < invalid_from`` (treating ``invalid_from is
    None`` as ``+inf``).

    PRD-CORE-244 FR05: a past ``expires`` date also closes the window. Before
    this, ``trw_mcp.scoring._decay`` floored an expired entry's utility at 0.01
    while THIS path still called it an open record, so the two ranking paths
    disagreed about what an expired record means. Reusing ineligibility is what
    gives expired records the demotion superseded records already get — the
    default-exclude and the ``include_superseded`` append-after-open behaviour —
    with no new ranking code.
    """
    if _is_expired_at(entry, as_of, reference_time=reference_time):
        return False
    if as_of is None:
        return entry.invalid_from is None
    if entry.valid_from > as_of:
        return False
    return entry.invalid_from is None or as_of < entry.invalid_from


def apply_validity_prior(
    entries: list[MemoryEntry],
    *,
    as_of: datetime | None = None,
    reference_time: datetime | None = None,
    valid_from_min: datetime | None = None,
    include_superseded: bool = False,
    age_decay: bool = False,
    fusion_scores: Mapping[str, float] | None = None,
) -> list[MemoryEntry]:
    """Apply the validity prior to an already-fused, ordered candidate list.

    Args:
        entries: Fusion-ordered candidates (highest relevance first).
        as_of: When set, re-scope eligibility to records whose validity window
            contained this instant ("what was believed true as of T").
        valid_from_min: When set, only include entries whose ``valid_from`` is
            at or after this datetime.  Useful for narrowing results to a
            specific date range — e.g. when temporal arithmetic resolves
            "10 days ago" to a target date, pass
            ``valid_from_min = target - slack`` to exclude older sessions.
            Applied in addition to (AND with) *as_of* eligibility.
        include_superseded: When True, ineligible records are appended AFTER all
            eligible records (positional rank penalty, OQ2) rather than dropped.
        age_decay: When True, apply a tie-only age advantage so the newer
            ``valid_from`` floats above an older one when fusion ranked them
            equal — fusion order is otherwise preserved.
        fusion_scores: Optional fused score by entry id. Required for age decay
            to prove a true score tie; when absent, age_decay preserves input
            order rather than guessing and globally sorting by recency.

    Returns:
        The reordered/filtered list of entries.
    """
    eligible: list[MemoryEntry] = []
    ineligible: list[MemoryEntry] = []
    reference_time = reference_time or datetime.now(timezone.utc)
    for entry in entries:
        in_window = _is_open_at(entry, as_of, reference_time=reference_time)
        if in_window and valid_from_min is not None and entry.valid_from < valid_from_min:
            in_window = False
        if in_window:
            eligible.append(entry)
        else:
            ineligible.append(entry)

    if age_decay:
        eligible = _apply_age_decay(eligible, fusion_scores=fusion_scores)

    if include_superseded:
        # Positional penalty: superseded/out-of-window records always trail the
        # open ones, preserving their relative fused order among themselves.
        return [*eligible, *ineligible]
    return eligible


def _apply_age_decay(
    eligible: list[MemoryEntry],
    *,
    fusion_scores: Mapping[str, float] | None,
    tie_epsilon: float = 1e-12,
) -> list[MemoryEntry]:
    """Tie-aware age preference: newer ``valid_from`` floats up within a tie group.

    The decay term is intentionally minimal (OQ1 non-blocking) and BOUNDED: it
    may reorder records only inside consecutive fused-score tie buckets. An older
    record that earned a higher fused score stays above a newer lower-scored
    record, preserving the post-fusion relevance order except for true ties.
    """
    if not fusion_scores:
        return eligible

    ordered: list[MemoryEntry] = []
    bucket: list[MemoryEntry] = []
    bucket_score: float | None = None

    def flush_bucket() -> None:
        if not bucket:
            return
        # Stable sort: newer valid_from first. Equal valid_from keeps fused order.
        ordered.extend(sorted(bucket, key=lambda e: _neg_epoch(e.valid_from)))
        bucket.clear()

    for entry in eligible:
        score = fusion_scores.get(str(entry.id))
        if score is None:
            flush_bucket()
            ordered.append(entry)
            bucket_score = None
            continue
        if bucket_score is None or abs(score - bucket_score) <= tie_epsilon:
            bucket.append(entry)
            bucket_score = score if bucket_score is None else bucket_score
            continue
        flush_bucket()
        bucket.append(entry)
        bucket_score = score

    flush_bucket()
    return ordered


def _neg_epoch(when: datetime) -> float:
    """Negative POSIX timestamp so a NEWER instant sorts FIRST (ascending sort)."""
    return -when.timestamp()
