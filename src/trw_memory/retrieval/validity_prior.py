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
from datetime import datetime

from trw_memory.models.memory import MemoryEntry


def _is_open_at(entry: MemoryEntry, as_of: datetime | None) -> bool:
    """Eligibility test for a single entry.

    ``as_of is None`` (default): a record is eligible iff its window is OPEN now
    (``invalid_from is None``). ``as_of=T``: eligible iff its window contained T
    — half-open ``valid_from <= T < invalid_from`` (treating ``invalid_from is
    None`` as ``+inf``).
    """
    if as_of is None:
        return entry.invalid_from is None
    if entry.valid_from > as_of:
        return False
    return entry.invalid_from is None or as_of < entry.invalid_from


def apply_validity_prior(
    entries: list[MemoryEntry],
    *,
    as_of: datetime | None = None,
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
    for entry in entries:
        in_window = _is_open_at(entry, as_of)
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
