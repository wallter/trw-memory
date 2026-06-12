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
    include_superseded: bool = False,
    age_decay: bool = False,
) -> list[MemoryEntry]:
    """Apply the validity prior to an already-fused, ordered candidate list.

    Args:
        entries: Fusion-ordered candidates (highest relevance first).
        as_of: When set, re-scope eligibility to records whose validity window
            contained this instant ("what was believed true as of T").
        include_superseded: When True, ineligible records are appended AFTER all
            eligible records (positional rank penalty, OQ2) rather than dropped.
        age_decay: When True, apply a tie-only age advantage so the newer
            ``valid_from`` floats above an older one when fusion ranked them
            equal — fusion order is otherwise preserved.

    Returns:
        The reordered/filtered list of entries.
    """
    eligible: list[MemoryEntry] = []
    ineligible: list[MemoryEntry] = []
    for entry in entries:
        if _is_open_at(entry, as_of):
            eligible.append(entry)
        else:
            ineligible.append(entry)

    if age_decay:
        eligible = _apply_age_decay(eligible)

    if include_superseded:
        # Positional penalty: superseded/out-of-window records always trail the
        # open ones, preserving their relative fused order among themselves.
        return [*eligible, *ineligible]
    return eligible


def _apply_age_decay(eligible: list[MemoryEntry]) -> list[MemoryEntry]:
    """Tie-aware age preference: newer ``valid_from`` floats up within a tie group.

    The decay term is intentionally minimal (OQ1 non-blocking) and BOUNDED: it is
    a STABLE sort that prefers the newer ``valid_from``. Because the caller hands
    this helper a candidate list it considers relevance-tied for age purposes
    (``hybrid_search`` passes its fused candidates so the term layers onto the
    ``importance_alpha`` blend, never replacing fusion), the newer record always
    receives a NON-NEGATIVE advantage and is never ranked below an older one.
    Records that compare equal on ``valid_from`` keep their original fused order
    (stable sort), so the adjustment is monotone and order-preserving on ties.
    """
    # Stable sort: newer valid_from first. Equal valid_from keeps fused order.
    return sorted(eligible, key=lambda e: _neg_epoch(e.valid_from))


def _neg_epoch(when: datetime) -> float:
    """Negative POSIX timestamp so a NEWER instant sorts FIRST (ascending sort)."""
    return -when.timestamp()
