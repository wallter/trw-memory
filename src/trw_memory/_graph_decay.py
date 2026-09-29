"""Importance boost / decay helpers for the graph layer.

Belongs to the ``graph.py`` facade, which re-exports its public names.

3 helpers covering the importance-modulation pipeline:

- ``apply_importance_boost`` — round-up importance + record in
  outcome_history + flip cross_validated flag (default reason
  ``cross_validated``, default delta ``IMPORTANCE_BOOST=0.05``).
- ``apply_importance_decay`` — round-down importance with floor at
  0.0 + record in outcome_history.
- ``memory_decay_pass`` — batch decay sweep for entries unused for
  cutoff_days (default 90). Direct SQL for batch performance. Wired to
  production by PRD-CORE-244 FR09 as a deferred-delivery step. Advances a
  persisted keyset ``cursor`` over the table's ``(namespace, id)`` primary
  key so a repeated pass reaches every eligible row before any repeats
  (PRD-CORE-307 FR05): each pass examines the next *batch_size* rows after
  the cursor, in primary-key order, whether or not they qualify -- so one
  pass's cost is bounded by rows examined, not rows qualifying, and the
  primary key's own index (present on every store) makes that scan a seek,
  not a full table scan (NFR03), with no new index required.

Plus 2 module constants: ``IMPORTANCE_BOOST`` and ``DECAY_DELTA``.

Extracted as PRD-DIST-245 Phase 2 batch 94.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Collection
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Final

import structlog

from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

IMPORTANCE_BOOST = 0.05
DECAY_DELTA = 0.1


def apply_importance_boost(
    entry: MemoryEntry,
    reason: str = "cross_validated",
    delta: float = IMPORTANCE_BOOST,
) -> MemoryEntry:
    """Apply an importance boost to an entry, capped at 1.0.

    Records the boost in outcome_history.
    """
    new_importance = min(round(entry.importance + delta, 4), 1.0)
    now = datetime.now(timezone.utc).isoformat()
    outcome = f"importance_boost:delta=+{delta:.2f}:reason={reason}:new_value={new_importance:.4f}:timestamp={now}"

    return entry.model_copy(
        update={
            "importance": new_importance,
            "outcome_history": [*entry.outcome_history, outcome],
            "cross_validated": True,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def apply_importance_decay(
    entry: MemoryEntry,
    delta: float = DECAY_DELTA,
) -> MemoryEntry:
    """Apply importance decay for unused shared memories.

    Floors at 0.0. Records in outcome_history.
    """
    new_importance = max(round(entry.importance - delta, 4), 0.0)
    now = datetime.now(timezone.utc).isoformat()
    outcome = f"importance_decay:delta=-{delta:.2f}:reason=unused_90d:new_value={new_importance:.4f}:timestamp={now}"

    return entry.model_copy(
        update={
            "importance": new_importance,
            "outcome_history": [*entry.outcome_history, outcome],
            "updated_at": datetime.now(timezone.utc),
        }
    )


#: The one predicate this sweep selects on. Decay is a function of DISUSE.
#:
#: Until PRD-CORE-244 FR09 it also required ``cross_validated = 1``, which was
#: re-measured at 0 of 9,366 rows on 2026-09-03 — so the sweep, had anything
#: ever called it, would have decayed nothing while reporting success. That is a
#: default asserting a result, which is precisely what this PRD exists to remove.
#: Cross-project validation says an entry was CONFIRMED elsewhere; it says
#: nothing about whether anyone has used it since, and gating decay on it made
#: importance a one-directional ratchet.
_DECAY_PREDICATE: Final = "COALESCE(last_accessed_at, created_at) < ?"


def memory_decay_pass(
    conn: sqlite3.Connection,
    cutoff_days: int = 90,
    batch_size: int = 1000,
    *,
    lock: threading.Lock | None = None,
    namespaces: Collection[str] | None = None,
    cursor: tuple[str, str] | None = None,
) -> dict[str, object]:
    """Lower the importance of memories unused for *cutoff_days*.

    *namespaces* narrows the pass to those namespaces (a daemon token's grant,
    PRD-CORE-298 FR02); ``None`` decays the whole store. *cursor* resumes the
    keyset scan after a ``(namespace, id)`` pair a prior pass returned as
    ``"next"``; ``None`` starts (or wraps to) the beginning.

    The caller owns the connection and the lock. ``cutoff_days`` and
    ``batch_size`` are supplied by the production caller (``tools.maintain._run_decay``)
    from ``MemoryConfig.decay_cutoff_days`` / ``decay_batch_size``
    (PRD-CORE-331 FR10 B71-135h); the literal defaults here exist only so a
    direct library caller has a sane one.

    Returns ``{"processed": int, "total_decayed": int, "next": [namespace, id]
    | None, "more": bool}``. ``"next"`` is the last row this pass examined --
    not the last it decayed -- so a later pass resumes scanning forward even
    through a run of ineligible rows; ``None`` means this pass reached the end
    of the table (the next pass starts over). ``"more"`` is ``True`` exactly
    when ``"next"`` is not ``None`` -- a cheap continuation signal that costs
    nothing beyond the window this pass already read, unlike the O(namespace)
    ``COUNT(*)`` a "remaining" count required (PRD-CORE-331 FR10).
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    effective_batch_size = min(batch_size, 1000)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)).isoformat()
    scope = sorted(namespaces) if namespaces is not None else []
    scope_predicate = f" AND namespace IN ({', '.join('?' * len(scope)) or 'NULL'})" if namespaces is not None else ""
    seek_predicate = " AND (namespace, id) > (?, ?)" if cursor is not None else ""
    seek_params: tuple[str, ...] = cursor if cursor is not None else ()
    predicate = _DECAY_PREDICATE

    # Acquire the lock BEFORE both SELECT statements so concurrent backend
    # writes that hold the same lock cannot interleave on the shared connection
    # between the read and the subsequent updates. Without the lock the SELECTs
    # and UPDATEs could race on the single sqlite3.Connection object, which is
    # not thread-safe for concurrent use without external serialisation.
    decayed = 0
    batch_now = datetime.now(timezone.utc).isoformat()
    with lock or nullcontext():
        # A window of the NEXT effective_batch_size rows in primary-key order,
        # eligible or not (PRD-CORE-307 FR05): bounding by rows EXAMINED, not
        # rows qualifying, is what lets the cursor advance past a long run of
        # fresh rows instead of re-scanning them full-scan every pass forever.
        # The primary key's own index makes this a seek (NFR03) -- see EXPLAIN
        # QUERY PLAN coverage in tests/test_graph_decay.py.
        window = conn.execute(
            # S608 justified: predicate/scope_predicate/seek_predicate are built
            # from a module-level Final literal and a caller-controlled *count*
            # of placeholders only, never caller-controlled SQL text.
            f"SELECT namespace, id, importance, ({predicate}) AS eligible FROM memories "  # noqa: S608
            f"WHERE 1=1{seek_predicate}{scope_predicate} ORDER BY namespace, id LIMIT ?",
            (cutoff, *seek_params, *scope, effective_batch_size),
        ).fetchall()

        try:
            for namespace, entry_id, raw_importance, eligible in window:
                if not eligible:
                    continue
                new_value = max(round(float(raw_importance) - DECAY_DELTA, 4), 0.0)
                outcome = (
                    f"importance_decay:delta=-{DECAY_DELTA:.2f}:"
                    f"reason=unused_{cutoff_days}d:new_value={new_value:.4f}:timestamp={batch_now}"
                )
                conn.execute(
                    "UPDATE memories SET importance = ?, "
                    "outcome_history = json_insert(outcome_history, '$[#]', ?) "
                    "WHERE namespace = ? AND id = ?",
                    (new_value, outcome, namespace, entry_id),
                )
                decayed += 1
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("memory_decay_pass_failed")
            raise

    reached_end = len(window) < effective_batch_size
    next_cursor = None if reached_end or not window else (window[-1][0], window[-1][1])

    return {
        "processed": decayed,
        "total_decayed": decayed,
        "next": list(next_cursor) if next_cursor is not None else None,
        "more": next_cursor is not None,
    }
