"""Importance boost / decay helpers for the graph layer.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

3 helpers covering the importance-modulation pipeline:

- ``apply_importance_boost`` — round-up importance + record in
  outcome_history + flip cross_validated flag (default reason
  ``cross_validated``, default delta ``IMPORTANCE_BOOST=0.05``).
- ``apply_importance_decay`` — round-down importance with floor at
  0.0 + record in outcome_history.
- ``memory_decay_pass`` — batch decay sweep for entries unused for
  cutoff_days (default 90). Direct SQL for batch performance. Wired to
  production by PRD-CORE-244 FR09 as a deferred-delivery step.

Plus 2 module constants: ``IMPORTANCE_BOOST`` and ``DECAY_DELTA``.

Extracted as PRD-DIST-245 Phase 2 batch 94.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Final

import structlog

from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

IMPORTANCE_BOOST = 0.05
DECAY_DELTA = 0.1


def _optional_lock_safe(lock: threading.Lock | None) -> contextlib.AbstractContextManager[bool]:
    """Look up _optional_lock via the parent graph module."""
    from trw_memory import graph as _graph_module

    return _graph_module._optional_lock(lock)


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
) -> dict[str, int]:
    """Lower the importance of memories unused for *cutoff_days*.

    The caller owns the connection and the lock. ``cutoff_days`` and
    ``batch_size`` are supplied by the production caller from typed config
    (``TRWConfig.memory_decay_cutoff_days`` / ``memory_decay_batch_size``); the
    literal defaults here exist only so a direct library caller has a sane one.

    Returns ``{"processed": int, "remaining": int, "total_decayed": int}``.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    effective_batch_size = min(batch_size, 1000)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)).isoformat()

    # Acquire the lock BEFORE both SELECT statements so concurrent backend
    # writes that hold the same lock cannot interleave on the shared connection
    # between the read and the subsequent updates. Without the lock the SELECTs
    # and UPDATEs could race on the single sqlite3.Connection object, which is
    # not thread-safe for concurrent use without external serialisation.
    decayed = 0
    batch_now = datetime.now(timezone.utc).isoformat()
    with _optional_lock_safe(lock):
        rows = conn.execute(
            # S608 justified: _DECAY_PREDICATE is a module-level Final literal,
            # never caller input. It is interpolated rather than duplicated
            # because the SELECT and the COUNT must select the SAME set — a
            # drift between them would report a "remaining" computed over
            # different rows than "processed", the class of silent miscount FR09
            # exists to remove.
            f"SELECT id, importance FROM memories WHERE {_DECAY_PREDICATE} LIMIT ?",  # noqa: S608
            (cutoff, effective_batch_size),
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {_DECAY_PREDICATE}",  # noqa: S608
            (cutoff,),
        ).fetchone()
        total_qualifying = total[0] if total else 0
        try:
            for entry_id, raw_importance in rows:
                new_value = max(round(float(raw_importance) - DECAY_DELTA, 4), 0.0)
                outcome = (
                    f"importance_decay:delta=-{DECAY_DELTA:.2f}:"
                    f"reason=unused_{cutoff_days}d:new_value={new_value:.4f}:timestamp={batch_now}"
                )
                conn.execute(
                    "UPDATE memories SET importance = ?, "
                    "outcome_history = json_insert(outcome_history, '$[#]', ?) "
                    "WHERE id = ?",
                    (new_value, outcome, entry_id),
                )
                decayed += 1
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("memory_decay_pass_failed")
            raise

    return {
        "processed": decayed,
        "remaining": max(total_qualifying - decayed, 0),
        "total_decayed": decayed,
    }
