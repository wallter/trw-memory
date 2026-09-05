"""The one place a production writer builds a :class:`MemoryEntry`.

PRD-CORE-245 FR08. Three writers used to construct entries by hand and exactly
one of them set ``vector_clock``: ``MemoryClient.store`` called ``init_clock`` /
``increment_clock``, while ``trw_memory.tools.store`` (what
``trw-memory-server`` writes through) and ``trw_mcp.state._memory_transforms``
(the flagship consumer's real write path) left the Pydantic default ``{}``.
Measured on the reference store 2026-09-03: **55 of 9,366 rows (0.6%)** carried
a non-empty clock.

That field is not decorative. ``resolve_conflict`` reads it on every org-shared
pull, and the remote side of the comparison DOES get a clock (coerced from the
peer payload). ``compare_clocks`` then fails two ways, silently:

* both clocks empty -> ``all()`` over an empty key set is vacuously true both
  ways, so the result is ``"concurrent"`` and every pull merges regardless of
  causal order;
* local empty, peer populated -> ``"b_wins"``, and the resolver returns the
  REMOTE entry outright. **A local edit that strictly postdates the remote one
  is discarded, not merged, with no error.**

For 99.4% of live rows the second case is the one a real federation produces.
Routing every writer through here is what makes causality real: this is the one
place that sets ``vector_clock``, ``namespace`` and the bi-temporal validity
fields, so a future writer cannot omit one by building the model directly.
"""

from __future__ import annotations

import hashlib
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trw_memory.models.memory import MemoryEntry
from trw_memory.sync.conflict import increment_clock, init_clock

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["local_node_id_for", "new_entry", "revise_entry"]


def local_node_id_for(storage_path: str | Path) -> str:
    """Return this installation's vector-clock node id for *storage_path*.

    Mirrors ``MemoryClient``'s own derivation (hostname plus the resolved
    storage path, double-hashed) so a row written through the client and a row
    written through the tool surface name the SAME node. Two node ids for one
    installation would make its own successive edits look concurrent.
    """
    raw = f"{socket.gethostname()}:{Path(storage_path).resolve()}"
    first = hashlib.sha256(raw.encode()).hexdigest()
    return hashlib.sha256(first.encode()).hexdigest()[:16]


def new_entry(
    *,
    entry_id: str,
    content: str,
    namespace: str,
    local_node_id: str,
    now: datetime | None = None,
    fields: Mapping[str, object] | None = None,
) -> MemoryEntry:
    """Build a brand-new entry with its identity fields already set.

    ``fields`` carries whatever else the caller wants on the model; the four
    identity fields this exists to guarantee (``namespace``, ``vector_clock``,
    ``created_at``/``updated_at``) are applied AFTER it, so a caller cannot
    accidentally blank one.
    """
    stamp = now or datetime.now(timezone.utc)
    payload: dict[str, object] = dict(fields or {})
    payload.update(
        {
            "id": entry_id,
            "content": content,
            "namespace": namespace,
            "created_at": stamp,
            "updated_at": stamp,
            "vector_clock": init_clock(local_node_id),
        }
    )
    return MemoryEntry.model_validate(payload)


def revise_entry(
    existing: MemoryEntry,
    *,
    local_node_id: str,
    now: datetime | None = None,
    fields: Mapping[str, object] | None = None,
) -> MemoryEntry:
    """Return *existing* updated with *fields*, advancing its causal clock.

    Advancing rather than re-initialising is load-bearing: resetting the clock
    would make a newer local update look concurrent with the stale snapshot it
    supersedes, which is the shape ``compare_clocks`` cannot order.
    """
    stamp = now or datetime.now(timezone.utc)
    update: dict[str, object] = dict(fields or {})
    update.update(
        {
            "updated_at": stamp,
            "vector_clock": increment_clock(existing.vector_clock, local_node_id),
        }
    )
    return existing.model_copy(update=update)
