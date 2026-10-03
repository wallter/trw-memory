"""The quarantine check every platform send makes immediately before it POSTs (PRD-CORE-333, CORE-333-PUBLISHER-BYPASS).

Belongs to the ``quarantine_ledger`` seam. The read layer filters what a backend returns, but a row can be paged,
queued or read from the mirror before it is quarantined and sent after; and some senders never read through a
backend at all. So each send asks the ledger again, by the row's full identity: its namespace-qualified id, its
source and remote ids, and its content hash. A ledger that cannot be read refuses the send.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from typing import Literal

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.quarantine_ledger import entry_identity_keys, ledger_for_config

#: ``blocked``: the ledger blocks the row now. ``unreadable``: the ledger could not be read (refuse; retry later).
Refusal = Literal["blocked", "unreadable"]


def identity_keys(entry: MemoryEntry) -> list[list[str]]:
    """*entry*'s ledger keys as JSON-safe lists, to carry with a queued send."""
    return [list(key) for key in entry_identity_keys(entry)]


def egress_refused(keys: Iterable[Sequence[str]], config: MemoryConfig | None = None) -> Refusal | None:
    """Why a row with these ledger *keys* must not leave the host now, or ``None`` when it may."""
    try:
        view = ledger_for_config(config or MemoryConfig()).view()
    except (OSError, sqlite3.Error):  # trw-fail-silent-allow: fail closed -- returned as "unreadable", never sent
        return "unreadable"
    return "blocked" if view and view.blocks_keys(tuple(key) for key in keys) else None


def entry_egress_refused(entry: MemoryEntry, config: MemoryConfig | None = None) -> Refusal | None:
    """:func:`egress_refused` for a stored *entry*."""
    return egress_refused(entry_identity_keys(entry), config)
