"""One resumable page of the forced knowledge-graph sweep over a namespace's existing rows.

Belongs to the ``graph.py`` facade. Re-exported there.

Why (F5 root-cause B): ``update_entry_graph`` runs for each row the store path
writes, so a corpus written before that wiring, or merged in from a checkout,
never gets its MATERIALISED edges -- similarity, consolidation, co-anchored and
cross-project validation. Tags need no sweep: the schema-5 migration fills
``memory_tags`` for every row. The caller owns the resume point (trw-mcp keeps
it beside the checkout, where an embedding migration can restart it); this
module only walks one page after it, on the store's own connection.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict
from typing import Any

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import EntryCursor

logger = structlog.get_logger(__name__)


def backfill_graph_page(
    backend: Any,
    namespace: str,
    *,
    after: EntryCursor | None,
    limit: int,
    deadline_seconds: float | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, Any]:
    """Enrich up to *limit* rows of *namespace* listed after *after*; report counts and where to resume.

    Each row reuses its stored vector, so nothing re-embeds. A row whose
    enrichment raises is counted as failed and passed, so one poison row cannot
    stall the sweep. ``complete`` is true only when the listing ran out and no
    deadline cut the page short; ``next`` is the cursor after the last row read.

    *deadline_seconds* is a soft budget that counts from before the listing is
    read: once spent, no further row starts, so at most one row's enrichment runs
    past it, and the next call resumes from ``next``.
    """
    from trw_memory import graph

    start, interrupted = time.monotonic(), False
    entries = backend.list_entries(namespace=namespace, limit=limit, after=after)
    counts = dict.fromkeys(("processed", "edges_built", "skipped", "failed"), 0)
    for entry in entries:
        if deadline_seconds is not None and time.monotonic() - start >= deadline_seconds:
            interrupted = True
            break
        if entry.metadata.get("system_canary") == "true":
            counts["skipped"] += 1
        else:
            try:
                vector = backend.get_stored_embeddings([entry.id], namespace=namespace).get(entry.id)
                built = graph.update_entry_graph(
                    entry, backend, embedding=list(vector) if vector is not None else None, config=config
                )
                counts["edges_built"] += sum(value for value in built.values() if isinstance(value, int))
                counts["processed"] += 1
            except (sqlite3.Error, ValueError, RuntimeError):
                counts["failed"] += 1
                logger.debug("graph_backfill_entry_failed", entry_id=entry.id, exc_info=True)
        after = EntryCursor.from_entry(entry)
    complete = not interrupted and len(entries) < limit
    logger.info("graph_backfill_page", namespace=namespace, complete=complete, **counts)
    return {**counts, "next": asdict(after) if after is not None else None, "complete": complete}
