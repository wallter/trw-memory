"""Edge-creation helpers for the graph layer.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

2 helpers covering the edge-creation pipeline. ``tag_cooccurrence`` used to be
a third: PRD-CORE-245 FR07 replaced those 98,288 materialised rows — which
captured only 3.3% of the relation they claimed to store — with the
``memory_tags`` inverted index and the bounded derivation in
:mod:`trw_memory.retrieval.tag_derivation`.

- ``create_similarity_edges`` — write similarity edges between an entry
  and its top candidates above ``SIMILARITY_THRESHOLD``, scored through
  ``CandidateVectors`` (normalised once per candidate set).
- ``create_consolidation_edges`` — consolidation lineage edges from
  ``entry.consolidated_from`` after verifying source exists.

Looks up ``_optional_lock`` / ``_upsert_edge`` via the parent ``graph``
module so test monkeypatches on those helpers still propagate.

Extracted as PRD-DIST-245 Phase 2 batch 95.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

import structlog

from trw_memory._graph_primitives import CandidateVectors, ScoredCandidates
from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

SIMILARITY_THRESHOLD = 0.75


def create_similarity_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    embedding: list[float] | None = None,
    candidate_embeddings: list[tuple[str, list[float]]] | None = None,
    *,
    lock: threading.Lock | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    candidates: ScoredCandidates | None = None,
) -> int:
    """Create similarity edges between entry and candidates above *threshold*.

    *threshold* is in the scale of the vectors compared; the caller calibrates
    the reference-scale ``SIMILARITY_THRESHOLD`` to their embedding space.
    *candidates* is *candidate_embeddings* already normalised; a caller
    enriching many entries against one candidate set builds it once and
    passes it here instead (it takes precedence); graph enrichment passes a
    view of the namespace's similarity index (``_graph_namespace_index``).
    Candidates are scored BEFORE *lock* is taken: an index view reads the
    database itself, and scoring needs no lock the writes hold.
    """
    if embedding is None or (candidate_embeddings is None and candidates is None):
        return 0

    from trw_memory import graph as g

    if candidates is None:
        candidates = CandidateVectors(candidate_embeddings or [])
    hits = candidates.above(embedding, threshold)
    created = 0
    now = datetime.now(timezone.utc).isoformat()

    with g._optional_lock(lock):
        for cand_id, sim in hits:
            if cand_id == entry.id:
                continue
            weight = round(sim, 4)
            created += g._upsert_edge(conn, entry.id, cand_id, "similarity", weight, now, namespace=entry.namespace)
            created += g._upsert_edge(conn, cand_id, entry.id, "similarity", weight, now, namespace=entry.namespace)
        conn.commit()
    logger.debug("similarity_edges_created", entry_id=entry.id, count=created)
    return created


def create_consolidation_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    *,
    lock: threading.Lock | None = None,
) -> int:
    """Consolidation lineage edges from ``entry.consolidated_from``."""
    if not entry.consolidated_from:
        return 0

    from trw_memory import graph as g

    created = 0
    now = datetime.now(timezone.utc).isoformat()

    with g._optional_lock(lock):
        for source_id in entry.consolidated_from:
            row = conn.execute(
                "SELECT id FROM memories WHERE namespace = ? AND id = ?", (entry.namespace, source_id)
            ).fetchone()
            if row is None:
                logger.debug("consolidation_edge_skip_missing", source_id=source_id)
                continue
            created += g._upsert_edge(conn, entry.id, source_id, "consolidation", 1.0, now, namespace=entry.namespace)
        conn.commit()
    logger.debug("consolidation_edges_created", entry_id=entry.id, count=created)
    return created
