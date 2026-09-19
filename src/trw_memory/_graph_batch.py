"""Graph enrichment for a batch of freshly written entries, in one pass.

Belongs to the ``graph.py`` facade (``update_entry_graph`` is the one-entry
form). Re-exported there.

Why a batch: enrichment reads the namespace's candidate set (every active
row's vector, through the per-process ``_graph_namespace_index``; the
``CANDIDATE_LIMIT`` most recent where that index is unavailable), resolves each entry's
embedding space, and cross-validates against every sibling project store. Done
per entry -- one background thread, one backend, and one full sibling-store
walk for every row of a ``bulk_store`` -- ingest cost grew with the number of
projects in the store: LOCOMO conversations ingested into one store took 60 s,
101 s, 350 s, 412 s, 598 s for the 1st..5th conversation (2026-09-18). Here the
candidate rows, their vectors, and the sibling stores are read once per batch
and normalised once (``CandidateVectors``); each entry then costs one
vectorised scoring pass per candidate set.

The graph-module helpers (``create_similarity_edges`` & co.) are looked up on
the ``graph`` module so test monkeypatches there still apply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import structlog

from trw_memory._graph_namespace_index import namespace_candidates
from trw_memory._graph_primitives import CandidateVectors, ScoredCandidates
from trw_memory.embeddings._similarity_calibration import calibrated_threshold
from trw_memory.embeddings._space_gate import recorded_spaces, select_space_vectors
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

#: One freshly written entry and the embedding stored with it (``None`` when unembedded).
GraphItem = tuple[MemoryEntry, list[float] | None]

_COUNT_KEYS = ("similarity_edges", "consolidation_edges", "co_anchored_edges", "cross_validated_projects")


def update_entries_graph(
    items: Sequence[GraphItem],
    backend: StorageBackend,
    *,
    config: MemoryConfig | None = None,
) -> dict[str, int]:
    """Best-effort graph enrichment for *items*; returns summed edge/validation counts.

    The graph is a secondary index over the canonical memory rows. A backend
    without a SQLite connection skips enrichment without touching the rows.
    Items may span namespaces; each namespace is enriched against its own
    candidates only.
    """
    totals = dict.fromkeys(_COUNT_KEYS, 0)
    raw_conn = getattr(backend, "_conn", None)
    if not callable(getattr(raw_conn, "execute", None)) or not callable(getattr(raw_conn, "commit", None)):
        logger.debug("graph_update_skipped", count=len(items), reason="no_sqlite_connection")
        return totals
    # Optional DB-API drivers are structurally compatible but have no shared
    # nominal Connection base class; the capability check above guards this.
    conn: Any = raw_conn
    by_namespace: dict[str, list[GraphItem]] = {}
    for item in items:
        by_namespace.setdefault(item[0].namespace, []).append(item)
    for namespace, group in by_namespace.items():
        for key, value in _update_namespace(namespace, group, backend, conn, config).items():
            totals[key] += value
    return totals


def _update_namespace(
    namespace: str,
    group: Sequence[GraphItem],
    backend: StorageBackend,
    conn: Any,
    config: MemoryConfig | None,
) -> dict[str, int]:
    from trw_memory import graph as g

    embedded = [(entry, embedding) for entry, embedding in group if embedding is not None]
    spaces = (
        recorded_spaces(backend, [(entry.id, embedding) for entry, embedding in embedded], namespace=namespace)
        if embedded
        else {}
    )
    # Similarity is only meaningful within one embedding space: each entry is
    # compared with the candidates whose vectors share the space of its own --
    # every ACTIVE row of the namespace through the per-process index, or the
    # ``CANDIDATE_LIMIT`` most recent ones where the index is unavailable.
    candidates = _candidates(namespace, embedded, spaces, backend) if embedded else None
    lock = getattr(backend, "_lock", None)
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    for entry, embedding in group:
        if embedding is not None and candidates is not None:
            space = spaces.get(entry.id)
            counts["similarity_edges"] += g.create_similarity_edges(
                entry,
                conn,
                embedding=embedding,
                lock=lock,
                threshold=calibrated_threshold(g.SIMILARITY_THRESHOLD, space),
                candidates=candidates(space),
            )
        counts["consolidation_edges"] += g.create_consolidation_edges(entry, conn, lock=lock)
        counts["co_anchored_edges"] += g.create_co_anchored_edges(
            conn,
            entry.id,
            list(dict.fromkeys(anchor.file for anchor in entry.anchors)),
            namespace=entry.namespace,
            lock=lock,
            min_shared_anchors=3,
        )
    validated = g.cross_validate_entries(
        [(entry, embedding, spaces.get(entry.id)) for entry, embedding in embedded], backend, config=config
    )
    counts["cross_validated_projects"] = sum(validated.values())
    return counts


def _candidates(
    namespace: str,
    embedded: Sequence[tuple[MemoryEntry, list[float]]],
    spaces: dict[str, EmbeddingSpace | None],
    backend: StorageBackend,
) -> Callable[[EmbeddingSpace | None], ScoredCandidates] | None:
    """Candidate sets per space for *namespace*: the whole-namespace index, else the recent window."""
    from trw_memory import graph as g

    index = namespace_candidates(backend, namespace)
    if index is not None:
        for entry, embedding in embedded:
            index.add(entry.id, embedding, spaces.get(entry.id))
        return index.in_space
    records = backend.recent_vector_records(namespace=namespace, limit=g.CANDIDATE_LIMIT)
    if not records:
        return None
    by_space: dict[EmbeddingSpace | None, CandidateVectors] = {}

    def window(space: EmbeddingSpace | None) -> CandidateVectors:
        if space not in by_space:
            by_space[space] = CandidateVectors(select_space_vectors(records, space).vectors.items())
        return by_space[space]

    return window
