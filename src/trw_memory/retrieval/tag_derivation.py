"""Bounded tag co-occurrence derivation over the ``memory_tags`` index.

PRD-CORE-245 FR07. The knowledge graph used to materialise ``tag_cooccurrence``
edges: 98,288 of the reference store's 102,428 edges (95.96%), written 500
candidates at a time and never recomputed. Measured over 50 random roots that
set held a mean 19.1 neighbours against the 573.3 the same predicate yields over
the full corpus — **3.3% of the relation it claimed to store**, biased toward
whichever entries happened to be recent at write time.

Schema 5 replaced it with ``memory_tags(namespace, tag, entry_id)`` and this
query. Two consequences the caller must know:

* **Single-root only.** A batched derivation across 25 roots measured 912 ms
  against 195 ms for the same work as a per-root loop and 1.0 ms for the old
  materialised lookup. It is therefore never called from the multi-root recall
  expansion path; ``graph_query`` walks materialised edges alone.
* **Explicit ordering.** Neighbours are ranked by shared-tag count descending,
  then by entry id ascending. The old ordering that happened to favour semantic
  edges was an artefact of ``idx_mge_source(source_id, edge_type)`` returning
  ``edge_type`` alphabetically; that accident is not a contract.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import NamedTuple

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.storage._sql_utils import iter_bind_chunks

logger = structlog.get_logger(__name__)

__all__ = ["DerivedTagNeighbour", "derive_tag_neighbours"]


class DerivedTagNeighbour(NamedTuple):
    """One derived tag-co-occurrence neighbour of a root entry.

    ``weight`` is the Jaccard coefficient over the two entries' tag sets, the
    same measure the deleted materialised edges carried, so a consumer that
    ranked on edge weight keeps ranking on the same quantity.
    """

    entry_id: str
    shared_tags: int
    weight: float


def derive_tag_neighbours(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    namespace: str,
    config: MemoryConfig,
) -> list[DerivedTagNeighbour]:
    """Return *root_id*'s tag neighbours within *namespace*, bounded by config.

    Every bound is a typed field on :class:`MemoryConfig`:
    ``graph_tag_min_shared_tags`` (how much overlap counts as a relation),
    ``graph_tag_max_tag_postings`` (above which a tag is a label, not a
    relation) and ``graph_tag_derive_top_k`` (result cap).

    Returns an empty list when the root has no tags, when every one of its tags
    is too common to be informative, or when the store predates the
    ``memory_tags`` index (the one SQLite failure this suppresses).

    Raises:
        sqlite3.Error: for every other storage failure. A locked database, a
            corrupt page, or a ``memory_tags`` whose columns have drifted is
            "derivation unavailable", which is NOT the same answer as "this
            root has no tag neighbours"; collapsing the two let a caller read a
            broken store as an empty relation.
    """
    min_shared = config.graph_tag_min_shared_tags
    max_postings = config.graph_tag_max_tag_postings
    top_k = config.graph_tag_derive_top_k
    try:
        # Informative tags of the root: its own postings, minus any tag whose
        # posting list is larger than the noise ceiling. Computed first so the
        # neighbour scan below never touches a high-fanout posting list.
        root_tags = [
            str(row[0])
            for row in conn.execute(
                "SELECT t.tag FROM memory_tags t WHERE t.namespace = ? AND t.entry_id = ? "
                "AND (SELECT COUNT(*) FROM memory_tags p WHERE p.namespace = t.namespace AND p.tag = t.tag) <= ?",
                (namespace, root_id, max_postings),
            ).fetchall()
        ]
        if len(root_tags) < min_shared:
            return []

        placeholders = ", ".join("?" for _ in root_tags)
        rows = conn.execute(
            f"SELECT entry_id, COUNT(*) AS shared FROM memory_tags "  # noqa: S608 — placeholders is ? repeated; tags are parameterised values
            f"WHERE namespace = ? AND tag IN ({placeholders}) AND entry_id != ? "
            "GROUP BY entry_id HAVING shared >= ? "
            "ORDER BY shared DESC, entry_id ASC LIMIT ?",
            (namespace, *root_tags, root_id, min_shared, top_k),
        ).fetchall()

        # Jaccard needs each neighbour's FULL tag count, including the tags the
        # noise ceiling excluded from the match — otherwise a broadly-tagged
        # entry scores as though it were narrowly tagged. One GROUP BY over the
        # root plus every matched neighbour, not a COUNT(*) per neighbour: the
        # per-neighbour form was N+1 round trips bounded only by
        # ``graph_tag_derive_top_k`` (up to 200), for a table this query has
        # already scanned.
        totals = _tag_counts(conn, namespace, [root_id, *(str(row[0]) for row in rows)])
        root_total = totals.get(root_id, 0)

        neighbours: list[DerivedTagNeighbour] = []
        for entry_id, shared in rows:
            other_total = totals.get(str(entry_id), 0)
            union = root_total + other_total - int(shared)
            weight = round(int(shared) / union, 4) if union > 0 else 0.0
            neighbours.append(DerivedTagNeighbour(str(entry_id), int(shared), min(weight, 1.0)))
    except sqlite3.OperationalError as exc:
        if not _is_missing_tag_index(exc):
            raise
        # A store opened before schema 5 has no memory_tags table. Degrade to
        # "no derived neighbours" rather than failing a traversal: the
        # materialised semantic edges are a separate query and still answer.
        logger.debug("tag_derivation_unavailable", reason="memory_tags_absent")
        return []
    return neighbours


def _is_missing_tag_index(exc: sqlite3.OperationalError) -> bool:
    """Return whether *exc* is the pre-schema-5 "no ``memory_tags`` table" case.

    Message inspection is the only signal SQLite offers: ``sqlite3`` raises the
    same ``OperationalError`` class for a missing table, a locked database and a
    drifted column list. Matching on both halves of the phrase keeps the
    suppression pinned to the one documented condition, so any other
    ``OperationalError`` reaches the caller instead of becoming an empty result.
    """
    message = str(exc).lower()
    return "no such table" in message and "memory_tags" in message


def _tag_counts(conn: sqlite3.Connection, namespace: str, entry_ids: Sequence[str]) -> dict[str, int]:
    """Return how many tags each of *entry_ids* carries in *namespace*.

    One statement for the whole set. Ids absent from ``memory_tags`` are absent
    from the result, so callers read a missing id as zero. ``entry_ids`` is the
    root plus at most ``graph_tag_derive_top_k`` (<= 200) neighbours, which is
    well inside the SQLite bind ceiling, but the chunking is explicit so a
    future cap raise cannot silently exceed it.
    """
    counts: dict[str, int] = {}
    for chunk in iter_bind_chunks(list(dict.fromkeys(entry_ids)), reserved_bindings=1):
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT entry_id, COUNT(*) FROM memory_tags "  # noqa: S608 — placeholders is ? repeated; ids are parameterised values
            f"WHERE namespace = ? AND entry_id IN ({placeholders}) GROUP BY entry_id",
            (namespace, *chunk),
        ).fetchall()
        counts.update({str(row[0]): int(row[1]) for row in rows})
    return counts
