"""Graph primitives — edge upsert + cosine-similarity helper.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

Two primitives shared by every helper cluster:

- ``_safe_cosine_similarity`` — :mod:`retrieval.dense.cosine_similarity`
  wrapper that returns 0.0 on dimension mismatch instead of raising.
- ``_upsert_edge`` — INSERT/UPDATE edge row in ``memory_graph_edges``
  with edge-type validation and 4096-byte metadata cap.
- ``CandidateVectors`` — a candidate set normalised ONCE and scored against
  many query vectors (graph enrichment compares every written entry with up to
  ``CANDIDATE_LIMIT`` vectors per namespace; one pure-Python cosine per pair
  was ~0.35 ms, so a 10-namespace store spent ~1.7 s per write on it).

Validates ``edge_type`` against ``VALID_EDGE_TYPES`` imported lazily
from the parent ``graph`` module so test patches on
``trw_memory.graph.VALID_EDGE_TYPES`` propagate.

Extracted as PRD-DIST-245 Phase 2 batch 98.
"""

from __future__ import annotations

import importlib
import json
import math
import operator
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import structlog

from trw_memory.exceptions import DimensionMismatchError
from trw_memory.retrieval.dense import cosine_similarity

logger = structlog.get_logger(__name__)


def _safe_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity with graceful degradation for graph operations.

    Delegates to ``retrieval.dense.cosine_similarity`` but returns 0.0
    on dimension mismatch.  Other ``ValueError`` subclasses are re-raised
    so callers can distinguish true zero-similarity from incompatible vectors.
    """
    try:
        return cosine_similarity(a, b)
    except DimensionMismatchError:
        logger.debug(
            "cosine_dimension_mismatch",
            len_a=len(a),
            len_b=len(b),
        )
        return 0.0


def _upsert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    created_at: str,
    *,
    namespace: str,
    metadata: dict[str, str] | None = None,
) -> bool:
    """Insert or update an edge in the graph; ``True`` when a row was written, ``False`` when a canary refused it.

    Args:
        namespace: The namespace both endpoints belong to. Schema 5 keys edge
            uniqueness on ``(namespace, source_id, target_id, edge_type)``
            (PRD-CORE-245 FR02), so an edge cannot be written without saying
            which namespace it lives in.
        metadata: Optional key-value metadata stored as JSON alongside the edge.

    Raises:
        ValueError: If *edge_type* is not in ``trw_memory.graph.VALID_EDGE_TYPES``.
    """
    from trw_memory import graph as _graph

    if edge_type not in _graph.VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type {edge_type!r}. Must be one of: {', '.join(sorted(_graph.VALID_EDGE_TYPES))}"
        )
    meta_json = json.dumps(metadata) if metadata else "{}"
    if len(meta_json) > 4096:
        raise ValueError(f"edge metadata exceeds 4096 byte limit ({len(meta_json)} bytes)")
    # PRD-CORE-245 FR02: the uniqueness constraint is namespace-qualified under
    # schema 5, so the ON CONFLICT target must name the same columns or SQLite
    # rejects the statement outright. Every edge is written here, so this is the
    # one place a system canary is kept out of the graph: no edge may touch one,
    # whichever enrichment path proposed it, or the decoy would surface as a neighbour.
    cursor = conn.execute(
        "INSERT INTO memory_graph_edges "
        "(namespace, source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "SELECT ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS ("
        "  SELECT 1 FROM memories m WHERE m.namespace = ? AND m.id IN (?, ?)"
        "  AND json_valid(m.metadata) AND json_extract(m.metadata, '$.system_canary') = 'true'"
        ") ON CONFLICT (namespace, source_id, target_id, edge_type) "
        "DO UPDATE SET weight = ?, edge_metadata = ?",
        (
            namespace,
            source_id,
            target_id,
            edge_type,
            weight,
            created_at,
            meta_json,
            namespace,
            source_id,
            target_id,
            weight,
            meta_json,
        ),
    )
    return cursor.rowcount > 0


def _numpy() -> Any | None:
    """numpy when installed (it ships with the embeddings extra), else ``None``."""
    try:
        return importlib.import_module("numpy")
    except (
        ImportError
    ):  # trw-fail-silent-allow: numpy is an optional accelerator; CandidateVectors has an exact pure-Python path
        return None


class ScoredCandidates(Protocol):
    """A candidate set graph enrichment scores query vectors against (``CandidateVectors`` or an index view)."""

    def above(self, query: Sequence[float], threshold: float) -> list[tuple[str, float]]: ...


class CandidateVectors:
    """Unit-normalised candidate vectors, scored against query vectors in one pass.

    Scores equal :func:`_safe_cosine_similarity` for every candidate: zero
    vectors score 0.0 and a candidate of a different dimension than the query
    is never returned (the helper scores it 0.0, which no positive threshold
    admits). Candidates are normalised once here instead of once per pair.

    ``compact=True`` holds the matrix as float32 (half the memory) for sets
    that are CACHED across writes; stored vectors are float32 already, so only
    the dot product's accumulation loses precision (~1e-7).
    """

    def __init__(self, vectors: Iterable[tuple[str, Sequence[float]]], *, compact: bool = False) -> None:
        by_dim: dict[int, tuple[list[str], list[Sequence[float]]]] = {}
        for candidate_id, vector in vectors:
            ids, rows = by_dim.setdefault(len(vector), ([], []))
            ids.append(candidate_id)
            rows.append(vector)
        self._np = _numpy()
        self._dtype = None if self._np is None else (self._np.float32 if compact else self._np.float64)
        self._groups: dict[int, tuple[list[str], Any]] = {
            dim: self._normalised(ids, rows) for dim, (ids, rows) in by_dim.items()
        }

    def _normalised(self, ids: list[str], rows: list[Sequence[float]]) -> tuple[list[str], Any]:
        """Unit rows (zero vectors dropped) as a matrix, or as lists without numpy."""
        np = self._np
        if np is not None:
            matrix = np.asarray(rows, dtype=np.float64)
            norms = np.sqrt((matrix * matrix).sum(axis=1))
            keep = norms > 0.0
            kept = [candidate_id for candidate_id, ok in zip(ids, keep.tolist(), strict=True) if ok]
            return kept, (matrix[keep] / norms[keep, None]).astype(self._dtype, copy=False)
        kept, units = [], []
        for candidate_id, row in zip(ids, rows, strict=True):
            norm = math.sqrt(sum(x * x for x in row))
            if norm != 0.0:
                kept.append(candidate_id)
                units.append([x / norm for x in row])
        return kept, units

    def __len__(self) -> int:
        return sum(len(ids) for ids, _rows in self._groups.values())

    @property
    def nbytes(self) -> int:
        """Approximate memory held by the normalised vectors (a pure-Python float is ~32 bytes)."""
        if self._np is not None:
            return sum(int(rows.nbytes) for _ids, rows in self._groups.values())
        return sum(len(ids) * dim * 32 for dim, (ids, _rows) in self._groups.items())

    def above(self, query: Sequence[float], threshold: float) -> list[tuple[str, float]]:
        """``(candidate_id, cosine)`` for every candidate scoring above *threshold*, in insertion order."""
        group = self._groups.get(len(query))
        norm = math.sqrt(sum(x * x for x in query))
        if group is None or norm == 0.0:
            return []
        ids, rows = group
        unit = [x / norm for x in query]
        if self._np is not None:
            scores: list[float] = (rows @ self._np.asarray(unit, dtype=self._dtype)).tolist()
        else:
            scores = [sum(map(operator.mul, row, unit)) for row in rows]
        return [(candidate_id, score) for candidate_id, score in zip(ids, scores, strict=True) if score > threshold]
