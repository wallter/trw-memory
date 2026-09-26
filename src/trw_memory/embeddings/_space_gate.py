"""Dense-scoring admission: only vectors from the active embedding space.

A cosine between a query vector and a stored vector is meaningful only when both
came from the same encoder. After an embedding-model change a store holds
vectors from the old model; scoring a new-model query against them produces
confident-looking noise. This module is the single gate every dense read path
uses before it scores anything:

- the active space is the provider's own descriptor (``embedding_space()``);
- a stored vector is admitted only when its recorded provenance names exactly
  that space;
- a vector from another space, or one with no provenance (written before
  provenance existed, or by an unidentifiable provider), is excluded from dense
  scoring. The row itself stays in the candidate pool, so BM25 still ranks it;
- exclusions are reported once per call as a structured warning naming the
  re-embed command, so a degraded store is visible instead of silently worse.

Fail closed: an active provider with no descriptor admits nothing, because
nothing can be proven compatible with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, provider_embedding_space

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "REEMBED_HINT",
    "SpaceSelection",
    "active_embedding_space",
    "admit_space_vectors",
    "comparable_neighbours",
    "recorded_spaces",
    "select_space_vectors",
    "vector_in_space",
]

logger = structlog.get_logger(__name__)

REEMBED_HINT = "run `trw-memory reembed --namespace <ns>` (or MemoryClient.reembed()) to re-encode them"


@dataclass(frozen=True)
class SpaceSelection:
    """Vectors admitted for dense scoring and what was held back."""

    vectors: dict[str, list[float]]
    mismatched: int
    unqualified: int

    @property
    def excluded(self) -> int:
        return self.mismatched + self.unqualified


def active_embedding_space(embedder: object | None) -> EmbeddingSpace | None:
    """Return the space *embedder* encodes into, loading it only if it must.

    A provider reports its space only once its model is loaded; ``available()``
    is the protocol's load trigger and is a cached no-op afterwards. Dense
    scoring would load the model to encode the query anyway.
    """
    if embedder is None:
        return None
    space = provider_embedding_space(embedder)
    if space is not None:
        return space
    available = getattr(embedder, "available", None)
    if callable(available) and available():
        return provider_embedding_space(embedder)
    return None


def recorded_spaces(
    backend: StorageBackend, embedded: Sequence[tuple[str, list[float]]], *, namespace: str
) -> dict[str, EmbeddingSpace | None]:
    """The space recorded for each ``(entry_id, embedding)``'s stored vector, if it IS that embedding."""
    records = backend.get_vector_records([entry_id for entry_id, _embedding in embedded], namespace=namespace)
    spaces: dict[str, EmbeddingSpace | None] = {}
    for entry_id, embedding in embedded:
        record = records.get(entry_id)
        proof = record.provenance if record is not None else None
        spaces[entry_id] = proof.space if proof is not None and proof.matches_vector(embedding) else None
    return spaces


def vector_in_space(record: StoredVector | None, space: EmbeddingSpace) -> bool:
    """True when *record* carries provenance naming exactly *space*."""
    return record is not None and record.provenance is not None and record.provenance.space == space


def select_space_vectors(records: Mapping[str, StoredVector], space: EmbeddingSpace | None) -> SpaceSelection:
    """Split *records* into vectors of *space* and the counts held back."""
    vectors: dict[str, list[float]] = {}
    mismatched = 0
    unqualified = 0
    for entry_id, record in records.items():
        proof = record.provenance
        if proof is None:
            unqualified += 1
        elif space is None or proof.space != space:
            mismatched += 1
        else:
            vectors[entry_id] = list(record.embedding)
    return SpaceSelection(vectors, mismatched, unqualified)


def admit_space_vectors(
    records: Mapping[str, StoredVector],
    space: EmbeddingSpace | None,
    *,
    namespace: str,
    surface: str,
) -> dict[str, list[float]]:
    """Return only *space*'s vectors, warning once when any were excluded."""
    selection = select_space_vectors(records, space)
    if selection.excluded:
        logger.warning(
            "dense_vectors_excluded_embedding_space",
            surface=surface,
            namespace=namespace,
            excluded=selection.excluded,
            mismatched_space=selection.mismatched,
            no_provenance=selection.unqualified,
            admitted=len(selection.vectors),
            active_space=space.encoding if space is not None else None,
            reembed_required=True,
            detail="excluded vectors are not dense-scored; BM25 still ranks their rows",
            hint=REEMBED_HINT,
        )
    return selection.vectors


def comparable_neighbours(
    backend: StorageBackend, vector: list[float], space: EmbeddingSpace, *, namespace: str, top_k: int, surface: str
) -> list[tuple[str, float]] | None:
    """*namespace*'s KNN window for *vector*, or ``None`` when a dense verdict over it would be incomplete.

    An excluded hit (another space, or no provenance) has a meaningless distance and
    may be a textual duplicate or hide in-space rows past the window; and a window
    all in *space* proves nothing about rows past it unless the store's census
    shows the WHOLE namespace in *space*. ``None`` tells the caller to decide
    exhaustively. An empty window is a complete (empty) verdict.
    """
    hits = backend.search_vectors(vector, top_k=top_k, namespace=namespace)
    if not hits:
        return []
    records = backend.get_vector_records([entry_id for entry_id, _ in hits], namespace=namespace)
    admitted = admit_space_vectors(records, space, namespace=namespace, surface=surface)
    census = backend.vector_space_census(namespace=namespace)
    # the census must cover every row _exhaustive would examine: a vectorless row may be the duplicate (C12 rc4)
    if len(admitted) < len(hits) or not _census_proves(census, space, rows=backend.count(namespace=namespace)):
        logger.debug("dense_window_incomplete", surface=surface, window=len(hits), admitted=len(admitted))
        return None
    return [(entry_id, distance) for entry_id, distance in hits if entry_id in admitted]


def _census_proves(census: object, space: EmbeddingSpace, *, rows: int) -> bool:
    """A census proves one space only if it is a mapping of positive int counts, all
    keyed by *space*, that accounts for at least the namespace's *rows* (not just the window). An
    empty census beside a nonempty window, or any invalid count, proves nothing."""
    if not isinstance(census, dict) or not census:
        return False
    counts = list(census.values())
    if not all(type(count) is int and count > 0 for count in counts):
        return False
    return all(key == space for key in census) and sum(counts) >= rows
