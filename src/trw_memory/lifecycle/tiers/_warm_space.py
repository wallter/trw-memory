"""Embedding-space handling for the warm tier's own vector index.

The warm tier keeps its own vector index (``memory/warm.db``) beside the JSONL
sidecar. Its vectors are copies taken when an entry was stored or promoted, so
after an embedding-model change they are stale in exactly the way primary
vectors are. Two halves live here:

- :func:`admit_warm_vectors` / :func:`admit_warm_hits` -- the warm read-side
  gate: a vector (or KNN hit) is scored only when its stored provenance names
  the query's space (``_space_gate``);
- :func:`reembed_warm_vectors` -- the warm half of the re-embed migration; the
  primary half lives in ``trw_memory._client_reembed``.

Only vectors that already exist are re-encoded -- a sidecar row with no vector
was mirrored without one on purpose (recall mirrors carry no vector) and stays
keyword-only. Each batch commits on its own, so an interrupted run resumes by
running again: rows already in the active space are skipped.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

from trw_memory.embeddings._space_gate import admit_space_vectors, vector_in_space
from trw_memory.embeddings.provenance import EmbeddingSpace, generation_provenance_kwargs
from trw_memory.exceptions import EmbeddingUnavailableError
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.storage._vector_ops import get_vector_records

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.lifecycle.tiers._warm import WarmTierStore
    from trw_memory.storage.interface import StorageBackend

__all__ = ["admit_warm_hits", "admit_warm_vectors", "reembed_warm_vectors"]

#: Mirrors ``_warm.WARM_TIER_NAMESPACE`` (imported there from the same place);
#: not imported from ``_warm`` because ``_warm`` imports this module.
WARM_TIER_NAMESPACE = DEFAULT_NAMESPACE


def admit_warm_vectors(
    conn: sqlite3.Connection, entry_ids: list[str], space: EmbeddingSpace | None
) -> dict[str, list[float]]:
    """Decode the warm vectors for *entry_ids* that are recorded in *space*."""
    records = get_vector_records(
        conn, threading.Lock(), vec_available=True, entry_ids=entry_ids, namespace=WARM_TIER_NAMESPACE
    )
    return admit_space_vectors(records, space, namespace=WARM_TIER_NAMESPACE, surface="warm_discovery")


def admit_warm_hits(
    backend: StorageBackend, hits: list[tuple[str, float]], space: EmbeddingSpace | None
) -> list[tuple[str, float]]:
    """Keep only KNN hits whose stored vector is in *space*, in rank order."""
    if not hits:
        return hits
    records = backend.get_vector_records([entry_id for entry_id, _ in hits], namespace=WARM_TIER_NAMESPACE)
    admitted = admit_space_vectors(records, space, namespace=WARM_TIER_NAMESPACE, surface="warm_search")
    return [(entry_id, distance) for entry_id, distance in hits if entry_id in admitted]


def _entry_text(payload: dict[str, object]) -> str:
    """The document text the store paths encode: ``f"{content} {detail}"``."""
    return f"{payload.get('content', '')} {payload.get('detail', '')}"


def reembed_warm_vectors(
    store: WarmTierStore,
    embedder: EmbeddingProvider,
    space: EmbeddingSpace,
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Re-encode stale warm vectors; return ``(examined, reembedded)``."""
    entries = store._warm_sidecar_entries_by_id()
    if not entries:
        return 0, 0
    backend = store._get_warm_backend(dim=space.dimensions)
    if backend is None or not backend.supports_vectors():
        return 0, 0
    examined = 0
    reembedded = 0
    ids = list(entries)
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        records = backend.get_vector_records(chunk, namespace=WARM_TIER_NAMESPACE)
        examined += len(records)
        stale = [
            entry_id for entry_id in chunk if entry_id in records and not vector_in_space(records[entry_id], space)
        ]
        texts = [_entry_text(entries[entry_id]) for entry_id in stale]
        vectors = embedder.embed_batch(texts) if texts else []
        with backend.transaction():
            for entry_id, text, vector in zip(stale, texts, vectors, strict=True):
                if vector is None:
                    continue
                proof = generation_provenance_kwargs(embedder, text, vector)
                if not proof:
                    raise EmbeddingUnavailableError("the embedder stopped reporting its embedding space mid-run")
                backend.upsert_vector(entry_id, vector, namespace=WARM_TIER_NAMESPACE, **proof)
                reembedded += 1
    return examined, reembedded
