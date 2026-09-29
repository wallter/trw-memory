"""Embedding-space handling for the warm tier's own vector index.

The warm tier keeps its own vector index (``memory/warm.db``) beside the JSONL
sidecar. Its vectors are copies taken when an entry was stored or promoted, so
after an embedding-model change they are stale in exactly the way primary
vectors are. Two halves live here:

- :func:`admit_warm_vectors` / :func:`admit_warm_hits` -- the warm read-side
  gate: a vector (or KNN hit) is scored only when its stored provenance names
  the query's space (``_space_gate``);
- :func:`warm_page` / :func:`commit_warm_page` -- the warm half of the re-embed
  migration, one page at a time (``trw_memory._client_reembed`` drives it and
  encodes between the two, outside the process-wide tier lock).

Only vectors that already exist are re-encoded -- a sidecar row with no vector
was mirrored without one on purpose (recall mirrors carry no vector) and stays
keyword-only. Each page commits on its own, so an interrupted run resumes by
running again: rows already in the active space are skipped.
"""

from __future__ import annotations

import heapq
import sqlite3
import threading
from typing import TYPE_CHECKING, NamedTuple

from trw_memory.embeddings._space_gate import admit_space_vectors
from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, generation_provenance_kwargs
from trw_memory.exceptions import EmbeddingUnavailableError
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.storage._vector_ops import get_vector_records

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.lifecycle.tiers._warm import WarmTierStore
    from trw_memory.storage.interface import StorageBackend

__all__ = ["WarmSnapshot", "admit_warm_hits", "admit_warm_vectors", "commit_warm_page", "warm_page"]

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


class WarmSnapshot(NamedTuple):
    """One warm sidecar row as a re-embed page saw it: its text and its stored vector, if any."""

    entry_id: str
    text: str
    record: StoredVector | None


def warm_page(store: WarmTierStore, space: EmbeddingSpace, after: str | None, limit: int) -> list[WarmSnapshot]:
    """The next *limit* sidecar rows by id after *after*, each with its text and stored vector.

    The page's ids come off the cached parse before any payload is copied, and only its payloads are
    copied, since this runs under the process-wide tier lock once per re-embed page (B71-84). The id scan
    still reads the whole cached parse; bounding that needs an indexed lookup.
    """
    rows = store._warm_rows()
    ids = heapq.nsmallest(
        limit, {i for _, rec in rows if (i := str(rec.get("id", ""))) and (after is None or i > after)}
    )
    entries = store._entries_by_id(rows, set(ids))
    backend = store._get_warm_backend(dim=space.dimensions, create=False)
    vectored = backend is not None and backend.supports_vectors()
    records = backend.get_vector_records(ids, namespace=WARM_TIER_NAMESPACE) if backend and vectored else {}
    return [
        WarmSnapshot(entry_id, _entry_text(entries[entry_id]), records.get(entry_id))
        for entry_id in ids
        if entry_id in entries
    ]


def commit_warm_page(
    store: WarmTierStore,
    embedder: EmbeddingProvider,
    space: EmbeddingSpace,
    snapshot: list[WarmSnapshot],
    vectors: list[list[float] | None],
) -> int:
    """Write each re-encoded vector whose row still holds the text and vector *snapshot* saw; how many."""
    backend = store._get_warm_backend(dim=space.dimensions)
    if backend is None or not snapshot:
        return 0
    entries = store._entries_by_id(store._warm_rows(), {item.entry_id for item in snapshot})
    written = 0
    with backend.transaction():
        now = backend.get_vector_records([item.entry_id for item in snapshot], namespace=WARM_TIER_NAMESPACE)
        for item, vector in zip(snapshot, vectors, strict=True):
            payload = entries.get(item.entry_id)
            if vector is None or payload is None or _entry_text(payload) != item.text:
                continue
            if now.get(item.entry_id) != item.record:
                continue
            proof = generation_provenance_kwargs(embedder, item.text, vector)
            if not proof:
                raise EmbeddingUnavailableError("the embedder stopped reporting its embedding space mid-run")
            backend.upsert_vector(item.entry_id, vector, namespace=WARM_TIER_NAMESPACE, **proof)
            written += 1
    return written
