"""Re-embed migration: move a namespace's vectors into the active embedding space.

Dense recall scores only vectors whose recorded space equals the active
embedder's (``embeddings/_space_gate.py``). After an embedding-model change,
or for vectors written before provenance existed, rows fall back to BM25 until
their vectors are re-encoded. This module is that re-encode.

Contract:

- **Scope**: one namespace (the client's), every row in it regardless of status,
  so the store ends in a single space and ``as_of``/superseded recall is covered
  too. Rows with no vector at all are encoded as well.
- **Idempotent**: a row whose vector already carries the active space is left
  untouched; a second run reports ``reembedded == 0``.
- **Resumable, bounded memory**: keyset pages of ``batch_size`` rows; each page
  commits on its own, so an interrupted run loses at most one page of work and
  picks up where it stopped when run again.
- **Offline-safe**: the embedder is the same cached provider recall uses, so
  ``TRW_OFFLINE`` / ``HF_HUB_OFFLINE`` / ``local_only`` apply unchanged and an
  uncached model raises ``LocalOnlyViolationError`` rather than downloading.
- **Fail closed**: no embedder, or an embedder that cannot name its space,
  raises -- re-encoding into an unidentifiable space would only mint more
  vectors dense recall must refuse.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from typing_extensions import TypedDict

from trw_memory.embeddings._space_gate import active_embedding_space, vector_in_space
from trw_memory.embeddings.provenance import generation_provenance_kwargs
from trw_memory.exceptions import EmbeddingUnavailableError, StorageError
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import EntryCursor

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.embeddings.provenance import EmbeddingSpace

__all__ = ["DEFAULT_REEMBED_BATCH_SIZE", "ReembedResultDict", "reembed_namespace"]

logger = structlog.get_logger(__name__)

DEFAULT_REEMBED_BATCH_SIZE = 64


class ReembedResultDict(TypedDict):
    """Counts from one :meth:`MemoryClient.reembed` run."""

    namespace: str
    embedding_model: str
    embedding_space: str
    examined: int
    reembedded: int
    already_current: int
    skipped: int
    warm_examined: int
    warm_reembedded: int


async def reembed_namespace(client: MemoryClient, *, batch_size: int = DEFAULT_REEMBED_BATCH_SIZE) -> ReembedResultDict:
    """Re-encode every vector in the client's namespace not in the active space."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    client._require_permission(Permission.WRITE, "reembed")
    namespace = client._namespace
    embedder = client._get_embedder()
    if embedder is None:
        raise EmbeddingUnavailableError(
            f"no embedding model is available for {client._config.embedding_model!r}; "
            "install the extra with `pip install trw-memory[embeddings]`"
        )
    space = await asyncio.to_thread(active_embedding_space, embedder)
    if space is None:
        raise EmbeddingUnavailableError("the embedder does not report an embedding space; refusing to re-embed")
    backend = client._get_backend()
    if not backend.supports_vectors():
        raise StorageError("this backend stores no vectors (sqlite-vec unavailable); nothing can be re-embedded")

    examined = reembedded = already_current = skipped = 0
    cursor: EntryCursor | None = None
    while True:
        async with client._lock:
            batch = backend.list_entries(namespace=namespace, limit=batch_size, after=cursor)
            if not batch:
                break
            cursor = EntryCursor.from_entry(batch[-1])
            records = backend.get_vector_records([entry.id for entry in batch], namespace=namespace)
        # SEC-001 canaries are never recalled, so they need no vector.
        rows = [entry for entry in batch if entry.metadata.get("system_canary") != "true"]
        examined += len(rows)
        stale = [entry for entry in rows if not vector_in_space(records.get(entry.id), space)]
        already_current += len(rows) - len(stale)
        texts = [f"{entry.content} {entry.detail}" for entry in stale]
        vectors = await asyncio.to_thread(embedder.embed_batch, texts) if texts else []
        async with client._lock:
            with backend.transaction():
                for entry, text, vector in zip(stale, texts, vectors, strict=True):
                    if vector is None:
                        skipped += 1
                        continue
                    proof = generation_provenance_kwargs(embedder, text, vector)
                    if not proof:
                        raise EmbeddingUnavailableError("the embedder stopped reporting its embedding space mid-run")
                    backend.upsert_vector(entry.id, vector, namespace=namespace, **proof)
                    reembedded += 1

    warm_examined, warm_reembedded = await asyncio.to_thread(_reembed_warm, client, embedder, space, batch_size)
    result: ReembedResultDict = {
        "namespace": namespace,
        "embedding_model": client._config.embedding_model,
        "embedding_space": space.encoding,
        "examined": examined,
        "reembedded": reembedded,
        "already_current": already_current,
        "skipped": skipped,
        "warm_examined": warm_examined,
        "warm_reembedded": warm_reembedded,
    }
    logger.info("memory_reembed_complete", op="reembed", **result)
    return result


def _reembed_warm(
    client: MemoryClient, embedder: EmbeddingProvider, space: EmbeddingSpace, batch_size: int
) -> tuple[int, int]:
    from trw_memory.lifecycle.tiers._runtime import _TIER_MANAGER_CACHE_LOCK, get_tier_manager, tier_runtime_enabled
    from trw_memory.lifecycle.tiers._warm_space import reembed_warm_vectors

    if not tier_runtime_enabled(client._config):
        return 0, 0
    with _TIER_MANAGER_CACHE_LOCK:
        manager = get_tier_manager(client._config, client._namespace)
        return reembed_warm_vectors(manager._warm_store, embedder, space, batch_size=batch_size)
