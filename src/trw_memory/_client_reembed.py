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
- **Bounded per daemon call**: the shared daemon's ``memory_reembed`` does one
  pass of at most ``REEMBED_CALL_ROWS`` rows or ``REEMBED_CALL_SECONDS`` and
  returns a ``cursor`` to call again with (rc9 sweep B2); the SDK, alone in its
  process, runs to the end in one call. The warm tier's vectors are encoded
  outside the process-wide tier lock, which guards only each page's snapshot
  and conditional write (rc9 sweep C1).
- **Never downloads**: the embedder is the same cached provider recall uses, and
  an uncached model raises ``ModelNotCachedError`` (PLAN W40).
- **Fail closed**: no embedder, or an embedder that cannot name its space,
  raises -- re-encoding into an unidentifiable space would only mint more
  vectors dense recall must refuse.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import structlog
from typing_extensions import TypedDict

from trw_memory._sweep import decode_token, encode_token, sweep
from trw_memory.embeddings._space_gate import active_embedding_space, vector_in_space
from trw_memory.embeddings.provenance import generation_provenance_kwargs
from trw_memory.exceptions import EmbeddingUnavailableError, ModelNotCachedError, StorageError
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import EntryCursor

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.embeddings.provenance import EmbeddingSpace
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "DEFAULT_REEMBED_BATCH_SIZE",
    "MAX_REEMBED_BATCH",
    "ReembedResultDict",
    "reembed_namespace",
    "reembed_rows",
]

logger = structlog.get_logger(__name__)

DEFAULT_REEMBED_BATCH_SIZE = 64

# Same cap _graph_decay.py already applies to its own batch_size (precedent for
# "generous multiple of the default, still bounded"). A shared daemon serves every
# tenant from one process; each page hydrates `batch_size` entries + vector records
# and embeds every stale text in a single embed_batch() call, so an unbounded value
# (e.g. INT_MAX) is a single-caller memory/CPU exhaustion of the whole daemon.
MAX_REEMBED_BATCH = 1000

#: What one shared-daemon ``memory_reembed`` call may spend before it hands back a cursor.
REEMBED_CALL_ROWS = 512
REEMBED_CALL_SECONDS = 2.0


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
    cursor: str | None


async def reembed_namespace(client: MemoryClient, *, batch_size: int = DEFAULT_REEMBED_BATCH_SIZE) -> ReembedResultDict:
    """Re-encode every vector in the client's namespace not in the active space."""
    if batch_size < 1 or batch_size > MAX_REEMBED_BATCH:
        raise ValueError(f"batch_size must be between 1 and {MAX_REEMBED_BATCH}, got {batch_size}")
    client._require_permission(Permission.WRITE, "reembed")
    embedder = client._get_embedder()
    if embedder is None and client._embedder_refusal:
        # Re-embedding is the one operation that cannot degrade to keyword-only.
        raise ModelNotCachedError(client._embedder_refusal)
    if embedder is None:
        raise EmbeddingUnavailableError(
            f"no embedding model is available for {client._config.embedding_model!r}; "
            "install the extra with `pip install trw-memory[embeddings]`"
        )
    return await asyncio.to_thread(
        reembed_rows,
        client._get_backend(),
        embedder,
        namespace=client._namespace,
        config=client._config,
        batch_size=batch_size,
    )


def reembed_rows(
    backend: StorageBackend,
    embedder: EmbeddingProvider,
    *,
    namespace: str,
    config: MemoryConfig,
    batch_size: int = DEFAULT_REEMBED_BATCH_SIZE,
    cursor: str | None = None,
    bounded: bool = False,
) -> ReembedResultDict:
    """Re-encode the vectors of *namespace* in *backend* (then its warm tier's) not in *embedder*'s space.

    The one implementation behind the SDK's ``reembed`` and the daemon's
    ``memory_reembed`` (PRD-CORE-302 FR07). Each page commits on its own; the
    backend's lock serializes it against concurrent writers. *bounded* stops at
    the per-call budget and returns the ``cursor`` to resume from (``None`` when
    done); a malformed *cursor* raises ``ValueError``.
    """
    if batch_size < 1 or batch_size > MAX_REEMBED_BATCH:
        raise ValueError(f"batch_size must be between 1 and {MAX_REEMBED_BATCH}, got {batch_size}")
    space = active_embedding_space(embedder)
    if space is None:
        raise EmbeddingUnavailableError("the embedder does not report an embedding space; refusing to re-embed")
    if not backend.supports_vectors():
        raise StorageError("this backend stores no vectors (sqlite-vec unavailable); nothing can be re-embedded")
    phase, updated_at, entry_id = decode_token(cursor, 3) if cursor is not None else ("rows", "", "")
    if phase not in ("rows", "warm"):
        raise ValueError("malformed cursor")
    rows, seconds = (REEMBED_CALL_ROWS, REEMBED_CALL_SECONDS) if bounded else (sys.maxsize, math.inf)
    counts = dict.fromkeys(
        ("examined", "reembedded", "already_current", "skipped", "warm_examined", "warm_reembedded"), 0
    )
    token: str | None = None
    if phase == "rows":
        after = EntryCursor(updated_at, entry_id) if entry_id else None
        resume = sweep(
            lambda key, limit: backend.list_entries(namespace=namespace, limit=limit, after=key),
            EntryCursor.from_entry,
            lambda page, _deadline: _reembed_page(backend, embedder, space, namespace, page, counts),
            after=after,
            page=batch_size,
            rows=rows,
            seconds=seconds,
        )
        if resume is not None:
            token = encode_token(["rows", resume.updated_at, resume.entry_id])
        elif bounded:
            token = encode_token(["warm", "", ""])
    if token is None:
        warm_after = entry_id if phase == "warm" and entry_id else None
        resume_id = _reembed_warm(config, namespace, embedder, space, warm_after, batch_size, rows, seconds, counts)
        token = None if resume_id is None else encode_token(["warm", "", resume_id])
    result: ReembedResultDict = {
        "namespace": namespace,
        "embedding_model": config.embedding_model,
        "embedding_space": space.encoding,
        "examined": counts["examined"],
        "reembedded": counts["reembedded"],
        "already_current": counts["already_current"],
        "skipped": counts["skipped"],
        "warm_examined": counts["warm_examined"],
        "warm_reembedded": counts["warm_reembedded"],
        "cursor": token,
    }
    logger.info("memory_reembed_pass", op="reembed", **result)
    return result


def _reembed_page(
    backend: StorageBackend,
    embedder: EmbeddingProvider,
    space: EmbeddingSpace,
    namespace: str,
    batch: Sequence[MemoryEntry],
    counts: dict[str, int],
) -> int:
    records = backend.get_vector_records([entry.id for entry in batch], namespace=namespace)
    # SEC-001 canaries are never recalled, so they need no vector.
    rows = [entry for entry in batch if entry.metadata.get("system_canary") != "true"]
    counts["examined"] += len(rows)
    stale = [entry for entry in rows if not vector_in_space(records.get(entry.id), space)]
    counts["already_current"] += len(rows) - len(stale)
    texts = [f"{entry.content} {entry.detail}" for entry in stale]
    vectors = embedder.embed_batch(texts) if texts else []
    with backend.transaction():
        now = backend.get_vector_records([entry.id for entry in stale], namespace=namespace)
        for entry, text, vector in zip(stale, texts, vectors, strict=True):
            # Encoded outside this transaction: a row rewritten, deleted or re-vectored since
            # then already carries its own write's vector, so this one would be stale (C12).
            current = backend.get(entry.id, namespace=namespace)
            changed = current is None or f"{current.content} {current.detail}" != text
            if vector is None or changed or now.get(entry.id) != records.get(entry.id):
                counts["skipped"] += 1
                continue
            proof = generation_provenance_kwargs(embedder, text, vector)
            if not proof:
                raise EmbeddingUnavailableError("the embedder stopped reporting its embedding space mid-run")
            backend.upsert_vector(entry.id, vector, namespace=namespace, **proof)
            counts["reembedded"] += 1
    return len(batch)


def _reembed_warm(
    config: MemoryConfig,
    namespace: str,
    embedder: EmbeddingProvider,
    space: EmbeddingSpace,
    after: str | None,
    page: int,
    rows: int,
    seconds: float,
    counts: dict[str, int],
) -> str | None:
    """The warm tier's half, by sorted sidecar id: the id to resume after, or ``None`` when done."""
    from trw_memory.lifecycle.tiers._runtime import _TIER_MANAGER_CACHE_LOCK, get_tier_manager, tier_runtime_enabled
    from trw_memory.lifecycle.tiers._warm_space import WarmSnapshot, commit_warm_page, warm_page

    if not tier_runtime_enabled(config):
        return None

    def fetch(key: str | None, limit: int) -> list[WarmSnapshot]:
        with _TIER_MANAGER_CACHE_LOCK:
            return warm_page(get_tier_manager(config, namespace)._warm_store, space, key, limit)

    def visit(snapshot: Sequence[WarmSnapshot], _deadline: float) -> int:
        vectored = [item for item in snapshot if item.record is not None]
        counts["warm_examined"] += len(vectored)
        stale = [item for item in vectored if not vector_in_space(item.record, space)]
        vectors = embedder.embed_batch([item.text for item in stale]) if stale else []  # outside the tier lock
        with _TIER_MANAGER_CACHE_LOCK:
            store = get_tier_manager(config, namespace)._warm_store
            counts["warm_reembedded"] += commit_warm_page(store, embedder, space, stale, vectors)
        return len(snapshot)

    return sweep(fetch, lambda item: item.entry_id, visit, after=after, page=page, rows=rows, seconds=seconds)
