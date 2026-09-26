"""MCP tool: memory_similar -- the trw_learn dedup verdict, decided where the store and the model are (PRD-CORE-302 FR01).

The caller sends the new learning's text and its reference-scale thresholds; the
daemon encodes the text in its active space, calibrates the thresholds into that
model's scale and answers skip, merge or store (contract C1). The KNN window is
trusted only when it is non-empty and every stored vector of the namespace is
provably in the active space; otherwise every row is compared, encoding the text
of rows without an active-space vector -- the 6.1.0 YAML-scan parity -- within a
daemon-owned budget, past which the answer is ``unavailable``. A read.
"""

from __future__ import annotations

import dataclasses
import time

from trw_memory.embeddings._space_gate import active_embedding_space, comparable_neighbours, select_space_vectors
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.lifecycle.dedup import _validated_thresholds
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import EntryCursor, StorageBackend
from trw_memory.tools._embedder import resolve_embedder
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import serve_namespace

#: Rows read and encoded per page in exhaustive mode, and what one call's exhaustive pass may spend
#: on the shared daemon (rc9 sweep B1): past either bound the answer is unavailable, never "store".
#: The clock is read after each page's fetch, so one page (a read and at most EXHAUSTIVE_PAGE
#: encodes, which cannot be interrupted) may finish past the deadline.
EXHAUSTIVE_PAGE = 64
EXHAUSTIVE_MAX_ROWS = 2_000
EXHAUSTIVE_SECONDS = 2.0
#: trw-mcp sends a summary (<= 2,000) and a detail (<= 4,000); a longer text is refused before the
#: shared model and its encode lock ever see it (rc9 sweep round 2).
MAX_SIMILAR_TEXT_CHARS = 8_000


def _knn(
    backend: StorageBackend, vector: list[float], space: EmbeddingSpace, namespace: str, top_k: int
) -> list[tuple[str, float, bool]] | None:
    """``(id, similarity, active)`` for a trustworthy window, else ``None`` (decide exhaustively)."""
    window = comparable_neighbours(backend, vector, space, namespace=namespace, top_k=top_k, surface="memory_similar")
    if not window:  # None: incomplete; []: empty, which 6.1.0 also scanned (a vectorless namespace)
        return None
    hits: list[tuple[str, float, bool]] = []
    for entry_id, distance in window:
        entry = backend.get(entry_id, namespace=namespace)
        if entry is not None:
            # Unit-normalised vectors: distance² = 2 * (1 - cosine_similarity).
            hits.append((entry_id, 1.0 - (distance * distance) / 2.0, entry.status == MemoryStatus.ACTIVE))
    return hits


def _exhaustive(
    backend: StorageBackend, vector: list[float], space: EmbeddingSpace, embedder: EmbeddingProvider, namespace: str
) -> tuple[list[tuple[str, float, bool]], int] | None:
    """Every row of *namespace*, whatever its status: its active-space vector, else its text encoded
    now; ``None`` once the pass would outrun its budget (the hits it keeps are bounded with it)."""
    hits: list[tuple[str, float, bool]] = []
    examined = 0
    deadline = time.monotonic() + EXHAUSTIVE_SECONDS
    cursor: EntryCursor | None = None
    while time.monotonic() <= deadline:
        page = backend.list_entries(namespace=namespace, limit=EXHAUSTIVE_PAGE, after=cursor)
        if not page:
            return hits, examined
        if examined + len(page) > EXHAUSTIVE_MAX_ROWS or time.monotonic() > deadline:
            return None  # a slow read spends the budget as surely as an encode
        cursor = EntryCursor.from_entry(page[-1])
        stored = select_space_vectors(backend.get_vector_records([e.id for e in page], namespace=namespace), space)
        missing: list[MemoryEntry] = [e for e in page if e.id not in stored.vectors]
        encoded = embedder.embed_batch([f"{e.content} {e.detail}" for e in missing]) if missing else []
        vectors = dict(stored.vectors) | {e.id: v for e, v in zip(missing, encoded, strict=True) if v is not None}
        for entry in page:
            examined += 1
            candidate = vectors.get(entry.id)
            if candidate is None or len(candidate) != len(vector):
                continue  # no vector, or another width: not a duplicate
            hits.append((entry.id, cosine_similarity(vector, candidate), entry.status == MemoryStatus.ACTIVE))
        if len(page) < EXHAUSTIVE_PAGE:  # the last page: no further read, inside the budget or not
            return hits, examined
    return None


def memory_similar_impl(
    namespace: str,
    text: str,
    skip_threshold: float,
    merge_threshold: float,
    top_k: int,
    *,
    backend: StorageBackend,
    config: MemoryConfig,
) -> dict[str, object]:
    """The contract C1 answer: a verdict, ``invalid`` with a code, or ``unavailable`` with a reason."""
    started = time.monotonic()
    if not text.strip():
        return {"status": "invalid", "code": "empty_text", "error": "text is empty after stripping"}
    if not (-1.0 <= merge_threshold <= 1.0 and -1.0 <= skip_threshold <= 1.0):
        return {"status": "invalid", "code": "bad_thresholds", "error": "thresholds must lie in [-1, 1]"}
    embedder = resolve_embedder(config, surface="memory_similar")
    if isinstance(embedder, dict):
        return embedder
    vector = embedder.embed(text)
    space = active_embedding_space(embedder)
    if vector is None or space is None:
        return {"status": "unavailable", "reason": "embedder_error"}
    skip, merge = _validated_thresholds(
        MemoryConfig(dedup_skip_threshold=skip_threshold, dedup_merge_threshold=merge_threshold), embedder
    )
    hits = _knn(backend, vector, space, namespace, top_k)
    mode, examined = "knn", len(hits or ())
    if hits is None:
        if (scanned := _exhaustive(backend, vector, space, embedder, namespace)) is None:
            return {"status": "unavailable", "reason": "similar_budget", "fix": "trw-mcp memory reembed"}
        mode, (hits, examined) = "exhaustive", scanned
    best = max(hits, key=lambda hit: hit[1], default=None)
    action, existing_id, similarity = "store", None, max(best[1], 0.0) if best else 0.0
    if best is not None and best[1] >= skip:
        action, existing_id = "skip", best[0]
    elif best is not None and best[1] >= merge and best[2]:
        action, existing_id = "merge", best[0]
    return {
        "status": "ok",
        "action": action,
        "existing_id": existing_id,
        "similarity": similarity,
        "mode": mode,
        "examined": examined,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "space": dataclasses.asdict(space),
    }


def register_similar_tool(mcp: McpServer) -> None:
    """Register memory_similar with a FastMCP server instance."""

    async def memory_similar(
        namespace: str, text: str, skip_threshold: float, merge_threshold: float, top_k: int = 10
    ) -> dict[str, object]:
        """Skip, merge or store for a new learning's *text* in *namespace*, against the caller's reference thresholds."""
        # Off the event loop: exhaustive mode encodes a namespace, which must not stall other callers.
        return await serve_namespace(
            namespace,
            Permission.READ,
            "similar",
            lambda backend, config: memory_similar_impl(
                namespace, text, skip_threshold, merge_threshold, top_k, backend=backend, config=config
            ),
            exclusive=False,
        )

    mcp.tool()(memory_similar)


__all__ = ["memory_similar_impl", "register_similar_tool"]
