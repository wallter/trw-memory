"""Org-shared cluster — SSE cache + remote fetch + dedup.

Belongs to ``client.py``. Re-exported there for back-compat.

11 helpers covering shared-memory retrieval (fetched from a remote
SSE stream + per-namespace cache):

- ``merge_shared_results`` — fetch + dedupe shared remote candidates.
- ``load_entries_for_results`` — materialize local entries for dedup.
- ``shared_result_to_result`` — normalize remote payload to result dict.
- ``coerce_float`` — loose payload → float fallback.
- ``is_retired_shared_result`` — detect remote retirement markers.
- ``merge_shared_candidates`` — append shared after local, dedup.
- ``snapshot_cached_shared_results`` — query-filtered SSE-cache view.
- ``matches_query`` — simple token-match for cache filter.
- ``dedupe_cached_shared_results`` — exact + semantic dedup.
- ``strip_shared_prefix`` — normalize ``[shared] `` prefix.
- ``mark_fetch_retirements`` — track retirement markers.

All async helpers take ``client: MemoryClient`` as first arg
(backend-handle pattern).

Extracted as PRD-DIST-246 batch 107.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.sync._remote_common import decode_learning_api_v1_result

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger


async def merge_shared_results(
    client: MemoryClient,
    query: str,
    local_results: list[MemoryResultDict],
    limit: int,
) -> list[MemoryResultDict]:
    """Fetch shared memories and append them after local results."""
    try:
        await client._apply_pending_remote_retirements()
        local_entries = await load_entries_for_results(client, local_results)
        embedder = client._get_embedder()
        cached_shared = await dedupe_cached_shared_results(
            client,
            snapshot_cached_shared_results(client, query),
            local_entries=local_entries,
            embedder=embedder,
        )
        query_embedding: list[float] | None = None
        if embedder is not None and query.strip():
            query_embedding = await asyncio.to_thread(embedder.embed, query)

        # Look up `fetch_shared_memories` via parent module so test patches
        # on `trw_memory.client.fetch_shared_memories` propagate.
        from trw_memory import client as _client_module

        _fetch = _client_module.fetch_shared_memories
        fetched = await asyncio.to_thread(
            functools.partial(
                _fetch,
                query,
                client._config,
                backend=client._get_backend(),
                embedding=query_embedding,
                limit=limit,
                local_entries=local_entries,
                embedder=embedder,
            )
        )
        shared = fetched.results
        if fetched.status not in {"ok", "disabled"}:
            # An empty ``shared`` here is not necessarily an empty shared
            # corpus. Say which it was, at a level an operator will see.
            _client_logger().warning(
                "memory_shared_fetch_degraded",
                op="recall",
                outcome=fetched.status,
                namespace=client._namespace,
                fetched=fetched.fetched,
                refused=fetched.refused,
            )
    except Exception:
        _client_logger().debug(
            "memory_shared_recall_failed",
            op="recall",
            outcome="failure",
            namespace=client._namespace,
            exc_info=True,
        )
        return merge_shared_candidates(local_results, snapshot_cached_shared_results(client, query))
    await mark_fetch_retirements(client, shared)
    live_shared = [shared_result_to_result(item) for item in shared if not is_retired_shared_result(item)]
    return merge_shared_candidates(local_results, [*live_shared, *cached_shared])


async def load_entries_for_results(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> list[MemoryEntry]:
    """Materialize local entries for dedup against shared results."""
    result_ids = [result["memory_id"] for result in results if result.get("source", "local") == "local"]
    if not result_ids:
        return []

    async with client._lock:
        backend = client._get_backend()
        loaded: list[MemoryEntry] = []
        for entry_id in result_ids:
            entry = backend.get(entry_id, namespace=client._namespace)
            if entry is not None:
                loaded.append(entry)
        return loaded


def shared_result_to_result(result: dict[str, object]) -> MemoryResultDict:
    """Normalize a shared remote result into the client result shape."""
    # This dict arrives from the remote/org learning API, so its external
    # impact vocabulary MUST cross the versioned learning_api_v1 boundary rather
    # than a local fallback read — that is the single place the wire field is
    # mapped onto canonical importance (PRD-CORE-181-FR06 source-census rule).
    decoded = decode_learning_api_v1_result(dict(result))
    memory_id = str(decoded.get("memory_id", decoded.get("id", decoded.get("remote_id", ""))))
    detail = str(decoded.get("detail", ""))
    raw_tags = decoded.get("tags", [])
    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
    importance_raw = decoded.get("importance", 0.0)
    score_raw = decoded.get("score", importance_raw)
    namespace = str(decoded.get("namespace", "shared"))
    created_at = str(decoded.get("created_at", ""))
    updated_at = str(decoded.get("updated_at", created_at))
    source = str(decoded.get("source", "shared"))
    shared_result: MemoryResultDict = {
        "memory_id": memory_id,
        "content": str(decoded.get("content", "")),
        "detail": detail,
        "tags": tags,
        "importance": coerce_float(importance_raw),
        "score": coerce_float(score_raw),
        "created_at": created_at,
        "updated_at": updated_at,
        "namespace": namespace,
        "source": source,
    }
    return shared_result


def coerce_float(value: object) -> float:
    """Convert loosely typed payload values into floats with a safe default."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def is_retired_shared_result(result: dict[str, object]) -> bool:
    """Return whether a shared result represents a remote retirement."""
    status = str(result.get("status", "")).lower()
    return status in {"obsolete", "deleted"}


def merge_shared_candidates(
    local_results: list[MemoryResultDict],
    shared_results: list[MemoryResultDict],
) -> list[MemoryResultDict]:
    """Append shared results after local ones while suppressing exact duplicates."""
    seen_ids = {result["memory_id"] for result in local_results}
    seen_content = {result["content"] for result in local_results}
    merged = list(local_results)
    for result in shared_results:
        if result["memory_id"] in seen_ids or result["content"] in seen_content:
            continue
        merged.append(result)
        seen_ids.add(result["memory_id"])
        seen_content.add(result["content"])
    return merged


def snapshot_cached_shared_results(
    client: MemoryClient,
    query: str,
) -> list[MemoryResultDict]:
    """Return cached SSE shared results relevant to the current query."""
    with client._shared_event_cache_lock:
        cached = list(client._shared_event_cache)
    if not query.strip():
        return cached
    return [result for result in cached if matches_query(result, query)]


def matches_query(result: MemoryResultDict, query: str) -> bool:
    """Apply the same simple token matching used by fallback recall."""
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return True
    text = f"{result['content']} {result['detail']} {' '.join(result['tags'])}".lower()
    return any(term in text for term in query_terms)


async def dedupe_cached_shared_results(
    client: MemoryClient,
    cached_results: list[MemoryResultDict],
    *,
    local_entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    dedup_threshold: float = 0.92,
) -> list[MemoryResultDict]:
    """Apply the same exact/semantic dedup rules to cached SSE results."""
    if not cached_results or not local_entries:
        return cached_results

    local_remote_ids = {str(entry.remote_id) for entry in local_entries if entry.remote_id}
    local_contents = {entry.content.lower().strip() for entry in local_entries}

    candidates: list[MemoryResultDict] = []
    candidate_texts: list[str] = []
    for result in cached_results:
        normalized_content = strip_shared_prefix(result["content"]).strip()
        if result["memory_id"] in local_remote_ids or normalized_content.lower() in local_contents:
            continue
        candidates.append(result)
        candidate_texts.append(f"{normalized_content} {result['detail']}".strip())

    if not candidates or embedder is None or not embedder.available():
        return candidates

    local_texts = [f"{entry.content} {entry.detail}".strip() for entry in local_entries]
    vectors = await asyncio.to_thread(embedder.embed_batch, [*local_texts, *candidate_texts])
    local_vectors = [vector for vector in vectors[: len(local_entries)] if vector is not None]
    remote_vectors = vectors[len(local_entries) :]
    if not local_vectors:
        return candidates

    deduped: list[MemoryResultDict] = []
    for candidate, remote_vector in zip(candidates, remote_vectors, strict=False):
        if remote_vector is None:
            deduped.append(candidate)
            continue
        if any(cosine_similarity(remote_vector, local_vector) > dedup_threshold for local_vector in local_vectors):
            continue
        deduped.append(candidate)
    return deduped


def strip_shared_prefix(content: str) -> str:
    """Normalize cached shared content for dedup comparisons."""
    return content.removeprefix("[shared] ")


async def mark_fetch_retirements(
    client: MemoryClient,
    shared_results: list[dict[str, object]],
) -> None:
    """Record retirement markers returned from remote fetches."""
    remote_ids = {
        str(result.get("id", result.get("remote_id", "")))
        for result in shared_results
        if is_retired_shared_result(result)
    }
    if not remote_ids:
        return
    with client._pending_remote_retirements_lock:
        client._pending_remote_retirements.update(remote_id for remote_id in remote_ids if remote_id)
    await client._apply_pending_remote_retirements()
