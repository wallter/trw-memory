"""Remote fetch helpers for memory sync."""

from __future__ import annotations

import json

import httpx
import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.security.pii import mask_query_credentials
from trw_memory.sync._remote_common import (
    FETCH_TIMEOUT,
    _raise_local_only_violation,
    build_platform_headers,
    decode_learning_api_v1_result,
    encode_learning_api_v1_search,
    is_valid_platform_url,
)

logger = structlog.get_logger(__name__)


def _dedupe_shared_results(
    results: list[dict[str, object]],
    *,
    local_entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    dedup_threshold: float,
) -> list[dict[str, object]]:
    if not local_entries:
        return results

    local_ids = {entry.id for entry in local_entries}
    local_contents = {entry.content.lower().strip() for entry in local_entries}
    filtered: list[dict[str, object]] = []
    remote_texts: list[str] = []
    remote_candidates: list[dict[str, object]] = []

    for result in results:
        source_learning_id = str(result.get("source_learning_id", "")).strip()
        summary = str(result.get("summary", result.get("content", ""))).strip()
        detail = str(result.get("detail", "")).strip()
        if source_learning_id and source_learning_id in local_ids:
            continue
        if summary.lower() in local_contents:
            continue
        remote_candidates.append(result)
        remote_texts.append(f"{summary} {detail}".strip())

    if not remote_candidates or embedder is None:
        return remote_candidates

    try:
        if not embedder.available():
            return remote_candidates

        local_texts = [f"{entry.content} {entry.detail}".strip() for entry in local_entries]
        texts = [*local_texts, *remote_texts]
        vectors = embedder.embed_batch(texts)
        if len(vectors) != len(texts):
            raise ValueError(f"embedding batch length mismatch: expected {len(texts)}, got {len(vectors)}")
        local_vectors = [vector for vector in vectors[: len(local_entries)] if vector is not None]
        remote_vectors = vectors[len(local_entries) :]
        if not local_vectors:
            return remote_candidates

        for result, remote_vector in zip(remote_candidates, remote_vectors, strict=False):
            if remote_vector is None:
                filtered.append(result)
                continue
            if any(cosine_similarity(remote_vector, local_vector) > dedup_threshold for local_vector in local_vectors):
                continue
            filtered.append(result)
    except Exception:
        logger.warning("memory_shared_semantic_dedup_failed", exc_info=True)
        return remote_candidates
    return filtered


def fetch_shared_memories(
    query: str,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    limit: int = 10,
    local_entries: list[MemoryEntry] | None = None,
    embedder: EmbeddingProvider | None = None,
    dedup_threshold: float = 0.92,
) -> list[dict[str, object]]:
    if cfg.local_only:
        logger.warning("memory_fetch_blocked_local_only")
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url:
        return []
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_fetch_invalid_platform_url")
        return []

    request_payload: dict[str, object] = encode_learning_api_v1_search(
        query=mask_query_credentials(query), limit=limit, min_importance=cfg.sync_min_importance
    )
    if embedding:
        request_payload["embedding"] = embedding

    try:
        with httpx.Client(timeout=FETCH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings/search",
                json=request_payload,
                headers=build_platform_headers(cfg.platform_api_key),
            )
            # A failed fetch and an empty shared corpus both arrive as ``[]``, and
            # the caller merges either one identically. Log the failure (status
            # only — never the body) so "the platform is 401ing on a rotated key"
            # is distinguishable from "nothing matched"; absence of a measurement
            # is not a measurement of absence.
            if resp.status_code != 200:
                logger.warning("memory_fetch_rejected", status_code=resp.status_code)
                return []
            raw = resp.json()
            if isinstance(raw, list):
                raw_results = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict):
                wrapped = raw.get("results", raw.get("items", []))
                raw_results = [item for item in wrapped if isinstance(item, dict)] if isinstance(wrapped, list) else []
            else:
                logger.warning("memory_fetch_malformed_body", body_type=type(raw).__name__)
                return []
            # Decode external wire vocabulary into canonical importance at the boundary.
            results = [decode_learning_api_v1_result(item) for item in raw_results]
    except (httpx.HTTPError, OSError, ConnectionError, json.JSONDecodeError, ValueError):
        logger.debug("memory_fetch_error", exc_info=True)
        return []

    deduped = _dedupe_shared_results(
        results,
        local_entries=local_entries or [],
        embedder=embedder,
        dedup_threshold=dedup_threshold,
    )
    shared: list[dict[str, object]] = []
    for result in deduped:
        summary = result.get("summary", result.get("content", ""))
        content = str(summary) if summary is not None else ""
        result["summary"] = f"[shared] {content}"
        result["content"] = f"[shared] {content}"
        result["source"] = "shared"
        shared.append(result)

    logger.debug("memory_fetch_complete", fetched=len(results), after_dedup=len(shared))
    return shared
