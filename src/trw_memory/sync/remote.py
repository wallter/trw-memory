"""Remote sync -- publish and fetch memory entries to/from the platform.

Implements FR01 (publish pipeline), FR02 (fetch pipeline), and FR07
(anonymization) from PRD-CORE-047.  All network operations are fail-open:
they never raise exceptions to the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import urlparse

import httpx
import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.security.pii import anonymize_installation_id, redact_paths, strip_pii
from trw_memory.sync.retry_queue import RetryQueue

logger = structlog.get_logger(__name__)

PUBLISH_TIMEOUT = 5.0  # seconds
FETCH_TIMEOUT = 3.0

MAX_SUMMARY_LENGTH = 1000
MAX_DETAIL_LENGTH = 10_000
MAX_TAGS_COUNT = 20
LOCAL_ONLY_ERROR_MESSAGE = "Operation blocked: memory_local_only=True disables all network access."


class AnonymizedEntry(TypedDict):
    """Typed structure for an anonymized memory entry ready for remote publish."""

    summary: str
    detail: str | None
    tags: list[str]
    impact: float
    embedding: list[float] | None
    source_project: str
    source_learning_id: str


class PublishResult(TypedDict):
    """Typed publish result with success flag and optional remote identifier."""

    success: bool
    remote_id: str | None
    retryable: bool


def _raise_local_only_violation() -> None:
    """Raise the shared local-only guard error for network entrypoints."""
    raise LocalOnlyViolationError(LOCAL_ONLY_ERROR_MESSAGE)


def is_valid_platform_url(platform_url: str) -> bool:
    """Allow only https URLs unless TRW_DEBUG explicitly enables http."""
    if not platform_url.strip():
        return False
    parsed = urlparse(platform_url)
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and os.getenv("TRW_DEBUG", "").lower() == "true"


def _anonymize_entry(
    entry: MemoryEntry,
    project_root: str = "",
) -> AnonymizedEntry:
    """Anonymize a MemoryEntry for remote transmission (FR07).

    Applies the full anonymization pipeline:
    1. ``strip_pii()`` on content and detail
    2. ``redact_paths()`` on content and detail
    3. ``anonymize_installation_id()`` on the installation ID
    4. Truncate content to 1000 chars, detail to 10000 chars, tags to 20 items
    """
    content = strip_pii(entry.content)
    content = redact_paths(content, project_root)

    detail = strip_pii(entry.detail)
    detail = redact_paths(detail, project_root)

    return AnonymizedEntry(
        summary=content[:MAX_SUMMARY_LENGTH],
        detail=detail[:MAX_DETAIL_LENGTH] if detail else None,
        tags=entry.tags[:MAX_TAGS_COUNT],
        impact=entry.importance,
        embedding=None,  # populated by caller if available
        source_project=anonymize_installation_id(
            entry.metadata.get("installation_id", ""),
        ),
        source_learning_id=entry.id,
    )


def publish_memory_result(
    entry: MemoryEntry,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    project_root: str = "",
) -> PublishResult:
    """Publish a memory entry and return the success flag plus remote ID."""
    if cfg.local_only:
        logger.warning("memory_publish_blocked_local_only", entry_id=entry.id)
        _raise_local_only_violation()

    if not cfg.sync_enabled or not cfg.platform_url:
        return {"success": False, "remote_id": None, "retryable": False}
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_publish_invalid_platform_url", entry_id=entry.id)
        return {"success": False, "remote_id": None, "retryable": False}

    if entry.importance < cfg.sync_min_importance:
        return {"success": False, "remote_id": None, "retryable": False}

    payload = _anonymize_entry(entry, project_root)
    if embedding:
        payload["embedding"] = embedding

    return _publish_payload_result(cast("dict[str, object]", payload), cfg, entry_id=entry.id)


def publish_memory(
    entry: MemoryEntry,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    project_root: str = "",
) -> bool:
    """Publish a memory entry to the remote platform (FR01).

    Returns ``True`` when the publish is handled without retry, including
    deliberate no-op cases such as local-only mode, disabled sync, empty remote
    config, or importance below the publish threshold. Returns ``False`` only
    for retryable remote failures. Fail-open: never raises exceptions.
    """
    result = publish_memory_result(
        entry,
        cfg,
        embedding=embedding,
        project_root=project_root,
    )
    return result["success"] or not result["retryable"]


def _publish_payload(
    payload: dict[str, object],
    cfg: MemoryConfig,
    *,
    entry_id: str = "",
) -> bool:
    """Publish a prepared payload to the remote platform."""
    return _publish_payload_result(payload, cfg, entry_id=entry_id)["success"]


def _publish_payload_result(
    payload: dict[str, object],
    cfg: MemoryConfig,
    *,
    entry_id: str = "",
) -> PublishResult:
    """Publish a prepared payload and extract the remote ID when available."""
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_publish_invalid_platform_url", entry_id=entry_id)
        return {"success": False, "remote_id": None, "retryable": False}

    headers: dict[str, str] = {}
    if cfg.platform_api_key:
        headers["Authorization"] = f"Bearer {cfg.platform_api_key}"
    headers["Content-Type"] = "application/json"

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings",
                json=payload,
                headers=headers,
            )
            if 200 <= resp.status_code < 300:
                remote_id: str | None = None
                try:
                    raw_body = resp.json()
                except (json.JSONDecodeError, ValueError, TypeError):
                    raw_body = None
                if isinstance(raw_body, dict):
                    raw_remote_id = raw_body.get("id")
                    if raw_remote_id is not None:
                        remote_id = str(raw_remote_id)
                logger.debug("memory_published", entry_id=entry_id, remote_id=remote_id)
                return {"success": True, "remote_id": remote_id, "retryable": False}
            logger.warning(
                "memory_publish_failed",
                entry_id=entry_id,
                status=resp.status_code,
            )
            return {"success": False, "remote_id": None, "retryable": True}
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_publish_error", entry_id=entry_id, exc_info=True)
        return {"success": False, "remote_id": None, "retryable": True}


def drain_retry_queue(queue: RetryQueue, cfg: MemoryConfig) -> dict[str, int]:
    """Drain queued publish payloads when sync is enabled and reachable."""
    if cfg.local_only:
        logger.warning("memory_retry_drain_blocked_local_only")
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url:
        return {"drained": 0, "failed": 0, "skipped": queue.depth()}
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_retry_drain_invalid_platform_url")
        return {"drained": 0, "failed": 0, "skipped": queue.depth()}
    return queue.drain(lambda payload: _publish_payload(payload, cfg))


def clear_retry_queue(queue: RetryQueue) -> None:
    """Clear all queued publish payloads."""
    queue.clear()


class SnapshotHashPayload(TypedDict):
    """Metadata-only envelope published for snapshot drift notification (PRD-INFRA-066 / C1).

    Critical invariant: this payload MUST NEVER carry snapshot contents — only
    a SHA-256 digest, size, timestamp, and anonymized installation_id. The
    existence of this type is the structural guard that prevents an accidental
    contents leak at the call site.
    """

    digest: str
    size_bytes: int
    created_at: str
    installation_id: str


def publish_snapshot_hash(
    snapshot_path: Path,
    cfg: MemoryConfig,
    *,
    installation_id: str = "",
) -> PublishResult:
    """Publish SHA-256 of a snapshot (NOT contents) to the platform (PRD-INFRA-066 / C1).

    Computes a SHA-256 of ``snapshot_path``'s bytes and posts a metadata-only
    envelope (:class:`SnapshotHashPayload`) to the platform. Contents are
    never transmitted. Gated on:

    - ``cfg.local_only is False`` (raises ``LocalOnlyViolationError`` if True)
    - ``cfg.sync_enabled is True``
    - ``cfg.memory_snapshot_publish_hash is True``
    - ``cfg.platform_url`` is set and valid

    When any condition is unmet, returns a non-retryable skip result without
    making a network call.

    Args:
        snapshot_path: Path to the snapshot file (e.g. from B4 rotation).
        cfg: Memory configuration.
        installation_id: Optional raw installation id; anonymized before publish.

    Returns:
        :class:`PublishResult` — ``success=True`` on 2xx, ``retryable=True`` on
        transport/5xx errors, ``retryable=False`` for skip paths.
    """
    if cfg.local_only:
        logger.warning(
            "snapshot_hash_publish_blocked_local_only",
            snapshot=str(snapshot_path),
        )
        _raise_local_only_violation()

    if not cfg.sync_enabled or not cfg.memory_snapshot_publish_hash:
        return {"success": False, "remote_id": None, "retryable": False}
    if not cfg.platform_url:
        return {"success": False, "remote_id": None, "retryable": False}
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning(
            "snapshot_hash_publish_invalid_platform_url",
            snapshot=str(snapshot_path),
        )
        return {"success": False, "remote_id": None, "retryable": False}

    if not snapshot_path.exists() or not snapshot_path.is_file():
        logger.debug(
            "snapshot_hash_publish_missing_file",
            snapshot=str(snapshot_path),
        )
        return {"success": False, "remote_id": None, "retryable": False}

    try:
        digest, size_bytes = _hash_snapshot_file(snapshot_path)
    except OSError as exc:
        logger.debug(
            "snapshot_hash_publish_read_failed",
            snapshot=str(snapshot_path),
            error=str(exc),
        )
        return {"success": False, "remote_id": None, "retryable": True}

    payload: SnapshotHashPayload = {
        "digest": digest,
        "size_bytes": size_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "installation_id": anonymize_installation_id(installation_id or ""),
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.platform_api_key:
        headers["Authorization"] = f"Bearer {cfg.platform_api_key}"

    endpoint = f"{cfg.platform_url.rstrip('/')}/v1/memory/snapshot-hash"

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.post(endpoint, json=cast("dict[str, object]", payload), headers=headers)
            if 200 <= resp.status_code < 300:
                logger.debug(
                    "snapshot_hash_published",
                    digest=digest,
                    size_bytes=size_bytes,
                )
                return {"success": True, "remote_id": None, "retryable": False}
            logger.warning(
                "snapshot_hash_publish_failed",
                status=resp.status_code,
                digest=digest,
            )
            return {"success": False, "remote_id": None, "retryable": True}
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("snapshot_hash_publish_error", exc_info=True)
        return {"success": False, "remote_id": None, "retryable": True}


def _hash_snapshot_file(path: Path, chunk_size: int = 65536) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` for ``path`` without loading it fully."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def retire_remote_memory(remote_id: str, cfg: MemoryConfig) -> bool:
    """Mark a remote memory obsolete using the backend status endpoint.

    The backend currently models retirement as a status transition rather than a
    hard DELETE. This keeps package-side deletion propagation aligned with the
    shipped backend contract while remaining fail-open for local callers.
    """
    if cfg.local_only:
        logger.warning("memory_retire_blocked_local_only", remote_id=remote_id)
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url or not remote_id:
        return True
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_retire_invalid_platform_url", remote_id=remote_id)
        return True

    headers: dict[str, str] = {}
    if cfg.platform_api_key:
        headers["Authorization"] = f"Bearer {cfg.platform_api_key}"
    headers["Content-Type"] = "application/json"

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.patch(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings/{remote_id}/status",
                json={"status": "obsolete"},
                headers=headers,
            )
            if 200 <= resp.status_code < 300:
                logger.debug("memory_retired_remote", remote_id=remote_id)
                return True
            logger.warning("memory_retire_failed", remote_id=remote_id, status=resp.status_code)
            return False
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_retire_error", remote_id=remote_id, exc_info=True)
        return False


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
    """Fetch shared memories from the remote platform (FR02).

    Returns a list of remote memory dicts with ``[shared]`` prefix on content.
    Deduplicates against *local_entries* by content string match.
    Fail-open: returns empty list on any error.
    """
    if cfg.local_only:
        logger.warning("memory_fetch_blocked_local_only")
        _raise_local_only_violation()

    if not cfg.sync_enabled or not cfg.platform_url:
        return []
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_fetch_invalid_platform_url")
        return []

    request_payload: dict[str, object] = {
        "query": query,
        "limit": limit,
        "min_impact": cfg.sync_min_importance,
    }
    if embedding:
        request_payload["embedding"] = embedding

    headers: dict[str, str] = {}
    if cfg.platform_api_key:
        headers["Authorization"] = f"Bearer {cfg.platform_api_key}"
    headers["Content-Type"] = "application/json"

    try:
        with httpx.Client(timeout=FETCH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings/search",
                json=request_payload,
                headers=headers,
            )
            if resp.status_code != 200:
                return []

            raw = resp.json()
            if isinstance(raw, list):
                results = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict):
                wrapped = raw.get("results", raw.get("items", []))
                results = [item for item in wrapped if isinstance(item, dict)] if isinstance(wrapped, list) else []
            else:
                return []
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
    for r in deduped:
        summary = r.get("summary", r.get("content", ""))
        content = str(summary) if summary is not None else ""
        r["summary"] = f"[shared] {content}"
        r["content"] = f"[shared] {content}"
        r["source"] = "shared"
        shared.append(r)

    logger.debug(
        "memory_fetch_complete",
        fetched=len(results),
        after_dedup=len(shared),
    )
    return shared


def _dedupe_shared_results(
    results: list[dict[str, object]],
    *,
    local_entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    dedup_threshold: float,
) -> list[dict[str, object]]:
    """Suppress remote results that match already-present local memories.

    Exact ID/content checks are used as a safe fallback. When an embedder is
    available, we compare remote and local text embeddings so semantically
    duplicate shared entries do not get appended to the recall output.
    """
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

    if not remote_candidates or embedder is None or not embedder.available():
        return remote_candidates

    local_texts = [f"{entry.content} {entry.detail}".strip() for entry in local_entries]
    vectors = embedder.embed_batch([*local_texts, *remote_texts])
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
    return filtered
