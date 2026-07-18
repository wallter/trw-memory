"""Remote publish helpers for memory sync."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import httpx
import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import anonymize_installation_id, redact_paths, strip_pii
from trw_memory.sync._remote_common import (
    MAX_DETAIL_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_TAGS_COUNT,
    PUBLISH_TIMEOUT,
    AnonymizedEntry,
    PublishResult,
    RetryDrainResult,
    SnapshotHashPayload,
    _raise_local_only_violation,
    build_platform_headers,
    encode_learning_api_v1,
    is_valid_platform_url,
)
from trw_memory.sync.retry_queue import RetryQueue

logger = structlog.get_logger(__name__)


def _anonymize_entry(entry: MemoryEntry, project_root: str = "") -> AnonymizedEntry:
    content = redact_paths(strip_pii(entry.content), project_root)
    detail = redact_paths(strip_pii(entry.detail), project_root)
    # Canonical ``importance`` -> external wire vocabulary via the sole
    # learning_api_v1 boundary encoder (PRD-CORE-181-FR06).
    return encode_learning_api_v1(
        summary=content[:MAX_SUMMARY_LENGTH],
        detail=detail[:MAX_DETAIL_LENGTH] if detail else None,
        tags=entry.tags[:MAX_TAGS_COUNT],
        importance=entry.importance,
        embedding=None,
        source_project=anonymize_installation_id(entry.metadata.get("installation_id", "")),
        source_learning_id=entry.id,
    )


def _extract_remote_id(response: httpx.Response) -> str | None:
    try:
        raw_body = response.json()
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if isinstance(raw_body, dict):
        raw_remote_id = raw_body.get("id")
        if raw_remote_id is not None:
            return str(raw_remote_id)
    return None


def _publish_payload_result(
    payload: dict[str, object],
    cfg: MemoryConfig,
    *,
    entry_id: str = "",
) -> PublishResult:
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_publish_invalid_platform_url", entry_id=entry_id)
        return {"success": False, "remote_id": None, "retryable": False}

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings",
                json=payload,
                headers=build_platform_headers(cfg.platform_api_key),
            )
            if 200 <= resp.status_code < 300:
                remote_id = _extract_remote_id(resp)
                logger.debug("memory_published", entry_id=entry_id, remote_id=remote_id)
                return {"success": True, "remote_id": remote_id, "retryable": False}
            logger.warning("memory_publish_failed", entry_id=entry_id, status=resp.status_code)
            return {"success": False, "remote_id": None, "retryable": True}
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_publish_error", entry_id=entry_id, exc_info=True)
        return {"success": False, "remote_id": None, "retryable": True}


def publish_memory_result(
    entry: MemoryEntry,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    project_root: str = "",
) -> PublishResult:
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
    result = publish_memory_result(entry, cfg, embedding=embedding, project_root=project_root)
    return result["success"] or not result["retryable"]


def drain_retry_queue(queue: RetryQueue, cfg: MemoryConfig) -> RetryDrainResult:
    result, _ = _drain_retry_queue_with_ids(queue, cfg)
    return result


def _drain_retry_queue_with_ids(
    queue: RetryQueue,
    cfg: MemoryConfig,
) -> tuple[RetryDrainResult, list[str]]:
    if cfg.local_only:
        logger.warning("memory_retry_drain_blocked_local_only")
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url:
        return {
            "drained": 0,
            "failed": 0,
            "skipped": queue.depth(),
            "remote_ids": {},
        }, []
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_retry_drain_invalid_platform_url")
        return {
            "drained": 0,
            "failed": 0,
            "skipped": queue.depth(),
            "remote_ids": {},
        }, []

    published_remote_ids: list[str | None] = []

    def publish_payload(payload: dict[str, object]) -> bool:
        source_learning_id = payload.get("source_learning_id")
        entry_id = str(source_learning_id) if isinstance(source_learning_id, str) else ""
        result = _publish_payload_result(payload, cfg, entry_id=entry_id)
        if result["success"]:
            published_remote_ids.append(result["remote_id"])
        return result["success"]

    drain_result, published_entry_ids = queue._drain_with_ids(publish_payload)
    remote_ids = {
        entry_id: remote_id
        for entry_id, remote_id in zip(published_entry_ids, published_remote_ids, strict=True)
        if remote_id is not None
    }
    return {
        "drained": drain_result["drained"],
        "failed": drain_result["failed"],
        "skipped": drain_result["skipped"],
        "remote_ids": remote_ids,
    }, published_entry_ids


def clear_retry_queue(queue: RetryQueue) -> None:
    queue.clear()


def publish_snapshot_hash(
    snapshot_path: Path,
    cfg: MemoryConfig,
    *,
    installation_id: str = "",
) -> PublishResult:
    if cfg.local_only:
        logger.warning("snapshot_hash_publish_blocked_local_only", snapshot=str(snapshot_path))
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.memory_snapshot_publish_hash:
        return {"success": False, "remote_id": None, "retryable": False}
    if not cfg.platform_url:
        return {"success": False, "remote_id": None, "retryable": False}
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("snapshot_hash_publish_invalid_platform_url", snapshot=str(snapshot_path))
        return {"success": False, "remote_id": None, "retryable": False}
    if not snapshot_path.exists() or not snapshot_path.is_file():
        logger.debug("snapshot_hash_publish_missing_file", snapshot=str(snapshot_path))
        return {"success": False, "remote_id": None, "retryable": False}

    try:
        digest, size_bytes = _hash_snapshot_file(snapshot_path)
    except OSError as exc:
        logger.debug("snapshot_hash_publish_read_failed", snapshot=str(snapshot_path), error=str(exc))
        return {"success": False, "remote_id": None, "retryable": True}

    payload: SnapshotHashPayload = {
        "digest": digest,
        "size_bytes": size_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "installation_id": anonymize_installation_id(installation_id or ""),
    }

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/memory/snapshot-hash",
                json=cast("dict[str, object]", payload),
                headers=build_platform_headers(cfg.platform_api_key),
            )
            if 200 <= resp.status_code < 300:
                logger.debug("snapshot_hash_published", digest=digest, size_bytes=size_bytes)
                return {"success": True, "remote_id": None, "retryable": False}
            logger.warning("snapshot_hash_publish_failed", status=resp.status_code, digest=digest)
            return {"success": False, "remote_id": None, "retryable": True}
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("snapshot_hash_publish_error", exc_info=True)
        return {"success": False, "remote_id": None, "retryable": True}


def _hash_snapshot_file(path: Path, chunk_size: int = 65536) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def retire_remote_memory(remote_id: str, cfg: MemoryConfig) -> bool:
    if cfg.local_only:
        logger.warning("memory_retire_blocked_local_only", remote_id=remote_id)
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url or not remote_id:
        return True
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_retire_invalid_platform_url", remote_id=remote_id)
        return True

    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.patch(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings/{remote_id}/status",
                json={"status": "obsolete"},
                headers=build_platform_headers(cfg.platform_api_key),
            )
            if 200 <= resp.status_code < 300:
                logger.debug("memory_retired_remote", remote_id=remote_id)
                return True
            logger.warning("memory_retire_failed", remote_id=remote_id, status=resp.status_code)
            return False
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_retire_error", remote_id=remote_id, exc_info=True)
        return False
