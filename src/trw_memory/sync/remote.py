"""Remote sync -- publish and fetch memory entries to/from the platform.

Implements FR01 (publish pipeline), FR02 (fetch pipeline), and FR07
(anonymization) from PRD-CORE-047.  All network operations are fail-open:
they never raise exceptions to the caller.
"""

from __future__ import annotations

from typing import TypedDict

import httpx
import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import anonymize_installation_id, redact_paths, strip_pii

logger = structlog.get_logger(__name__)

PUBLISH_TIMEOUT = 5.0  # seconds
FETCH_TIMEOUT = 3.0

MAX_SUMMARY_LENGTH = 1000
MAX_DETAIL_LENGTH = 10_000
MAX_TAGS_COUNT = 20


class AnonymizedEntry(TypedDict):
    """Typed structure for an anonymized memory entry ready for remote publish."""

    summary: str
    detail: str | None
    tags: list[str]
    impact: float
    embedding: list[float] | None
    source_project: str


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
    )


def publish_memory(
    entry: MemoryEntry,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    project_root: str = "",
) -> bool:
    """Publish a memory entry to the remote platform (FR01).

    Returns ``True`` on success, ``False`` on failure (entry should be queued
    for retry).  Fail-open: never raises exceptions.
    """
    if not cfg.sync_enabled or not cfg.platform_url:
        return False

    if entry.importance < cfg.sync_min_importance:
        return False

    payload = _anonymize_entry(entry, project_root)
    if embedding:
        payload["embedding"] = embedding

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
                logger.debug("memory_published", entry_id=entry.id)
                return True
            logger.warning(
                "memory_publish_failed",
                entry_id=entry.id,
                status=resp.status_code,
            )
            return False
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_publish_error", entry_id=entry.id, exc_info=True)
        return False


def fetch_shared_memories(
    query: str,
    cfg: MemoryConfig,
    *,
    embedding: list[float] | None = None,
    limit: int = 10,
    local_entries: list[MemoryEntry] | None = None,
    dedup_threshold: float = 0.92,
) -> list[dict[str, object]]:
    """Fetch shared memories from the remote platform (FR02).

    Returns a list of remote memory dicts with ``[shared]`` prefix on content.
    Deduplicates against *local_entries* by content string match.
    Fail-open: returns empty list on any error.
    """
    if not cfg.sync_enabled or not cfg.platform_url:
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
            results: list[dict[str, object]] = raw if isinstance(raw, list) else raw.get("results", [])
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_fetch_error", exc_info=True)
        return []

    # Dedup against local entries by content string match
    local_contents: set[str] = set()
    if local_entries:
        local_contents = {e.content.lower().strip() for e in local_entries}

    shared: list[dict[str, object]] = []
    for r in results:
        summary = r.get("summary", r.get("content", ""))
        content = str(summary) if summary is not None else ""
        if content.lower().strip() in local_contents:
            continue
        r["content"] = f"[shared] {content}"
        r["source"] = "shared"
        shared.append(r)

    logger.debug(
        "memory_fetch_complete",
        fetched=len(results),
        after_dedup=len(shared),
    )
    return shared
