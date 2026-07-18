"""Shared helpers and typed contracts for remote sync."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from typing_extensions import TypedDict

from trw_memory.exceptions import LocalOnlyViolationError

PUBLISH_TIMEOUT = 5.0
FETCH_TIMEOUT = 3.0

MAX_SUMMARY_LENGTH = 1000
MAX_DETAIL_LENGTH = 10_000
MAX_TAGS_COUNT = 20
LOCAL_ONLY_ERROR_MESSAGE = "Operation blocked: memory_local_only=True disables all network access."


class AnonymizedEntry(TypedDict):
    summary: str
    detail: str | None
    tags: list[str]
    impact: float
    embedding: list[float] | None
    source_project: str
    source_learning_id: str


# ---------------------------------------------------------------------------
# learning_api_v1 protocol boundary (PRD-CORE-181-FR06)
# ---------------------------------------------------------------------------
#
# The first-party learning API speaks the legacy ``impact`` / ``min_impact``
# vocabulary on the wire, but every local storage / lifecycle reader is
# canonical ``importance`` after the memory_model_v2 cutover. This module is the
# SOLE, explicitly versioned translation boundary: publish/fetch call through
# these encoders/decoders and never reference ``impact`` directly, so a source
# census can prove the external vocabulary is contained here.

LEARNING_API_VERSION = "learning_api_v1"


def encode_learning_api_v1(
    *,
    summary: str,
    detail: str | None,
    tags: list[str],
    importance: float,
    embedding: list[float] | None,
    source_project: str,
    source_learning_id: str,
) -> AnonymizedEntry:
    """Encode canonical fields into the external ``learning_api_v1`` publish payload.

    Maps canonical ``importance`` onto the external ``impact`` wire field. This
    is the only place the outbound ``impact`` vocabulary is produced.
    """
    return AnonymizedEntry(
        summary=summary,
        detail=detail,
        tags=tags,
        impact=importance,
        embedding=embedding,
        source_project=source_project,
        source_learning_id=source_learning_id,
    )


def encode_learning_api_v1_search(
    *,
    query: str,
    limit: int,
    min_importance: float,
) -> dict[str, object]:
    """Encode a search request, mapping ``min_importance`` -> external ``min_impact``."""
    return {"query": query, "limit": limit, "min_impact": min_importance}


def decode_learning_api_v1_result(result: dict[str, object]) -> dict[str, object]:
    """Decode an external ``learning_api_v1`` result into canonical vocabulary.

    Maps the external ``impact`` field back onto canonical ``importance`` so
    downstream local readers never see the wire vocabulary. Results without an
    ``impact`` field pass through unchanged.
    """
    if "impact" not in result:
        return result
    decoded = dict(result)
    decoded["importance"] = decoded.pop("impact")
    return decoded


class PublishResult(TypedDict):
    success: bool
    remote_id: str | None
    retryable: bool


class RetryDrainResult(TypedDict):
    drained: int
    failed: int
    skipped: int
    remote_ids: dict[str, str]


class SnapshotHashPayload(TypedDict):
    digest: str
    size_bytes: int
    created_at: str
    installation_id: str


def _raise_local_only_violation() -> None:
    raise LocalOnlyViolationError(LOCAL_ONLY_ERROR_MESSAGE)


def is_valid_platform_url(platform_url: str) -> bool:
    if not platform_url.strip():
        return False
    parsed = urlparse(platform_url)
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and os.getenv("TRW_DEBUG", "").lower() == "true"


def build_platform_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
