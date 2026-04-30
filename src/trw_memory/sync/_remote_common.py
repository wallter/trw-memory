"""Shared helpers and typed contracts for remote sync."""

from __future__ import annotations

import os
from typing import TypedDict
from urllib.parse import urlparse

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
