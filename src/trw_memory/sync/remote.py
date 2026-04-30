"""Remote sync -- compatibility surface for publish and fetch helpers."""

from __future__ import annotations

import httpx

from trw_memory.sync._remote_common import (
    AnonymizedEntry,
    FETCH_TIMEOUT,
    LOCAL_ONLY_ERROR_MESSAGE,
    MAX_DETAIL_LENGTH,
    MAX_TAGS_COUNT,
    MAX_SUMMARY_LENGTH,
    PUBLISH_TIMEOUT,
    PublishResult,
    RetryDrainResult,
    SnapshotHashPayload,
    _raise_local_only_violation,
    is_valid_platform_url,
)
from trw_memory.sync._remote_fetch import fetch_shared_memories
from trw_memory.sync._remote_publish import (
    _anonymize_entry,
    clear_retry_queue,
    drain_retry_queue,
    publish_memory,
    publish_memory_result,
    publish_snapshot_hash,
    retire_remote_memory,
)

__all__ = [
    "AnonymizedEntry",
    "FETCH_TIMEOUT",
    "LOCAL_ONLY_ERROR_MESSAGE",
    "MAX_DETAIL_LENGTH",
    "MAX_TAGS_COUNT",
    "MAX_SUMMARY_LENGTH",
    "PUBLISH_TIMEOUT",
    "PublishResult",
    "RetryDrainResult",
    "SnapshotHashPayload",
    "_anonymize_entry",
    "_raise_local_only_violation",
    "clear_retry_queue",
    "drain_retry_queue",
    "fetch_shared_memories",
    "is_valid_platform_url",
    "publish_memory",
    "publish_memory_result",
    "publish_snapshot_hash",
    "retire_remote_memory",
]
