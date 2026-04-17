"""Tests for trw_memory.sync — remote sync protocol (PRD-CORE-047).

Covers:
- FR04: Vector clock operations (compare, increment, init, merge)
- FR05: Conflict resolution (causal order, concurrent merge)
- FR01: Publish pipeline (anonymization, HTTP, fail-open)
- FR07: Anonymization (_anonymize_entry)
- FR02: Fetch pipeline (shared results, dedup, fail-open)
- FR06: Retry queue (enqueue, drain, depth cap, clear)
- FR03: SSE subscriber (start, stop, process_line)
- FR08: Config fields (verified via MemoryConfig construction)
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.sync.conflict import (
    MAX_MERGED_DETAIL_LENGTH,
    compare_clocks,
    increment_clock,
    init_clock,
    merge_clocks,
    resolve_conflict,
)
from trw_memory.sync.remote import (
    FETCH_TIMEOUT,
    PUBLISH_TIMEOUT,
    _anonymize_entry,
    clear_retry_queue,
    drain_retry_queue,
    fetch_shared_memories,
    publish_memory,
    publish_memory_result,
    retire_remote_memory,
)
from trw_memory.sync.retry_queue import MAX_QUEUE_BYTES, MAX_QUEUE_DEPTH, MAX_RETRIES, RetryQueue
from trw_memory.sync.subscriber import RECONNECT_DELAY, SSESubscriber

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "M-test",
    content: str = "test content",
    detail: str = "test detail",
    importance: float = 0.8,
    tags: list[str] | None = None,
    vector_clock: dict[str, int] | None = None,
    merged_from: list[str] | None = None,
    outcome_history: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        importance=importance,
        tags=tags or [],
        vector_clock=vector_clock or {},
        merged_from=merged_from or [],
        outcome_history=outcome_history or [],
        metadata=metadata or {},
    )


def _make_config(
    sync_enabled: bool = True,
    platform_url: str = "https://api.example.com",
    platform_api_key: str = "test-key-123",
    sync_min_importance: float = 0.7,
    local_only: bool = False,
) -> MemoryConfig:
    """Create a MemoryConfig for testing."""
    return MemoryConfig(
        sync_enabled=sync_enabled,
        platform_url=platform_url,
        platform_api_key=platform_api_key,
        sync_min_importance=sync_min_importance,
        local_only=local_only,
    )


def _mock_httpx_client(
    mock_client_cls: MagicMock,
    *,
    status_code: int = 200,
    json_data: Any = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Wire up an httpx.Client context-manager mock.

    Returns the mock client so callers can inspect ``post.call_args``.
    """
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    if side_effect is not None:
        mock_client.post.side_effect = side_effect
        mock_client.patch.side_effect = side_effect
    else:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        if json_data is not None:
            mock_resp.json.return_value = json_data
        mock_client.post.return_value = mock_resp
        mock_client.patch.return_value = mock_resp

    mock_client_cls.return_value = mock_client
    return mock_client


def test_local_only_blocks_immediately() -> None:
    cfg = _make_config(local_only=True)
    entry = _make_entry()

    with patch.object(socket, "socket") as mock_socket:
        start = time.perf_counter()
        with pytest.raises(LocalOnlyViolationError, match="memory_local_only=True"):
            publish_memory(entry, cfg)
        elapsed = time.perf_counter() - start

    mock_socket.assert_not_called()
    assert elapsed < 0.005


# ===========================================================================
# FR04: Vector Clock Operations
# ===========================================================================


class TestCompareClocks:
    """FR04: compare_clocks returns correct causal ordering."""

    def test_a_wins_when_strictly_dominates(self) -> None:
        """a_wins when a >= b on all keys and > on at least one."""
        result = compare_clocks({"A": 3, "B": 1}, {"A": 2, "B": 1})
        assert result == "a_wins"

    def test_b_wins_when_strictly_dominates(self) -> None:
        """b_wins when b >= a on all keys and > on at least one."""
        result = compare_clocks({"A": 1, "B": 1}, {"A": 2, "B": 1})
        assert result == "b_wins"

    def test_concurrent_when_neither_dominates(self) -> None:
        """concurrent when a > b on some key and b > a on another."""
        result = compare_clocks({"A": 2, "B": 1}, {"A": 1, "B": 2})
        assert result == "concurrent"

    def test_concurrent_when_clocks_are_equal(self) -> None:
        """Equal clocks are concurrent (not a win for either)."""
        result = compare_clocks({"A": 3}, {"A": 3})
        assert result == "concurrent"

    def test_concurrent_when_both_empty(self) -> None:
        """Empty clocks are concurrent."""
        result = compare_clocks({}, {})
        assert result == "concurrent"

    def test_handles_missing_keys_default_zero(self) -> None:
        """Missing keys default to 0 in comparison."""
        # a has key B that b doesn't; b has key C that a doesn't
        result = compare_clocks({"A": 1, "B": 1}, {"A": 1, "C": 1})
        assert result == "concurrent"

    def test_a_wins_with_superset_keys(self) -> None:
        """a_wins when a has all keys of b plus extra with > values."""
        result = compare_clocks({"A": 2, "B": 1}, {"A": 1})
        assert result == "a_wins"

    def test_b_wins_with_superset_keys(self) -> None:
        """b_wins when b has all keys of a plus extra with > values."""
        result = compare_clocks({"A": 1}, {"A": 2, "B": 1})
        assert result == "b_wins"

    def test_single_node_a_wins(self) -> None:
        """Single node: a_wins when a[node] > b[node]."""
        result = compare_clocks({"X": 5}, {"X": 3})
        assert result == "a_wins"


class TestInitClock:
    """FR04: init_clock creates initial vector clock."""

    def test_creates_clock_with_counter_one(self) -> None:
        """New clock has the node_id with counter 1."""
        clock = init_clock("node-abc")
        assert clock == {"node-abc": 1}

    def test_does_not_modify_original(self) -> None:
        """Init clock returns a new dict each time."""
        c1 = init_clock("node-1")
        c2 = init_clock("node-1")
        assert c1 == c2
        assert c1 is not c2


class TestIncrementClock:
    """FR04: increment_clock increments the node's counter."""

    def test_increments_existing_counter(self) -> None:
        """Incrementing an existing node increases its counter by 1."""
        clock = increment_clock({"node-1": 3, "node-2": 1}, "node-1")
        assert clock == {"node-1": 4, "node-2": 1}

    def test_adds_new_node_to_existing_clock(self) -> None:
        """Incrementing a new node adds it with counter 1."""
        clock = increment_clock({"node-1": 3}, "node-2")
        assert clock == {"node-1": 3, "node-2": 1}

    def test_does_not_mutate_original(self) -> None:
        """Returns a new dict without modifying the original."""
        original = {"node-1": 3}
        result = increment_clock(original, "node-1")
        assert original == {"node-1": 3}
        assert result == {"node-1": 4}


class TestMergeClocks:
    """FR04: merge_clocks takes max of each counter."""

    def test_merges_by_taking_max(self) -> None:
        """Max of each node's counter across both clocks."""
        result = merge_clocks({"A": 3, "B": 1}, {"A": 1, "B": 2})
        assert result == {"A": 3, "B": 2}

    def test_merges_disjoint_keys(self) -> None:
        """Disjoint keys included with their values."""
        result = merge_clocks({"A": 1}, {"B": 2})
        assert result == {"A": 1, "B": 2}

    def test_merges_empty_clocks(self) -> None:
        """Merging empty clocks yields empty clock."""
        result = merge_clocks({}, {})
        assert result == {}

    def test_merge_with_one_empty(self) -> None:
        """Merging with an empty clock returns the non-empty clock's values."""
        result = merge_clocks({"A": 5}, {})
        assert result == {"A": 5}


# ===========================================================================
# FR05: Conflict Resolution
# ===========================================================================


class TestResolveConflict:
    """FR05: resolve_conflict handles causal order and concurrent merge."""

    def test_a_wins_returns_local(self) -> None:
        """When local clock dominates, local entry is returned unchanged."""
        local = _make_entry(
            entry_id="L-1",
            content="local content",
            vector_clock={"A": 3, "B": 1},
        )
        remote = _make_entry(
            entry_id="R-1",
            content="remote content",
            vector_clock={"A": 2, "B": 1},
        )
        result = resolve_conflict(local, remote)
        assert result.id == "L-1"
        assert result.content == "local content"

    def test_b_wins_returns_remote(self) -> None:
        """When remote clock dominates, remote entry is returned."""
        local = _make_entry(
            entry_id="L-1",
            content="local content",
            vector_clock={"A": 1, "B": 1},
        )
        remote = _make_entry(
            entry_id="R-1",
            content="remote content",
            vector_clock={"A": 2, "B": 1},
        )
        result = resolve_conflict(local, remote)
        assert result.id == "R-1"
        assert result.content == "remote content"

    def test_concurrent_merges_detail_with_separator(self) -> None:
        """Concurrent clocks: details concatenated with separator."""
        local = _make_entry(
            entry_id="L-1",
            detail="local detail",
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            entry_id="R-1",
            detail="remote detail",
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert "local detail" in result.detail
        assert "remote detail" in result.detail
        assert "\n\n---\n\n" in result.detail

    def test_concurrent_takes_max_importance(self) -> None:
        """Concurrent: importance = max(local, remote)."""
        local = _make_entry(
            importance=0.6,
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            importance=0.9,
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert result.importance == 0.9

    def test_concurrent_unions_tags_sorted(self) -> None:
        """Concurrent: tags = sorted union of both tag sets."""
        local = _make_entry(
            tags=["python", "testing"],
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            tags=["testing", "deployment"],
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert result.tags == ["deployment", "python", "testing"]

    def test_concurrent_merges_clocks(self) -> None:
        """Concurrent: vector_clock = max of each counter."""
        local = _make_entry(vector_clock={"A": 2, "B": 1})
        remote = _make_entry(vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.vector_clock == {"A": 2, "B": 2}

    def test_merged_detail_truncated_to_max_length(self) -> None:
        """Concurrent: merged detail truncated to MAX_MERGED_DETAIL_LENGTH."""
        long_local = "x" * 1500
        long_remote = "y" * 1500
        local = _make_entry(
            detail=long_local,
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            detail=long_remote,
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert len(result.detail) <= MAX_MERGED_DETAIL_LENGTH

    def test_adds_conflict_merged_to_outcome_history(self) -> None:
        """Concurrent merge adds a conflict_merged record to outcome_history."""
        local = _make_entry(
            entry_id="L-1",
            vector_clock={"A": 2, "B": 1},
            outcome_history=["existing-event"],
        )
        remote = _make_entry(
            entry_id="R-1",
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert len(result.outcome_history) >= 2
        conflict_entry = result.outcome_history[-1]
        assert "conflict_merged" in conflict_entry
        assert "L-1" in conflict_entry
        assert "R-1" in conflict_entry

    def test_concurrent_preserves_local_content(self) -> None:
        """Concurrent merge uses local content (preferred)."""
        local = _make_entry(
            content="local preferred",
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            content="remote content",
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert result.content == "local preferred"

    def test_concurrent_unions_merged_from(self) -> None:
        """Concurrent merge records the participating vector-clock nodes."""
        local = _make_entry(
            merged_from=["src-1"],
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            merged_from=["src-2"],
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert result.merged_from == ["A", "B"]

    def test_concurrent_same_detail_no_double(self) -> None:
        """When local and remote detail are equal, don't duplicate."""
        local = _make_entry(
            detail="same detail",
            vector_clock={"A": 2, "B": 1},
        )
        remote = _make_entry(
            detail="same detail",
            vector_clock={"A": 1, "B": 2},
        )
        result = resolve_conflict(local, remote)
        assert result.detail == "same detail"


# ===========================================================================
# FR07: Anonymization (_anonymize_entry)
# ===========================================================================


class TestAnonymizeEntry:
    """FR07: _anonymize_entry strips PII and redacts paths."""

    def test_strips_pii_from_content(self) -> None:
        """Email addresses are replaced with <email>."""
        entry = _make_entry(content="Contact user@example.com for info")
        result = _anonymize_entry(entry)
        assert "<email>" in result["summary"]
        assert "user@example.com" not in result["summary"]

    def test_strips_pii_from_detail(self) -> None:
        """Email addresses in detail are also stripped."""
        entry = _make_entry(detail="See admin@corp.com")
        result = _anonymize_entry(entry)
        assert result["detail"] is not None
        assert "<email>" in result["detail"]
        assert "admin@corp.com" not in result["detail"]

    def test_redacts_paths_in_content(self) -> None:
        """Project root path is replaced with <project>."""
        entry = _make_entry(content="Error in /home/user/project/src/foo.py")
        result = _anonymize_entry(entry, project_root="/home/user/project")
        assert "<project>/src/foo.py" in result["summary"]

    def test_truncates_content_to_1000_chars(self) -> None:
        """Summary field is truncated to 1000 characters."""
        entry = _make_entry(content="x" * 2000)
        result = _anonymize_entry(entry)
        assert len(result["summary"]) == 1000

    def test_truncates_detail_to_10000_chars(self) -> None:
        """Detail field is truncated to 10000 characters."""
        entry = _make_entry(detail="y" * 20000)
        result = _anonymize_entry(entry)
        assert result["detail"] is not None
        assert len(result["detail"]) == 10000

    def test_tags_limited_to_20(self) -> None:
        """Tags list is truncated to 20 items."""
        entry = _make_entry(tags=[f"tag-{i}" for i in range(30)])
        result = _anonymize_entry(entry)
        assert len(result["tags"]) == 20

    def test_impact_maps_from_importance(self) -> None:
        """The 'impact' field maps from MemoryEntry.importance."""
        entry = _make_entry(importance=0.85)
        result = _anonymize_entry(entry)
        assert result["impact"] == 0.85

    def test_source_project_anonymized(self) -> None:
        """source_project is a 16-char hex string from double-SHA-256."""
        entry = _make_entry(metadata={"installation_id": "my-machine-id"})
        result = _anonymize_entry(entry)
        assert len(result["source_project"]) == 16
        assert all(c in "0123456789abcdef" for c in result["source_project"])

    def test_embedding_defaults_to_none(self) -> None:
        """Default embedding is None (populated by caller)."""
        entry = _make_entry()
        result = _anonymize_entry(entry)
        assert result["embedding"] is None

    def test_empty_detail_returns_none(self) -> None:
        """Empty detail field returns None in payload."""
        entry = _make_entry(detail="")
        result = _anonymize_entry(entry)
        assert result["detail"] is None


# ===========================================================================
# FR01: Publish Pipeline
# ===========================================================================


class TestPublishMemory:
    """FR01: publish_memory publishes entries with fail-open behavior."""

    def test_returns_false_when_sync_disabled(self) -> None:
        """Disabled sync is treated as a non-retryable skip."""
        cfg = _make_config(sync_enabled=False)
        entry = _make_entry(importance=0.9)
        assert publish_memory(entry, cfg) is True

    def test_raises_when_local_only_enabled(self) -> None:
        """Local-only mode blocks remote publish entrypoints explicitly."""
        cfg = _make_config(local_only=True)
        entry = _make_entry(importance=0.9)
        with pytest.raises(LocalOnlyViolationError, match="memory_local_only=True"):
            publish_memory(entry, cfg)

    def test_returns_false_when_platform_url_empty(self) -> None:
        """Empty remote config is treated as a non-retryable skip."""
        cfg = _make_config(platform_url="")
        entry = _make_entry(importance=0.9)
        assert publish_memory(entry, cfg) is True

    def test_returns_true_when_platform_url_scheme_invalid(self) -> None:
        """Invalid URL schemes are rejected before any network call."""
        cfg = _make_config(platform_url="file:///etc/passwd")
        entry = _make_entry(importance=0.9)
        assert publish_memory(entry, cfg) is True

    def test_returns_true_when_importance_below_threshold(self) -> None:
        """Below-threshold entries are skipped instead of retried."""
        cfg = _make_config(sync_min_importance=0.7)
        entry = _make_entry(importance=0.5)
        assert publish_memory(entry, cfg) is True

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_true_on_200_response(self, mock_client_cls: MagicMock) -> None:
        """Returns True when the backend responds with 200."""
        _mock_httpx_client(mock_client_cls, status_code=200)

        cfg = _make_config()
        entry = _make_entry(importance=0.9)
        assert publish_memory(entry, cfg) is True

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_publish_result_returns_remote_id(self, mock_client_cls: MagicMock) -> None:
        """Successful publishes surface the backend-assigned remote ID."""
        _mock_httpx_client(mock_client_cls, status_code=200, json_data={"id": 123, "status": "published"})

        cfg = _make_config()
        entry = _make_entry(importance=0.9)
        result = publish_memory_result(entry, cfg)

        assert result == {"success": True, "remote_id": "123", "retryable": False}

    def test_publish_result_invalid_platform_url_is_non_retryable_skip(self) -> None:
        """Unsafe platform URLs should not be marked published or retried."""
        cfg = _make_config(platform_url="file:///etc/passwd")
        entry = _make_entry(importance=0.9)

        assert publish_memory_result(entry, cfg) == {"success": False, "remote_id": None, "retryable": False}

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_calls_correct_url_with_auth(self, mock_client_cls: MagicMock) -> None:
        """POST goes to {platform_url}/v1/learnings with Bearer token."""
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200)

        cfg = _make_config(platform_url="https://api.test.com")
        entry = _make_entry(importance=0.9)
        publish_memory(entry, cfg)

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.test.com/v1/learnings"
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key-123"

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_false_on_503_response(self, mock_client_cls: MagicMock) -> None:
        """Returns False on 503 (fail-open, queued for retry)."""
        _mock_httpx_client(mock_client_cls, status_code=503)

        cfg = _make_config()
        entry = _make_entry(importance=0.9)
        assert publish_memory(entry, cfg) is False

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_false_on_connection_error(self, mock_client_cls: MagicMock) -> None:
        """Returns False on connection error (fail-open, no exception raised)."""
        _mock_httpx_client(mock_client_cls, side_effect=ConnectionError("refused"))

        cfg = _make_config()
        entry = _make_entry(importance=0.9)
        # Must not raise — fail-open
        assert publish_memory(entry, cfg) is False

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_includes_embedding_in_payload(self, mock_client_cls: MagicMock) -> None:
        """When embedding is provided, it's included in the payload."""
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200)

        cfg = _make_config()
        entry = _make_entry(importance=0.9)
        embedding = [0.1] * 384
        publish_memory(entry, cfg, embedding=embedding)

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["embedding"] == embedding

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_no_auth_header_without_api_key(self, mock_client_cls: MagicMock) -> None:
        """When platform_api_key is empty, no Authorization header."""
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200)

        cfg = _make_config(platform_api_key="")
        entry = _make_entry(importance=0.9)
        publish_memory(entry, cfg)

        call_args = mock_client.post.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" not in headers


# ===========================================================================
# FR02: Fetch Pipeline
# ===========================================================================


class TestFetchSharedMemories:
    """FR02: fetch_shared_memories retrieves and deduplicates remote results."""

    def test_returns_empty_when_sync_disabled(self) -> None:
        """Returns [] when sync_enabled=False."""
        cfg = _make_config(sync_enabled=False)
        assert fetch_shared_memories("query", cfg) == []

    def test_raises_when_local_only_enabled(self) -> None:
        """Local-only mode blocks remote fetch entrypoints explicitly."""
        cfg = _make_config(local_only=True)
        with pytest.raises(LocalOnlyViolationError, match="memory_local_only=True"):
            fetch_shared_memories("query", cfg)

    def test_returns_empty_when_platform_url_empty(self) -> None:
        """Returns [] when platform_url is empty."""
        cfg = _make_config(platform_url="")
        assert fetch_shared_memories("query", cfg) == []

    def test_returns_empty_when_platform_url_scheme_invalid(self) -> None:
        """Invalid URL schemes are rejected before any fetch."""
        cfg = _make_config(platform_url="file:///etc/passwd")
        assert fetch_shared_memories("query", cfg) == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_empty_on_connection_error(self, mock_client_cls: MagicMock) -> None:
        """Returns [] on connection error (fail-open)."""
        _mock_httpx_client(mock_client_cls, side_effect=ConnectionError("refused"))

        cfg = _make_config()
        assert fetch_shared_memories("query", cfg) == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_empty_on_timeout(self, mock_client_cls: MagicMock) -> None:
        """Timeouts fall back to local-only results."""
        _mock_httpx_client(mock_client_cls, side_effect=httpx.ReadTimeout("timed out"))

        cfg = _make_config()
        assert fetch_shared_memories("query", cfg) == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_results_with_shared_prefix(self, mock_client_cls: MagicMock) -> None:
        """Remote results get [shared] prefix on content."""
        _mock_httpx_client(
            mock_client_cls,
            status_code=200,
            json_data=[
                {"summary": "Use caching for speed"},
                {"summary": "Retry on 503 errors"},
            ],
        )

        cfg = _make_config()
        results = fetch_shared_memories("query", cfg)
        assert len(results) == 2
        assert str(results[0]["content"]).startswith("[shared] ")
        assert str(results[0]["summary"]).startswith("[shared] ")
        assert str(results[1]["content"]).startswith("[shared] ")
        assert results[0]["source"] == "shared"

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_deduplicates_against_local_entries(self, mock_client_cls: MagicMock) -> None:
        """Remote results matching local content are excluded."""
        _mock_httpx_client(
            mock_client_cls,
            status_code=200,
            json_data=[
                {"summary": "Use caching for speed"},
                {"summary": "Existing local knowledge"},
            ],
        )

        cfg = _make_config()
        local = [_make_entry(content="Existing local knowledge")]
        results = fetch_shared_memories("query", cfg, local_entries=local)
        assert len(results) == 1
        assert "Use caching" in str(results[0]["content"])

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_respects_limit_parameter(self, mock_client_cls: MagicMock) -> None:
        """Limit parameter is sent in the POST payload."""
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200, json_data=[])

        cfg = _make_config()
        fetch_shared_memories("query", cfg, limit=5)

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["limit"] == 5

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_calls_correct_search_url(self, mock_client_cls: MagicMock) -> None:
        """POST goes to {platform_url}/v1/learnings/search."""
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200, json_data=[])

        cfg = _make_config(platform_url="https://api.test.com")
        fetch_shared_memories("test query", cfg)

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.test.com/v1/learnings/search"

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_empty_on_non_200(self, mock_client_cls: MagicMock) -> None:
        """Returns [] on non-200 status code."""
        _mock_httpx_client(mock_client_cls, status_code=500)

        cfg = _make_config()
        assert fetch_shared_memories("query", cfg) == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_handles_results_wrapper_dict(self, mock_client_cls: MagicMock) -> None:
        """Handles response wrapped in {results: [...]} dict."""
        _mock_httpx_client(
            mock_client_cls,
            status_code=200,
            json_data={"results": [{"summary": "A finding"}]},
        )

        cfg = _make_config()
        results = fetch_shared_memories("query", cfg)
        assert len(results) == 1
        assert results[0]["content"] == "[shared] A finding"

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_handles_items_wrapper_dict(self, mock_client_cls: MagicMock) -> None:
        """Handles backend SearchResponse payloads wrapped in {items: [...]}."""
        _mock_httpx_client(
            mock_client_cls,
            status_code=200,
            json_data={"items": [{"summary": "A finding from items"}]},
        )

        cfg = _make_config()
        results = fetch_shared_memories("query", cfg)
        assert len(results) == 1
        assert results[0]["content"] == "[shared] A finding from items"

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_deduplicates_by_embedding_similarity(self, mock_client_cls: MagicMock) -> None:
        """Semantically duplicate remote entries are suppressed when embeddings match."""
        _mock_httpx_client(
            mock_client_cls,
            status_code=200,
            json_data=[{"summary": "Remote deployment guidance", "detail": "extra"}],
        )

        cfg = _make_config()
        local = [_make_entry(content="Local deployment advice", detail="detail")]
        embedder = MagicMock()
        embedder.available.return_value = True
        embedder.embed_batch.return_value = [[1.0, 0.0], [0.99, 0.01]]

        results = fetch_shared_memories("query", cfg, local_entries=local, embedder=embedder)
        assert results == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_empty_on_invalid_json(self, mock_client_cls: MagicMock) -> None:
        """Malformed JSON responses fail open to local-only behavior."""
        mock_client = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_response.json.side_effect = json.JSONDecodeError("bad", "x", 0)
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client_cls.return_value = mock_client

        cfg = _make_config()
        assert fetch_shared_memories("query", cfg) == []

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_returns_empty_on_unexpected_json_shape(self, mock_client_cls: MagicMock) -> None:
        """Unexpected JSON shapes are ignored instead of crashing recall."""
        _mock_httpx_client(mock_client_cls, status_code=200, json_data="not-a-result-list")

        cfg = _make_config()
        assert fetch_shared_memories("query", cfg) == []


class TestRetireRemoteMemory:
    """FR05: local delete propagation uses the backend status endpoint."""

    @patch("trw_memory.sync.remote.httpx.Client")
    def test_retire_marks_remote_entry_obsolete(self, mock_client_cls: MagicMock) -> None:
        mock_client = _mock_httpx_client(mock_client_cls, status_code=200)

        assert retire_remote_memory("42", _make_config()) is True

        call_args = mock_client.patch.call_args
        assert call_args[0][0] == "https://api.example.com/v1/learnings/42/status"
        assert call_args[1]["json"] == {"status": "obsolete"}
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key-123"

    def test_retire_skips_invalid_platform_url(self) -> None:
        assert retire_remote_memory("42", _make_config(platform_url="file:///etc/passwd")) is True

    def test_retire_raises_when_local_only_enabled(self) -> None:
        with pytest.raises(LocalOnlyViolationError, match="memory_local_only=True"):
            retire_remote_memory("42", _make_config(local_only=True))


# ===========================================================================
# FR06: Retry Queue
# ===========================================================================


class TestRetryQueue:
    """FR06: RetryQueue provides JSONL persistence with depth cap."""

    def test_enqueue_appends_to_file(self, tmp_path: Path) -> None:
        """Enqueue writes a JSONL record to the queue file."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        assert queue.enqueue("M-001", {"summary": "test"})
        assert queue.depth() == 1

    def test_enqueue_returns_false_at_max_depth(self, tmp_path: Path) -> None:
        """Returns False when queue is at MAX_QUEUE_DEPTH capacity."""
        queue_path = tmp_path / "queue.jsonl"
        queue = RetryQueue(queue_path)
        # Fill queue to capacity
        for i in range(MAX_QUEUE_DEPTH):
            queue.enqueue(f"M-{i}", {"summary": f"entry-{i}"})
        # Next enqueue should fail
        assert not queue.enqueue("M-overflow", {"summary": "overflow"})
        assert queue.depth() == MAX_QUEUE_DEPTH

    def test_enqueue_returns_false_when_size_limit_exceeded(self, tmp_path: Path) -> None:
        """Queue enforces the 5MB aggregate size cap before writing a new record."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        oversized_detail = "x" * ((MAX_QUEUE_BYTES // 2) + 1024)

        assert queue.enqueue("M-001", {"summary": "test", "detail": oversized_detail})
        assert not queue.enqueue("M-002", {"summary": "test", "detail": oversized_detail})

    def test_drain_publishes_and_removes_successful(self, tmp_path: Path) -> None:
        """Drain removes entries that publish successfully."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test-1"})
        queue.enqueue("M-002", {"summary": "test-2"})

        result = queue.drain(lambda _: True)
        assert result == {"drained": 2, "failed": 0, "skipped": 0}
        assert queue.depth() == 0

    def test_drain_increments_retry_count_on_failure(self, tmp_path: Path) -> None:
        """Drain increments retry_count when publish returns False."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = queue.drain(lambda _: False)
        assert result["failed"] == 1
        assert queue.depth() == 1

        # Read the file to verify retry_count was incremented
        lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["retry_count"] == 1
        assert record["last_error"] == "publish returned False"

    def test_drain_applies_exponential_backoff_after_failures(self, tmp_path: Path) -> None:
        """Retries after prior failures sleep using the documented backoff curve."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            json.dumps(
                {
                    "entry_id": "M-001",
                    "payload": {"summary": "test"},
                    "queued_at": "2026-01-01T00:00:00Z",
                    "retry_count": 3,
                    "last_error": "previous failure",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)

        with patch("trw_memory.sync.retry_queue.time.sleep") as sleep_mock:
            result = queue.drain(lambda _: False)

        assert result["failed"] == 1
        sleep_mock.assert_called_once_with(4.0)

    def test_drain_skips_entries_at_max_retries(self, tmp_path: Path) -> None:
        """Entries with retry_count >= MAX_RETRIES are skipped."""
        queue_path = tmp_path / "queue.jsonl"
        # Write a record with max retries directly
        record = {
            "entry_id": "M-001",
            "payload": {"summary": "exhausted"},
            "queued_at": "2026-01-01T00:00:00Z",
            "retry_count": MAX_RETRIES,
            "last_error": "previous error",
        }
        queue_path.write_text(json.dumps(record) + "\n")

        queue = RetryQueue(queue_path)
        result = queue.drain(lambda _: True)
        assert result == {"drained": 0, "failed": 0, "skipped": 1}
        assert queue.depth() == 1  # Still in queue, just skipped

    def test_drain_handles_publish_exception(self, tmp_path: Path) -> None:
        """Drain catches exceptions from publish_fn and increments retry_count."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        def failing_publish(_payload: Any) -> bool:
            raise ConnectionError("network down")

        result = queue.drain(failing_publish)
        assert result["failed"] == 1
        assert queue.depth() == 1

        lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["retry_count"] == 1
        assert "network down" in record["last_error"]

    def test_clear_empties_queue(self, tmp_path: Path) -> None:
        """Clear truncates the queue file to empty."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})
        assert queue.depth() == 1

        queue.clear()
        assert queue.depth() == 0

    def test_depth_returns_correct_count(self, tmp_path: Path) -> None:
        """Depth returns the number of entries in the queue."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        assert queue.depth() == 0
        queue.enqueue("M-001", {"summary": "first"})
        assert queue.depth() == 1
        queue.enqueue("M-002", {"summary": "second"})
        assert queue.depth() == 2

    def test_depth_on_nonexistent_file(self, tmp_path: Path) -> None:
        """Depth returns 0 when the queue file doesn't exist."""
        queue = RetryQueue(tmp_path / "nonexistent.jsonl")
        assert queue.depth() == 0

    def test_enqueue_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Enqueue creates parent directories if needed."""
        queue_path = tmp_path / "subdir" / "queue.jsonl"
        queue = RetryQueue(queue_path)
        assert queue.enqueue("M-001", {"summary": "test"})
        assert queue_path.exists()

    def test_corrupt_jsonl_lines_are_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSONL lines should be logged and skipped, not crash dequeue."""
        queue_path = tmp_path / "queue.jsonl"
        # Write a mix of valid and corrupt lines
        queue_path.write_text(
            '{"entry_id": "M-001", "data": {"summary": "valid"}, "retries": 0}\n'
            "not valid json at all\n"
            '{"entry_id": "M-002", "data": {"summary": "also valid"}, "retries": 0}\n',
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        # Depth should count only valid records
        assert queue.depth() == 2

    def test_empty_lines_in_queue_are_harmless(self, tmp_path: Path) -> None:
        """Empty lines in the JSONL file should not cause errors."""
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            '{"entry_id": "M-001", "data": {"summary": "test"}, "retries": 0}\n\n\n',
            encoding="utf-8",
        )
        queue = RetryQueue(queue_path)
        assert queue.depth() == 1

    def test_drain_retry_queue_republishes_payloads(self, tmp_path: Path) -> None:
        """The remote helper drains queued payloads through the publish transport."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test", "source_learning_id": "M-001"})

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            _mock_httpx_client(mock_client_cls, status_code=200, json_data={"id": "42"})
            result = drain_retry_queue(queue, _make_config())

        assert result == {"drained": 1, "failed": 0, "skipped": 0, "remote_ids": {"M-001": "42"}}
        assert queue.depth() == 0

    def test_drain_retry_queue_skips_when_sync_disabled(self, tmp_path: Path) -> None:
        """Drain helper leaves the queue intact when sync is disabled."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = drain_retry_queue(queue, _make_config(sync_enabled=False))

        assert result == {"drained": 0, "failed": 0, "skipped": 1, "remote_ids": {}}
        assert queue.depth() == 1

    def test_drain_retry_queue_skips_invalid_platform_url(self, tmp_path: Path) -> None:
        """Drain helper leaves queued payloads untouched for unsafe URL schemes."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        result = drain_retry_queue(queue, _make_config(platform_url="file:///etc/passwd"))

        assert result == {"drained": 0, "failed": 0, "skipped": 1, "remote_ids": {}}
        assert queue.depth() == 1

    def test_clear_retry_queue_helper_empties_file(self, tmp_path: Path) -> None:
        """The remote helper delegates to the queue clear operation."""
        queue = RetryQueue(tmp_path / "queue.jsonl")
        queue.enqueue("M-001", {"summary": "test"})

        clear_retry_queue(queue)

        assert queue.depth() == 0


# ===========================================================================
# FR03: SSE Subscriber
# ===========================================================================


class TestSSESubscriber:
    """FR03: SSESubscriber handles SSE connections in a daemon thread."""

    def test_start_does_nothing_when_sync_disabled(self) -> None:
        """No thread started when sync_enabled=False."""
        cfg = _make_config(sync_enabled=False)
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        sub.start()
        assert sub._thread is None

    def test_start_raises_when_local_only_enabled(self) -> None:
        """Local-only mode blocks the live SSE network subscriber."""
        cfg = _make_config(local_only=True)
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        with pytest.raises(LocalOnlyViolationError, match="memory_local_only=True"):
            sub.start()
        assert sub._thread is None

    def test_start_does_nothing_when_platform_url_empty(self) -> None:
        """No thread started when platform_url is empty."""
        cfg = _make_config(platform_url="")
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        sub.start()
        assert sub._thread is None

    def test_start_does_nothing_when_platform_url_invalid(self) -> None:
        """Invalid platform URLs are rejected before the SSE thread starts."""
        cfg = _make_config(platform_url="file:///etc/passwd")
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        sub.start()
        assert sub._thread is None

    def test_process_line_tracks_pending_event_id_until_learning_data(self) -> None:
        """id: lines become the reconnect cursor only after a learning event payload."""
        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        sub._process_line("id: evt-42")
        assert sub._last_event_id is None
        sub._process_line("event: learning_published")
        sub._process_line(f"data: {json.dumps({'type': 'learning_published', 'summary': 'new'})}")
        assert sub._last_event_id == "evt-42"

    def test_process_line_calls_on_event_for_learning_published(self) -> None:
        """data: lines with type=learning_published trigger the callback."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        data = json.dumps({"type": "learning_published", "summary": "new"})
        sub._process_line(f"data: {data}")
        assert len(received) == 1
        assert received[0]["type"] == "learning_published"

    def test_process_line_calls_on_event_for_learning_retired(self) -> None:
        """Retirement events flow through the real SSE parser."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        data = json.dumps({"type": "learning_retired", "id": 42})
        sub._process_line(f"data: {data}")
        assert len(received) == 1
        assert received[0]["type"] == "learning_retired"

    def test_process_line_calls_on_event_for_learning_updated(self) -> None:
        """Update events flow through the real SSE parser."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        data = json.dumps({"type": "learning_updated", "id": 42})
        sub._process_line(f"data: {data}")
        assert len(received) == 1
        assert received[0]["type"] == "learning_updated"

    def test_process_line_ignores_non_learning_events(self) -> None:
        """data: lines with other event types are ignored."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        data = json.dumps({"type": "heartbeat"})
        sub._process_line(f"data: {data}")
        assert len(received) == 0

    def test_process_line_does_not_advance_last_event_id_for_heartbeat(self) -> None:
        """Heartbeat IDs must not become the replay cursor for reconnect."""
        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        sub._process_line("id: evt-learning")
        sub._process_line("event: learning_published")
        sub._process_line(f"data: {json.dumps({'type': 'learning_published', 'summary': 'new'})}")
        sub._process_line("id: evt-heartbeat")
        sub._process_line("event: heartbeat")
        sub._process_line(f"data: {json.dumps({'type': 'heartbeat'})}")

        assert sub._last_event_id == "evt-learning"

    def test_process_line_ignores_empty_data(self) -> None:
        """Empty data: lines are ignored."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        sub._process_line("data: ")
        assert len(received) == 0

    def test_process_line_ignores_invalid_json(self) -> None:
        """Non-JSON data: lines are silently ignored."""
        cfg = _make_config()
        received: list[dict[str, Any]] = []
        sub = SSESubscriber(cfg, on_event=lambda data: received.append(data))
        sub._process_line("data: {invalid json}")
        assert len(received) == 0

    def test_stop_sets_event(self) -> None:
        """Stop sets the internal stop event."""
        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda data: None)
        assert not sub._stop_event.is_set()
        sub.stop()
        assert sub._stop_event.is_set()


# ===========================================================================
# FR08: Config Fields
# ===========================================================================


class TestConfigFields:
    """FR08: MemoryConfig has sync-related fields."""

    @staticmethod
    def _isolate_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run default-value assertions outside any project-level `.trw/config.yaml`."""
        monkeypatch.chdir(tmp_path)
        for key in (
            "MEMORY_SYNC_ENABLED",
            "MEMORY_SYNC_MIN_IMPORTANCE",
            "MEMORY_SYNC_NAMESPACE",
            "MEMORY_PLATFORM_URL",
            "MEMORY_PLATFORM_API_KEY",
            "MEMORY_LOCAL_ONLY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_sync_enabled_defaults_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """sync_enabled defaults to False."""
        self._isolate_defaults(tmp_path, monkeypatch)
        cfg = MemoryConfig()
        assert cfg.sync_enabled is False

    def test_sync_min_importance_defaults_0_7(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """sync_min_importance defaults to 0.7."""
        self._isolate_defaults(tmp_path, monkeypatch)
        cfg = MemoryConfig()
        assert cfg.sync_min_importance == 0.7

    def test_sync_namespace_defaults_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """sync_namespace defaults to empty string."""
        self._isolate_defaults(tmp_path, monkeypatch)
        cfg = MemoryConfig()
        assert cfg.sync_namespace == ""

    def test_platform_url_defaults_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """platform_url defaults to empty string."""
        self._isolate_defaults(tmp_path, monkeypatch)
        cfg = MemoryConfig()
        assert cfg.platform_url == ""

    def test_platform_api_key_defaults_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """platform_api_key defaults to empty string."""
        self._isolate_defaults(tmp_path, monkeypatch)
        cfg = MemoryConfig()
        assert cfg.platform_api_key == ""

    def test_sync_min_importance_rejects_out_of_range(self) -> None:
        """sync_min_importance rejects values outside [0.0, 1.0]."""
        with pytest.raises(Exception):
            MemoryConfig(sync_min_importance=1.5)


# ===========================================================================
# Module constants
# ===========================================================================


class TestModuleConstants:
    """Verify module-level constants are set correctly."""

    def test_publish_timeout(self) -> None:
        assert PUBLISH_TIMEOUT == 5.0

    def test_fetch_timeout(self) -> None:
        assert FETCH_TIMEOUT == 3.0

    def test_max_queue_depth(self) -> None:
        assert MAX_QUEUE_DEPTH == 500

    def test_max_retries(self) -> None:
        assert MAX_RETRIES == 5

    def test_reconnect_delay(self) -> None:
        assert RECONNECT_DELAY == 5.0

    def test_max_merged_detail_length(self) -> None:
        assert MAX_MERGED_DETAIL_LENGTH == 2000


# ===========================================================================
# FR01 (PRD-QUAL-038): SSESubscriber Reconnection & Branch Coverage
# ===========================================================================


class TestSSESubscriberDaemonThread:
    """FR01: SSESubscriber daemon thread lifecycle and reconnection tests."""

    def test_subscriber_start_creates_daemon_thread(self) -> None:
        """start() when sync_enabled=True creates a daemon thread."""
        cfg = _make_config(sync_enabled=True, platform_url="https://api.example.com")
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        with patch.object(sub, "_listen_loop"):
            sub.start()
            assert sub._thread is not None
            assert sub._thread.daemon is True
            assert sub._thread.name == "sse-subscriber"
            sub.stop()

    def test_subscriber_stop_sets_event_and_joins(self) -> None:
        """stop() sets _stop_event and joins thread."""
        cfg = _make_config(sync_enabled=True)
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        # Create a mock thread that reports alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        sub._thread = mock_thread
        sub.stop()
        assert sub._stop_event.is_set()
        mock_thread.join.assert_called_once_with(timeout=2.0)

    def test_subscriber_stop_closes_active_stream(self) -> None:
        """stop() closes the live SSE response/client before joining."""
        cfg = _make_config(sync_enabled=True)
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        sub._thread = mock_thread
        mock_response = MagicMock()
        mock_client = MagicMock()
        sub._active_response = mock_response
        sub._active_client = mock_client

        sub.stop()

        mock_response.close.assert_called_once()
        mock_client.close.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=2.0)

    @patch("trw_memory.sync.subscriber.httpx.Client")
    def test_listen_loop_reconnect_on_http_error(self, mock_client_cls: MagicMock) -> None:
        """HTTPError during stream triggers reconnect (loop re-enters)."""
        import httpx

        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda _data: None)

        # First call raises HTTPError, second sets stop event to exit
        call_count = 0

        def side_effect(*_args: Any, **_kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("server error")
            # On second call, set stop to exit loop
            sub._stop_event.set()
            raise httpx.HTTPError("stop")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = side_effect
        mock_client_cls.return_value = mock_client

        sub._listen_loop()
        # Should have attempted at least 2 connections (reconnect)
        assert call_count >= 2

    @patch("trw_memory.sync.subscriber.httpx.Client")
    def test_listen_loop_reconnect_on_os_error(self, mock_client_cls: MagicMock) -> None:
        """OSError during stream triggers reconnect."""
        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda _data: None)

        call_count = 0

        def side_effect(*_args: Any, **_kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("connection reset")
            sub._stop_event.set()
            raise OSError("stop")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = side_effect
        mock_client_cls.return_value = mock_client

        sub._listen_loop()
        assert call_count >= 2

    def test_listen_loop_stop_event_exits(self) -> None:
        """Set _stop_event before calling _listen_loop -> loop exits immediately."""
        cfg = _make_config()
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        sub._stop_event.set()
        # Should return immediately without error
        sub._listen_loop()
        assert sub._stop_event.is_set()

    @patch("trw_memory.sync.subscriber.httpx.Client")
    def test_subscriber_last_event_id_sent_on_reconnect(self, mock_client_cls: MagicMock) -> None:
        """After setting _last_event_id, reconnect sends it in headers."""
        cfg = _make_config(platform_api_key="test-key")
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        sub._last_event_id = "evt-99"

        captured_headers: list[dict[str, str]] = []

        def stream_side_effect(_method: str, _url: str, headers: dict[str, str] | None = None, **_kw: Any) -> MagicMock:
            if headers:
                captured_headers.append(dict(headers))
            sub._stop_event.set()
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.iter_lines.return_value = iter([])
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = stream_side_effect
        mock_client_cls.return_value = mock_client

        sub._listen_loop()

        assert len(captured_headers) >= 1
        assert captured_headers[0].get("Last-Event-ID") == "evt-99"
        assert "Authorization" in captured_headers[0]
