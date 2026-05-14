# ruff: noqa: F401
"""Tests for trw_memory.sync.remote — PRD-CORE-047."""

from __future__ import annotations

import json
import socket
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.sync.remote import (
    FETCH_TIMEOUT,
    PUBLISH_TIMEOUT,
    _anonymize_entry,
    fetch_shared_memories,
    publish_memory,
    publish_memory_result,
    retire_remote_memory,
)

from ._test_sync_support import (
    make_sync_config as _make_config,
)
from ._test_sync_support import (
    make_sync_entry as _make_entry,
)
from ._test_sync_support import (
    mock_httpx_client as _mock_httpx_client,
)


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
            json_data=[{"summary": "Use caching for speed"}, {"summary": "Retry on 503 errors"}],
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
            json_data=[{"summary": "Use caching for speed"}, {"summary": "Existing local knowledge"}],
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
        _mock_httpx_client(mock_client_cls, status_code=200, json_data={"results": [{"summary": "A finding"}]})

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


class TestModuleConstants:
    """Verify module-level remote constants are set correctly."""

    def test_publish_timeout(self) -> None:
        assert PUBLISH_TIMEOUT == 5.0

    def test_fetch_timeout(self) -> None:
        assert FETCH_TIMEOUT == 3.0
