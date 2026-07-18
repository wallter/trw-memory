"""Wave 15: coverage gap-fill for sync/_remote_publish.py.

Target lines: 52-53, 68-69, 90, 129-149, 158, 167-208, 212-218, 226, 241-245.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig
from trw_memory.sync._remote_publish import (
    _extract_remote_id,
    _hash_snapshot_file,
    _publish_payload_result,
    clear_retry_queue,
    drain_retry_queue,
    publish_snapshot_hash,
    retire_remote_memory,
)


def _cfg_sync_enabled(url: str = "https://platform.example.com") -> MemoryConfig:
    cfg = MemoryConfig()
    cfg.sync_enabled = True
    cfg.platform_url = url
    cfg.platform_api_key = "test-key"
    cfg.local_only = False
    return cfg


def _mock_response(status: int = 200, json_body: object = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("no body")
    return resp


# ---------------------------------------------------------------------------
# _extract_remote_id
# ---------------------------------------------------------------------------


class TestExtractRemoteId:
    def test_json_decode_error_returns_none(self) -> None:
        """Response.json() raises ValueError → return None (lines 52-53)."""
        resp = MagicMock(spec=httpx.Response)
        resp.json.side_effect = ValueError("invalid json")
        result = _extract_remote_id(resp)
        assert result is None


# ---------------------------------------------------------------------------
# _publish_payload_result
# ---------------------------------------------------------------------------


class TestPublishPayload:
    def test_invalid_platform_url_returns_failure(self) -> None:
        """Invalid platform_url → warning + return failure (lines 68-69)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = True
        cfg.platform_url = "not-a-valid-url"
        cfg.local_only = False
        result = _publish_payload_result({}, cfg)
        assert result == {"success": False, "remote_id": None, "retryable": False}

    def test_publish_payload_result_includes_remote_id(self) -> None:
        cfg = _cfg_sync_enabled()
        mock_resp = _mock_response(status=201, json_body={"id": "R-001"})
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = _publish_payload_result({"source_learning_id": "M-001"}, cfg, entry_id="M-001")
        assert result == {"success": True, "remote_id": "R-001", "retryable": False}


# ---------------------------------------------------------------------------
# drain_retry_queue
# ---------------------------------------------------------------------------


class TestDrainRetryQueue:
    def _mock_queue(self, depth: int = 0) -> MagicMock:
        q = MagicMock()
        q.depth.return_value = depth
        q.drain.return_value = {"drained": 1, "failed": 0, "skipped": 0}
        return q

    def test_sync_disabled_returns_skipped(self) -> None:
        """sync_enabled=False → return all skipped (lines 132-133)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = False
        cfg.platform_url = ""
        cfg.local_only = False
        q = self._mock_queue(depth=3)
        result = drain_retry_queue(q, cfg)
        assert result["skipped"] == 3
        assert result["drained"] == 0

    def test_invalid_url_returns_skipped(self) -> None:
        """Invalid platform_url → return all skipped (lines 134-136)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = True
        cfg.platform_url = "bad-url"
        cfg.local_only = False
        q = self._mock_queue(depth=2)
        result = drain_retry_queue(q, cfg)
        assert result["skipped"] == 2

    def test_successful_drain_maps_remote_ids(self) -> None:
        """drain_retry_queue with successful publish → remote_ids mapped (lines 138-154)."""
        cfg = _cfg_sync_enabled()
        q = self._mock_queue()

        def fake_drain(fn: object) -> tuple[dict[str, int], list[str]]:
            assert callable(fn)
            fn({"source_learning_id": "M-payload"})  # type: ignore[operator]
            return {"drained": 1, "failed": 0, "skipped": 0}, ["M-1"]

        q._drain_with_ids.side_effect = fake_drain
        mock_resp = _mock_response(status=201, json_body={"id": "REMOTE-1"})
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = drain_retry_queue(q, cfg)

        assert result["drained"] == 1
        assert result["remote_ids"] == {"M-1": "REMOTE-1"}


# ---------------------------------------------------------------------------
# clear_retry_queue
# ---------------------------------------------------------------------------


class TestClearRetryQueue:
    def test_delegates_to_queue_clear(self) -> None:
        """clear_retry_queue calls queue.clear() (line 158)."""
        q = MagicMock()
        clear_retry_queue(q)
        q.clear.assert_called_once()


# ---------------------------------------------------------------------------
# publish_snapshot_hash
# ---------------------------------------------------------------------------


class TestPublishSnapshotHash:
    def test_sync_disabled_returns_failure(self, tmp_path: Path) -> None:
        """sync_enabled=False → return failure immediately (lines 170-171)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = False
        cfg.local_only = False
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False

    def test_no_platform_url_returns_failure(self, tmp_path: Path) -> None:
        """platform_url empty → return failure (lines 172-173)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = True
        cfg.memory_snapshot_publish_hash = True
        cfg.platform_url = ""
        cfg.local_only = False
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False

    def test_invalid_platform_url_returns_failure(self, tmp_path: Path) -> None:
        """Invalid URL → warning + return failure (lines 174-176)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = True
        cfg.memory_snapshot_publish_hash = True
        cfg.platform_url = "not-a-url"
        cfg.local_only = False
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False

    def test_missing_snapshot_file_returns_failure(self, tmp_path: Path) -> None:
        """snapshot_path doesn't exist → debug + return failure (lines 177-179)."""
        cfg = _cfg_sync_enabled()
        cfg.memory_snapshot_publish_hash = True
        result = publish_snapshot_hash(tmp_path / "nonexistent.db", cfg)
        assert result["success"] is False

    def test_oserror_on_hash_returns_retryable(self, tmp_path: Path) -> None:
        """OSError from _hash_snapshot_file → retryable failure (lines 183-185)."""
        cfg = _cfg_sync_enabled()
        cfg.memory_snapshot_publish_hash = True
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        with patch("trw_memory.sync._remote_publish._hash_snapshot_file", side_effect=OSError("io")):
            result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False
        assert result["retryable"] is True

    def test_successful_publish_returns_success(self, tmp_path: Path) -> None:
        """Successful POST → return success (lines 187-203)."""
        cfg = _cfg_sync_enabled()
        cfg.memory_snapshot_publish_hash = True
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"binary data here")
        mock_resp = _mock_response(status=200)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is True

    def test_non_2xx_returns_retryable(self, tmp_path: Path) -> None:
        """Non-2xx status → retryable failure (lines 204-205)."""
        cfg = _cfg_sync_enabled()
        cfg.memory_snapshot_publish_hash = True
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        mock_resp = _mock_response(status=503)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False
        assert result["retryable"] is True

    def test_httpx_error_returns_retryable(self, tmp_path: Path) -> None:
        """httpx.HTTPError → retryable failure (lines 206-208)."""
        cfg = _cfg_sync_enabled()
        cfg.memory_snapshot_publish_hash = True
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("timeout")
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = publish_snapshot_hash(snap, cfg)
        assert result["success"] is False
        assert result["retryable"] is True


# ---------------------------------------------------------------------------
# _hash_snapshot_file
# ---------------------------------------------------------------------------


class TestHashSnapshotFile:
    def test_returns_sha256_digest_and_size(self, tmp_path: Path) -> None:
        """_hash_snapshot_file hashes file content and returns (digest, size) (lines 212-218)."""
        data = b"hello world" * 100
        snap = tmp_path / "snap.db"
        snap.write_bytes(data)
        digest, size = _hash_snapshot_file(snap)
        assert len(digest) == 64  # SHA-256 hex
        assert size == len(data)


# ---------------------------------------------------------------------------
# retire_remote_memory
# ---------------------------------------------------------------------------


class TestRetireRemoteMemory:
    def test_sync_disabled_returns_true(self) -> None:
        """sync_enabled=False or no remote_id → return True early (line 226)."""
        cfg = MemoryConfig()
        cfg.sync_enabled = False
        cfg.platform_url = ""
        cfg.local_only = False
        assert retire_remote_memory("REMOTE-1", cfg) is True

    def test_non_2xx_returns_false(self) -> None:
        """Non-2xx PATCH response → return False (lines 241-242)."""
        cfg = _cfg_sync_enabled()
        mock_resp = _mock_response(status=404)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.patch.return_value = mock_resp
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = retire_remote_memory("REMOTE-1", cfg)
        assert result is False

    def test_httpx_error_returns_false(self) -> None:
        """httpx.HTTPError during retire → return False (lines 243-245)."""
        cfg = _cfg_sync_enabled()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.patch.side_effect = httpx.ConnectError("timeout")
        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = retire_remote_memory("REMOTE-1", cfg)
        assert result is False


# ---------------------------------------------------------------------------
# local_only gate tests
# ---------------------------------------------------------------------------


class TestLocalOnlyGates:
    def test_drain_retry_queue_local_only_raises(self) -> None:
        """drain_retry_queue with local_only=True → LocalOnlyViolationError (lines 130-131)."""
        cfg = MemoryConfig()
        cfg.local_only = True
        q = MagicMock()
        with pytest.raises(LocalOnlyViolationError):
            drain_retry_queue(q, cfg)

    def test_publish_snapshot_hash_local_only_raises(self, tmp_path: Path) -> None:
        """publish_snapshot_hash with local_only=True → LocalOnlyViolationError (lines 168-169)."""
        cfg = MemoryConfig()
        cfg.local_only = True
        snap = tmp_path / "snap.db"
        snap.write_bytes(b"data")
        with pytest.raises(LocalOnlyViolationError):
            publish_snapshot_hash(snap, cfg)


# ---------------------------------------------------------------------------
# drain_retry_queue inner publish_payload closure
# ---------------------------------------------------------------------------


class TestDrainRetryQueueClosure:
    def test_inner_publish_payload_maps_remote_id(self) -> None:
        """Inner publish_payload closure maps remote_id when publish succeeds (lines 141-146)."""
        cfg = _cfg_sync_enabled()

        # Use a queue mock that actually invokes the callback with a payload
        captured_fn: list[object] = []

        def fake_drain(fn: object) -> tuple[dict[str, int], list[str]]:
            captured_fn.append(fn)
            assert callable(fn)
            payload = {"source_learning_id": "M-closure"}
            fn(payload)  # type: ignore[operator]
            return {"drained": 1, "failed": 0, "skipped": 0}, ["M-canonical"]

        q = MagicMock()
        q.depth.return_value = 0
        q._drain_with_ids.side_effect = fake_drain

        mock_resp = _mock_response(status=201, json_body={"id": "REMOTE-CLOSURE"})
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=mock_client):
            result = drain_retry_queue(q, cfg)

        assert result["drained"] == 1
        assert result["remote_ids"].get("M-canonical") == "REMOTE-CLOSURE"
