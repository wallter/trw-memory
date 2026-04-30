"""Tests for trw_memory.sync.subscriber and sync config — PRD-CORE-047."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig
from trw_memory.sync.conflict import MAX_MERGED_DETAIL_LENGTH
from trw_memory.sync.subscriber import RECONNECT_DELAY, SSESubscriber

from ._test_sync_support import make_sync_config as _make_config


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


class TestModuleConstants:
    """Verify module-level subscriber/conflict constants are set correctly."""

    def test_reconnect_delay(self) -> None:
        assert RECONNECT_DELAY == 5.0

    def test_max_merged_detail_length(self) -> None:
        assert MAX_MERGED_DETAIL_LENGTH == 2000


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
        call_count = 0

        def side_effect(*_args: Any, **_kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("server error")
            sub._stop_event.set()
            raise httpx.HTTPError("stop")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = side_effect
        mock_client_cls.return_value = mock_client

        sub._listen_loop()
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
        sub._listen_loop()
        assert sub._stop_event.is_set()

    @patch("trw_memory.sync.subscriber.httpx.Client")
    def test_subscriber_last_event_id_sent_on_reconnect(self, mock_client_cls: MagicMock) -> None:
        """After setting _last_event_id, reconnect sends it in headers."""
        cfg = _make_config(platform_api_key="test-key")
        sub = SSESubscriber(cfg, on_event=lambda _data: None)
        sub._last_event_id = "evt-99"

        captured_headers: list[dict[str, str]] = []

        def stream_side_effect(
            _method: str,
            _url: str,
            headers: dict[str, str] | None = None,
            **_kw: Any,
        ) -> MagicMock:
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
