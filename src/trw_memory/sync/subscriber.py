"""SSE subscriber for real-time learning updates from the platform.

Implements FR03 from PRD-CORE-047.  Runs in a daemon thread so it does not
block session shutdown.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

import structlog

from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger()

RECONNECT_DELAY = 5.0  # seconds
HEARTBEAT_TIMEOUT = 30.0


class SSESubscriber:
    """Background SSE subscriber that listens for ``learning_published`` events.

    Runs in a daemon thread so it terminates automatically when the main
    session thread exits.
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        on_event: Callable[[dict[str, Any]], None],
    ) -> None:
        self._cfg = cfg
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_event_id: str | None = None

    def start(self) -> None:
        """Start the SSE subscription in a daemon thread."""
        if not self._cfg.sync_enabled or not self._cfg.platform_url:
            return

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="sse-subscriber",
            daemon=True,
        )
        self._thread.start()
        logger.debug("sse_subscriber_started")

    def stop(self) -> None:
        """Signal the subscriber to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.debug("sse_subscriber_stopped")

    def _listen_loop(self) -> None:
        """Main event loop -- connects, reads SSE, reconnects on failure."""
        import httpx

        url = f"{self._cfg.platform_url.rstrip('/')}/v1/learnings/stream"

        while not self._stop_event.is_set():
            try:
                headers: dict[str, str] = {}
                if self._cfg.platform_api_key:
                    headers["Authorization"] = (
                        f"Bearer {self._cfg.platform_api_key}"
                    )
                if self._last_event_id:
                    headers["Last-Event-ID"] = self._last_event_id

                with httpx.Client(timeout=None) as client:
                    with client.stream("GET", url, headers=headers) as response:
                        for line in response.iter_lines():
                            if self._stop_event.is_set():
                                return
                            self._process_line(line)
            except Exception:
                logger.debug("sse_connection_error", exc_info=True)

            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=RECONNECT_DELAY)

    def _process_line(self, line: str) -> None:
        """Process a single SSE line."""
        if line.startswith("id:"):
            self._last_event_id = line[3:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if not data_str:
                return
            try:
                data = json.loads(data_str)
                event_type = data.get("type", "")
                if event_type == "learning_published":
                    self._on_event(data)
                    logger.debug(
                        "sse_event_received",
                        event_type=event_type,
                    )
            except json.JSONDecodeError:
                pass
