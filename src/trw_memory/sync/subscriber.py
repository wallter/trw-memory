"""SSE subscriber for real-time learning updates from the platform.

Implements FR03 from PRD-CORE-047.  Runs in a daemon thread so it does not
block session shutdown.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

import httpx
import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.sync._remote_common import build_platform_headers, platform_contact_blocked

logger = structlog.get_logger(__name__)

RECONNECT_DELAY = 5.0  # seconds
#: How often an open stream rechecks the platform contact switch (B71-106).
SWITCH_POLL = 5.0  # seconds


class SSESubscriber:
    """Background SSE subscriber that listens for ``learning_published`` events.

    Runs in a daemon thread so it terminates automatically when the main
    session thread exits.
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        on_event: Callable[[dict[str, object]], None],
    ) -> None:
        self._cfg = cfg
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._connection_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_event_id: str | None = None
        self._pending_event_id: str | None = None
        self._pending_event_type: str | None = None
        self._active_client: httpx.Client | None = None
        self._active_response: httpx.Response | None = None
        self._watcher: threading.Thread | None = None

    def start(self) -> None:
        """Start the SSE subscription in a daemon thread."""
        if platform_contact_blocked(self._cfg, "sse_subscriber", live=False):
            return
        self._thread = threading.Thread(target=self._listen_loop, name="sse-subscriber", daemon=True)
        self._watcher = threading.Thread(target=self._watch_switch, name="sse-contact-watch", daemon=True)
        self._thread.start()
        self._watcher.start()
        logger.debug("sse_subscriber_started")

    def stop(self) -> None:
        """Signal the subscriber to stop."""
        self._stop_event.set()
        self._close_active()
        for thread in (self._thread, self._watcher):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        logger.debug("sse_subscriber_stopped")

    def _close_active(self) -> None:
        with self._connection_lock:
            active_response, active_client = self._active_response, self._active_client
            self._active_response = self._active_client = None
        if active_response is not None:
            active_response.close()
        if active_client is not None:
            active_client.close()

    def _watch_switch(self) -> None:
        """Close an open stream once platform contact is switched off; the listen loop then waits (B71-106)."""
        while not self._stop_event.wait(timeout=SWITCH_POLL):
            if platform_contact_blocked(self._cfg, "sse_subscriber"):
                self._close_active()

    def _listen_loop(self) -> None:
        """Main event loop -- connects, reads SSE, reconnects on failure."""
        url = f"{self._cfg.platform_url.rstrip('/')}/v1/learnings/stream"

        while not self._stop_event.is_set():
            if platform_contact_blocked(self._cfg, "sse_subscriber"):  # every connect and reconnect asks
                self._stop_event.wait(timeout=RECONNECT_DELAY)
                continue
            try:
                # build_platform_headers is the ONE function that may build
                # this header (see trw_memory.sync._remote_common); it drops
                # "Content-Type" is harmless on a GET and withholds the
                # bearer from an untrusted host.
                headers: dict[str, str] = build_platform_headers(self._cfg.platform_api_key, url)
                headers.pop("Content-Type", None)
                if self._last_event_id:
                    headers["Last-Event-ID"] = self._last_event_id

                client = httpx.Client(timeout=None)  # noqa: S113 — timeout=None is intentional: SSE long-poll connection must stay open indefinitely until a message arrives
                # Registered BEFORE connecting, so stop() or the watcher can close a stalled handshake.
                with self._connection_lock:
                    if self._stop_event.is_set():  # stop() already ran: nothing may connect now
                        client.close()
                        return
                    self._active_client = client
                try:
                    with client, client.stream("GET", url, headers=headers) as response:
                        with self._connection_lock:
                            self._active_response = response
                        # The switch may have turned off during the handshake: ask again before reading.
                        lines = () if platform_contact_blocked(self._cfg, "sse_subscriber") else response.iter_lines()
                        for line in lines:
                            if self._stop_event.is_set():
                                return
                            self._process_line(line)
                finally:
                    with self._connection_lock:
                        self._active_response = self._active_client = None
            except (httpx.HTTPError, httpx.StreamError, OSError):  # StreamError: closed by stop() or the watcher
                logger.debug("sse_connection_error", exc_info=True)

            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=RECONNECT_DELAY)

    def _process_line(self, line: str) -> None:
        """Process a single SSE line."""
        if line.startswith("id:"):
            self._pending_event_id = line[3:].strip()
        elif line.startswith("event:"):
            self._pending_event_type = line[6:].strip() or None
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if not data_str:
                return
            try:
                raw = json.loads(data_str)
                if not isinstance(raw, dict):
                    return
                data: dict[str, object] = raw
                event_type = self._pending_event_type or str(data.get("type", ""))
                if event_type in {"learning_published", "learning_updated", "learning_retired"}:
                    # Standard SSE puts the event name in the ``event:`` line;
                    # downstream handlers consume the normalized payload only.
                    data["type"] = event_type
                    if self._pending_event_id:
                        self._last_event_id = self._pending_event_id
                    self._on_event(data)
                    logger.debug(
                        "sse_event_received",
                        event_type=event_type,
                    )
            except json.JSONDecodeError as exc:
                # Content-free diagnostic: the SSE payload carries platform
                # learning text, so log only structural locators (error class +
                # length), never the raw data, matching the persisted-state
                # readers (retry_queue / warm sidecar / recovery state).
                logger.debug(
                    "sse_malformed_json",
                    error_class=type(exc).__name__,
                    data_length=len(data_str),
                )
            finally:
                self._pending_event_id = None
                self._pending_event_type = None
