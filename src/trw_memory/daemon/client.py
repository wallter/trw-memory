"""Fail-closed client for the loopback daemon -- PRD-CORE-253 FR08.

When the store cannot be reached, **both reads and writes fail**. There is no
read-only snapshot fallback, and no local store is created anywhere. That is
deliberately stricter than the framework's previous fail-open recall posture:
an agent that recalls from a stale snapshot writes a conclusion derived from
it, and that write either never lands (so the conclusion is acted on but
unrecorded) or later merges against a corpus that had already contradicted it.
The value hierarchy puts truthfulness above velocity, and the absence of a
fallback location is what makes split-brain impossible rather than unlikely.

Four behaviours, one per FR08 clause:

1. **Connect failure** -- try once, retry exactly once, then raise
   :class:`~trw_memory.exceptions.DaemonUnreachableError` naming the discovery
   file, the start command and the underlying error class.
2. **No token file** -- generate one at 0600 and continue. First run is not an
   error. A token file that EXISTS but cannot be read is not first run, and
   raises :class:`~trw_memory.exceptions.TokenUnreadableError` rather than
   being replaced; likewise an untrusted ``daemon.json`` raises
   :class:`~trw_memory.exceptions.DaemonRecordInvalidError` instead of being
   read as an empty slot to spawn into.
3. **Token rejected** -- raise :class:`~trw_memory.exceptions.DaemonAuthError`
   and do NOT regenerate. Automatic rotation on rejection would let any local
   process force one by corrupting the file, and would mask a daemon started
   under a different account.
4. **No partial store** -- a failed attach opens no SQLite file, because this
   client has no SQLite path at all.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

import structlog
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from trw_memory.daemon._discovery import DaemonInfo, DiscoveryInvalid, read_live_discovery
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._token import ensure_token
from trw_memory.exceptions import DaemonAuthError, DaemonRecordInvalidError, DaemonUnreachableError
from trw_memory.models.config import MemoryConfig

__all__ = ["DAEMON_START_COMMAND", "DaemonClient", "start_daemon_detached"]

logger = structlog.get_logger(__name__)

#: The command an operator runs to start the daemon by hand. Quoted verbatim in
#: every unreachable error, so the failure carries its own remedy.
DAEMON_START_COMMAND = "trw-memory-server serve http"

#: Total attempts per call: the first, plus exactly one retry (FR08 clause 1).
_MAX_ATTEMPTS = 2

#: How often the auto-start wait re-reads the discovery file.
_DISCOVERY_POLL_SECONDS = 0.05

#: HTTP status the daemon returns for a missing or wrong bearer token.
_UNAUTHORIZED_STATUS = 401


def start_daemon_detached(paths: DaemonPaths) -> None:
    """Spawn a daemon in its own session, detached from this process.

    Invoked as ``python -m trw_memory.server serve http`` rather than through
    the console script, so auto-start does not depend on the installing
    environment having put the script on ``PATH``.
    """
    logger.info("daemon_auto_start", discovery=str(paths.discovery))
    subprocess.Popen(
        [sys.executable, "-m", "trw_memory.server", "serve", "http"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _is_unauthorized(exc: BaseException) -> bool:
    """Whether *exc* (or a cause in its chain) is a 401 rejection."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == _UNAUTHORIZED_STATUS:
            return True
        current = current.__cause__ or current.__context__
    return False


class DaemonClient:
    """Calls daemon-served tools, failing closed when the daemon is absent."""

    def __init__(self, config: MemoryConfig | None = None, paths: DaemonPaths | None = None) -> None:
        """Args: config: source of the startup deadline. paths: daemon files."""
        self._config = config or MemoryConfig()
        self._paths = paths or DaemonPaths.resolve()

    @property
    def paths(self) -> DaemonPaths:
        """The daemon file locations this client attaches through."""
        return self._paths

    def _unreachable(self, reason: str) -> DaemonUnreachableError:
        return DaemonUnreachableError(
            f"the trw-memory daemon is unreachable ({reason}). No memory was read or written, and no "
            f"local store was created. Discovery file: {self._paths.discovery}. "
            f"Start it with: {DAEMON_START_COMMAND}. "
            f"If a stale record names a process that no longer serves, remove {self._paths.discovery} and retry."
        )

    def _refuse_invalid(self, invalid: DiscoveryInvalid) -> DaemonRecordInvalidError:
        """Explain why an untrusted record stops the client rather than starting one."""
        return DaemonRecordInvalidError(
            f"{invalid.path} exists but cannot be trusted -- {invalid.reason}. No memory was read or "
            f"written, and no daemon was started: the record may name a daemon that is still serving "
            f"this store, and spawning a second one would put two writers on {self._paths.store}. "
            f"Inspect the file and remove it if no daemon is running, then retry."
        )

    def _attach(self) -> DaemonInfo:
        """Return a live daemon, auto-starting one only if the slot is free.

        Auto-start is gated on :class:`DiscoveryAbsent` specifically. An
        untrusted record is not an absent one: spawning on it would bind a
        second endpoint over a daemon that may still be serving.
        """
        result = read_live_discovery(self._paths)
        if isinstance(result, DaemonInfo):
            return result
        if isinstance(result, DiscoveryInvalid):
            raise self._refuse_invalid(result)
        # FR08 clause 2: a missing token is first-run, not an error. Generating
        # it HERE rather than letting the daemon do it means the client and the
        # daemon it spawns agree on one token even on a cold install.
        ensure_token(self._paths)
        start_daemon_detached(self._paths)
        deadline = time.monotonic() + self._config.memory_daemon_startup_timeout_seconds
        while time.monotonic() < deadline:
            result = read_live_discovery(self._paths)
            if isinstance(result, DaemonInfo):
                return result
            if isinstance(result, DiscoveryInvalid):
                raise self._refuse_invalid(result)
            time.sleep(_DISCOVERY_POLL_SECONDS)
        raise self._unreachable(
            f"auto-start did not publish a discovery file within {self._config.memory_daemon_startup_timeout_seconds}s"
        )

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a daemon-served tool, or fail closed.

        Args:
            name: Registered tool name, e.g. ``"memory_recall"``.
            arguments: Tool arguments.

        Returns:
            The tool's structured result.

        Raises:
            DaemonAuthError: The daemon rejected the token. Not retried, and
                the token file is left exactly as it is.
            DaemonRecordInvalidError: ``daemon.json`` exists but cannot be
                trusted. Nothing was spawned and no file was rewritten.
            TokenUnreadableError: The token file exists but cannot be read. It
                was NOT replaced.
            DaemonUnreachableError: The daemon could not be reached after the
                first attempt and its single retry.
        """
        last_error: BaseException | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Re-resolve on the retry. NFR02 requires that a daemon killed
            # mid-request results in either a restarted daemon serving the retry
            # or a fail-closed error; reusing the first attempt's endpoint could
            # only ever produce the second, because the retry would dial the
            # dead port again. Re-attaching lets the auto-start in ``_attach``
            # do its job, so the one retry is a real second chance.
            info = self._attach()
            # The authoritative secret is the TOKEN FILE, not the copy the daemon
            # recorded in its discovery file. That is what makes FR08 clause 3
            # reachable: a daemon started under a different account, or a corrupted
            # token file, produces a rejection the operator has to resolve rather
            # than a silent rotation.
            token = ensure_token(self._paths)
            try:
                transport = StreamableHttpTransport(url=info.url, auth=token)
                async with Client(transport) as client:
                    result = await client.call_tool(name, arguments or {})
                return result.data
            except Exception as exc:  # transport failures are classified immediately below
                if _is_unauthorized(exc):
                    logger.warning("daemon_token_rejected_by_server", tool=name)
                    raise DaemonAuthError(
                        f"the trw-memory daemon rejected this client's token. The token was NOT regenerated "
                        f"-- automatic rotation on rejection would let any local process force one. "
                        f"Remove {self._paths.token} and {self._paths.discovery}, then retry."
                    ) from exc
                last_error = exc
                logger.warning(
                    "daemon_call_failed",
                    tool=name,
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=type(exc).__name__,
                )
        raise self._unreachable(type(last_error).__name__ if last_error else "unknown error") from last_error

    async def store(self, content: str, namespace: str, **kwargs: Any) -> Any:
        """Write a memory entry through the daemon, or fail closed."""
        return await self.call_tool("memory_store", {"content": content, "namespace": namespace, **kwargs})

    async def recall(self, query: str, namespace: str, **kwargs: Any) -> Any:
        """Read memory entries through the daemon, or fail closed.

        Failing closed on a READ is the deliberate part: an empty-but-truthful
        error beats a partial view the caller cannot tell is partial.
        """
        return await self.call_tool("memory_recall", {"query": query, "namespace": namespace, **kwargs})
