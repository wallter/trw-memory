"""The loopback daemon's serving loop -- PRD-CORE-253 FR03 properties 2 and 5.

``trw-memory-server serve http`` lands here. The sequence is deliberate:

1. resolve the daemon's file locations and ensure the per-user token exists;
2. claim the single-instance slot AND bind the loopback socket under one lock,
   so a second start refuses before it can bind (see :mod:`._instance`);
3. build the fastmcp streamable-HTTP app with the token verifier attached, so
   an unauthenticated request never reaches a tool body; and
4. serve on the already-bound socket, exiting after the idle window and
   removing the discovery record on the way out.

Binding before serving is what makes ``port=0`` usable: the assigned port is
read off the socket and published in the discovery file before uvicorn starts,
so a client never races the bind.
"""

from __future__ import annotations

import asyncio
import os
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import structlog
import uvicorn
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from trw_memory.daemon._instance import claim_single_instance, release_single_instance
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._token import ensure_token
from trw_memory.daemon._verifier import LoopbackTokenVerifier
from trw_memory.models.config import MemoryConfig

__all__ = ["DaemonServeOptions", "serve_loopback"]

logger = structlog.get_logger(__name__)

#: Bounds on how often the idle watchdog wakes. It polls at a tenth of the idle
#: window so shutdown lands within 10% of the configured deadline, clamped so a
#: 30-minute window does not sleep through a shutdown signal and a 2-second one
#: (an operator's explicit short-lived daemon) does not spin.
_IDLE_POLL_DIVISOR = 10
_IDLE_POLL_MIN_SECONDS = 0.05
_IDLE_POLL_MAX_SECONDS = 1.0

#: Environment variables the daemon pins so every ``MemoryConfig()`` built
#: inside this process resolves the ONE user-space store rather than the
#: caller's working directory and a per-namespace file under it. Pinning the
#: single store is what makes PRD-CORE-253 FR01 a fact: without it each
#: namespace still got its own SQLite file and ``DaemonPaths.store`` was a path
#: no write path ever opened. ``setdefault`` keeps an operator's explicit value.
_STORAGE_PATH_ENV = "MEMORY_STORAGE_PATH"
#: NOTE the name: ``memory_single_store_path`` carries an explicit
#: ``validation_alias``, and pydantic-settings then reads the alias VERBATIM
#: rather than applying ``env_prefix``. So it is MEMORY_SINGLE_STORE_PATH,
#: not MEMORY_MEMORY_SINGLE_STORE_PATH as the prefix rule would suggest.
_SINGLE_STORE_ENV = "MEMORY_SINGLE_STORE_PATH"


class DaemonServeOptions(BaseModel):
    """Typed, per-invocation serving options.

    Defaults come from :class:`~trw_memory.models.config.MemoryConfig`; the
    ``serve http`` CLI can override them for one run. ``idle_shutdown_seconds``
    is a float with a bare ``gt=0`` bound rather than the config field's
    ``ge=60`` floor, because the floor is unattended-operation policy while an
    explicit flag is a deliberate operator act (and is what lets an integration
    test observe a real idle shutdown).
    """

    port: int = Field(ge=0, le=65535, description="Loopback port; 0 asks for an ephemeral one")
    idle_shutdown_seconds: float = Field(gt=0.0, description="Seconds without a request before the daemon exits")

    @classmethod
    def from_config(
        cls,
        config: MemoryConfig,
        *,
        port: int | None = None,
        idle_shutdown_seconds: float | None = None,
    ) -> DaemonServeOptions:
        """Build the options a ``serve http`` invocation uses.

        Args:
            config: Source of the defaults.
            port: CLI override for ``memory_daemon_port``.
            idle_shutdown_seconds: CLI override for
                ``memory_daemon_idle_shutdown_seconds``.
        """
        return cls(
            port=config.memory_daemon_port if port is None else port,
            idle_shutdown_seconds=(
                float(config.memory_daemon_idle_shutdown_seconds)
                if idle_shutdown_seconds is None
                else idle_shutdown_seconds
            ),
        )


class _IdleTracker:
    """ASGI wrapper recording when the last request arrived."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self.last_request_at = time.monotonic()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.last_request_at = time.monotonic()
        await self._app(scope, receive, send)


def _package_version() -> str:
    try:
        return package_version("trw-memory")
    except PackageNotFoundError:  # pragma: no cover - only in a source tree without metadata
        return "unknown"


def _idle_poll_seconds(idle_shutdown_seconds: float) -> float:
    return min(max(idle_shutdown_seconds / _IDLE_POLL_DIVISOR, _IDLE_POLL_MIN_SECONDS), _IDLE_POLL_MAX_SECONDS)


async def _watch_idle(tracker: _IdleTracker, server: uvicorn.Server, idle_shutdown_seconds: float) -> None:
    """Ask the server to exit once no request has arrived for the window."""
    poll = _idle_poll_seconds(idle_shutdown_seconds)
    while not server.should_exit:
        await asyncio.sleep(poll)
        if time.monotonic() - tracker.last_request_at >= idle_shutdown_seconds:
            logger.info("daemon_idle_shutdown", idle_seconds=idle_shutdown_seconds)
            server.should_exit = True
            return


def _build_app(token: str) -> ASGIApp:
    """Return the streamable-HTTP app with the token verifier attached."""
    from trw_memory.server import mcp

    mcp.auth = LoopbackTokenVerifier(token)
    return mcp.http_app(transport="streamable-http")


async def serve_loopback(options: DaemonServeOptions, *, paths: DaemonPaths | None = None) -> None:
    """Run the loopback daemon until its idle window elapses.

    Args:
        options: Port and idle window for this invocation.
        paths: Daemon file locations. Defaults to the machine-local user
            memory directory (FR01).

    Raises:
        DaemonAlreadyRunningError: A live daemon already holds the claim; this
            process exits without binding and without touching its files.
    """
    resolved = paths or DaemonPaths.resolve()
    os.environ.setdefault(_STORAGE_PATH_ENV, str(resolved.user_memory_dir))
    os.environ.setdefault(_SINGLE_STORE_ENV, str(resolved.store))
    token = ensure_token(resolved)
    claim = claim_single_instance(
        resolved,
        port=options.port,
        token=token,
        version=_package_version(),
    )
    tracker = _IdleTracker(_build_app(token))
    server = uvicorn.Server(uvicorn.Config(tracker, log_config=None, lifespan="on"))
    logger.info("daemon_serving", url=claim.info.url, pid=claim.info.pid)
    watchdog = asyncio.create_task(_watch_idle(tracker, server, options.idle_shutdown_seconds))
    try:
        await server.serve(sockets=[claim.sock])
    finally:
        watchdog.cancel()
        claim.sock.close()
        release_single_instance(resolved)
