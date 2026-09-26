"""The loopback daemon's serving loop -- PRD-CORE-253 FR03 properties 2 and 5.

``trw-memory-server serve http`` lands here. The sequence is deliberate:

1. resolve the daemon's file locations and refuse a retired Slice A bearer;
2. claim the single-instance slot AND bind the loopback socket under one lock,
   so a second start refuses before it can bind (see :mod:`._instance`);
3. build the fastmcp streamable-HTTP app with the grant verifier attached, so
   an unauthenticated request never reaches a tool body and an authenticated
   one reaches only its granted namespaces (PRD-CORE-298 FR02); and
4. serve on the already-bound socket, exiting after the idle window and
   removing the discovery record on the way out.

Binding before serving is what makes ``port=0`` usable: the assigned port is
read off the socket and published in the discovery file before uvicorn starts,
so a client never races the bind.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Iterator

import structlog
import uvicorn
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from trw_memory._dir_trust import verify_ancestor_chain_trusted
from trw_memory.daemon._arg_bounds import call_with_body_cap
from trw_memory.daemon._instance import claim_single_instance, release_single_instance
from trw_memory.daemon._offload import shutdown_offload_pool
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._verifier import LoopbackTokenVerifier
from trw_memory.daemon.client import _package_version
from trw_memory.exceptions import ConfigError, UntrustedDirectoryError
from trw_memory.models.config import MemoryConfig
from trw_memory.user_paths import require_supported_platform

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
    """ASGI wrapper recording HTTP activity for the idle watchdog.

    The daemon is idle only when no HTTP request is in flight AND none has
    started or finished within the window. Stamping arrival alone let the
    watchdog shut the server down underneath a tool call that ran longer than
    the window, and the caller waited on a response that never came. The
    ``lifespan`` scope passes through uncounted: it spans the server's whole
    life and would otherwise keep the daemon up forever.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self.last_request_at = time.monotonic()
        self.in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        self.in_flight += 1
        self.last_request_at = time.monotonic()
        try:
            await call_with_body_cap(self._app, scope, receive, send)
        finally:
            self.in_flight -= 1
            self.last_request_at = time.monotonic()

    def idle_for(self, idle_shutdown_seconds: float) -> bool:
        """True when nothing is in flight and the last activity is a window old."""
        return self.in_flight == 0 and time.monotonic() - self.last_request_at >= idle_shutdown_seconds


def _idle_poll_seconds(idle_shutdown_seconds: float) -> float:
    return min(max(idle_shutdown_seconds / _IDLE_POLL_DIVISOR, _IDLE_POLL_MIN_SECONDS), _IDLE_POLL_MAX_SECONDS)


async def _watch_idle(tracker: _IdleTracker, server: uvicorn.Server, idle_shutdown_seconds: float) -> None:
    """Ask the server to exit once nothing has been in flight for the window."""
    poll = _idle_poll_seconds(idle_shutdown_seconds)
    while not server.should_exit:
        await asyncio.sleep(poll)
        if tracker.idle_for(idle_shutdown_seconds):
            logger.info("daemon_idle_shutdown", idle_seconds=idle_shutdown_seconds)
            server.should_exit = True
            return


def _build_app(paths: DaemonPaths) -> ASGIApp:
    """Return the streamable-HTTP app with the grant verifier attached.

    Stateless with plain JSON responses (W27): no daemon tool uses MCP session
    state, and a session-holding server made every client open a standing SSE
    GET, which the idle tracker counts as in flight, plus a DELETE on close.
    """
    from trw_memory.daemon._version_gate import VersionGate
    from trw_memory.server import mcp

    mcp.auth = LoopbackTokenVerifier(paths)
    if not any(isinstance(layer, VersionGate) for layer in mcp.middleware):
        mcp.add_middleware(VersionGate(_package_version()))
    return mcp.http_app(transport="streamable-http", stateless_http=True, json_response=True)


@contextlib.contextmanager
def _record_termination_signals() -> Iterator[list[signal.Signals]]:
    """Capture SIGTERM/SIGINT without letting them kill the process.

    This is the fix for the discovery record that outlived the daemon
    (PRD-CORE-279 FR05). The cleanup below was always in a ``finally``; the
    reason it did not run is that it never got the chance. uvicorn's
    ``Server.capture_signals`` installs its own handlers, and when ``serve()``
    finishes it RESTORES the handlers it found and then calls
    ``signal.raise_signal()`` for every signal it captured. Whatever is
    installed here is therefore what runs at that re-raise -- and the default
    disposition, which is what used to be installed, terminates the process
    from inside ``serve()``, before this function's ``finally``.

    The handler does nothing but record the number. No file I/O, no lock: a
    handler runs between bytecodes of whatever the main thread was doing, and
    taking the claim lock there could deadlock against a lock the interrupted
    code already holds. Cleanup happens in ordinary code afterwards, and the
    signal is re-delivered under its default disposition once the record is
    gone, so the process still dies FROM the signal.
    """
    captured: list[signal.Signals] = []

    def _release_on_signal(signum: int, frame: object) -> None:
        captured.append(signal.Signals(signum))

    installed: dict[signal.Signals, object] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):  # not the main thread
            installed[sig] = signal.signal(sig, _release_on_signal)
    try:
        yield captured
    finally:
        for sig, previous in installed.items():
            with contextlib.suppress(ValueError):
                signal.signal(sig, previous)  # type: ignore[arg-type]


def _redeliver(captured: list[signal.Signals]) -> None:
    """Die from the signal that asked us to stop, now that cleanup is done."""
    if not captured:
        return
    signum = captured[-1]
    logger.info("daemon_signal_shutdown", signal=signum)
    with contextlib.suppress(ValueError):
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)


async def serve_loopback(options: DaemonServeOptions, *, paths: DaemonPaths | None = None) -> None:
    """Run the loopback daemon until its idle window elapses or it is signalled.

    Each request carries a checkout's token, and that token reaches only the
    namespaces its grant names (PRD-CORE-298 FR02); there is no store-wide
    bearer. A retired Slice A ``daemon-token`` refuses startup rather than
    being honoured, so an old all-namespace secret cannot outlive the upgrade.

    Args:
        options: Port and idle window for this invocation.
        paths: Daemon file locations. Defaults to the machine-local user
            memory directory (FR01).

    Raises:
        DaemonAlreadyRunningError: A live daemon already holds the claim; this
            process exits without binding and without touching its files.
    """
    require_supported_platform()  # before anything is opened, even with explicit paths
    resolved = paths or DaemonPaths.resolve()
    # PRD-SEC-016 FR04: refuse to serve a store whose directory chain another
    # principal can rewrite, BEFORE anything is opened or the socket is
    # bound. UntrustedDirectoryError is re-raised as ConfigError so it joins
    # the same "refuses to start" surface as the retired-token check below
    # (both name the reason in the exception message that the caller logs
    # and exits non-zero on).
    try:
        verify_ancestor_chain_trusted(resolved.user_memory_dir)
    except UntrustedDirectoryError as exc:
        raise ConfigError(str(exc)) from exc
    os.environ.setdefault(_STORAGE_PATH_ENV, str(resolved.user_memory_dir))
    os.environ.setdefault(_SINGLE_STORE_ENV, str(resolved.store))
    # Every tool body runs on one serialized lane, bounded on the single SQLite store only (rc9).
    if (backend := MemoryConfig().storage_backend) != "sqlite":
        raise ConfigError(f"refusing to start: the daemon serves one SQLite store, not storage_backend={backend!r}")
    if resolved.token.exists() or resolved.token.is_symlink():
        raise ConfigError(
            f"refusing to start: {resolved.token} is a retired all-namespace bearer (PRD-CORE-298 FR02). "
            f"Run `trw-mcp memory token --migrate` to delete it; each checkout then mints its own grant."
        )
    # Signal recording starts BEFORE the claim. The record is published by
    # ``claim_single_instance``, so a SIGTERM that lands between publication and
    # handler installation would otherwise kill the process with the default
    # disposition and leave exactly the residue this fixes (FR05/FR06).
    with _record_termination_signals() as captured:
        claim = claim_single_instance(
            resolved,
            port=options.port,
            version=_package_version(),
        )
        watchdog: asyncio.Task[None] | None = None
        try:
            # Everything past the claim is inside the try: a failure while
            # building the app or the server is the case that used to leak a
            # record naming a process that never served (FR06).
            tracker = _IdleTracker(_build_app(resolved))
            server = uvicorn.Server(uvicorn.Config(tracker, log_config=None, lifespan="on"))
            logger.info("daemon_serving", url=claim.info.url, pid=claim.info.pid)
            watchdog = asyncio.create_task(_watch_idle(tracker, server, options.idle_shutdown_seconds))
            if not captured:
                await server.serve(sockets=[claim.sock])
        finally:
            try:
                if watchdog is not None:
                    watchdog.cancel()
                claim.sock.close()
                # Drain the workers BEFORE withdrawing the endpoint, so a store
                # write that finishes in time cannot outlive the record that
                # advertised it. The drain is bounded; a stuck worker does not
                # get to hold the daemon past its shutdown.
                shutdown_offload_pool()
                release_single_instance(resolved, claimed=claim.info)
            finally:
                # Redelivery lives in a finally so an exception on the way out
                # cannot swallow the operator's SIGTERM: the process must still
                # die FROM the signal, after the record is gone.
                _redeliver(captured)
