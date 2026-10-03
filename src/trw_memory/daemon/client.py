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
   file, the start command and the underlying error class. A failure after the
   request was sent is retried only for a tool that lands the same way twice:
   a read, ``memory_update``, or ``memory_store`` carrying the ``entry_id`` the
   client mints before the first attempt. Any other write whose response is
   lost fails at once, saying it may have been applied. A draining daemon's
   marked 503 counts as never sent: the retry waits for the successor.
2. **The checkout's grant** -- the client presents the token it is given
   (PRD-CORE-298 FR02); it never mints one and never holds a store-wide
   bearer. An untrusted ``daemon.json`` raises
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

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, Literal
from uuid import uuid4

import structlog

from trw_memory.daemon import _direct
from trw_memory.daemon._discovery import (
    AGENT_MUST_NOT_STOP,
    DaemonInfo,
    DiscoveryInvalid,
    read_live_discovery,
)
from trw_memory.daemon._held_session import HeldSessions
from trw_memory.daemon._launcher_record import launch_from_record
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.daemon._session import is_unauthorized, never_sent, open_session
from trw_memory.daemon._spawn import SpawnedDaemon, start_daemon_detached
from trw_memory.daemon._upgrade import replace_older_daemon, withdrawn
from trw_memory.daemon._versions import majors_differ
from trw_memory.exceptions import (
    DaemonAuthError,
    DaemonProtocolError,
    DaemonRecordInvalidError,
    DaemonUnreachableError,
    DaemonVersionMismatchError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.user_paths import require_supported_platform

__all__ = ["DAEMON_START_COMMAND", "DaemonClient", "probe_endpoint", "start_daemon_detached"]

logger = structlog.get_logger(__name__)

#: The command an operator runs to start the daemon by hand. Quoted verbatim in
#: every unreachable error, so the failure carries its own remedy.
DAEMON_START_COMMAND = "trw-memory-server serve http"


def _package_version() -> str:
    """This trw-memory's own version: its source when run from one, not a stale editable dist-info (B71-111)."""
    from trw_memory._version import __version__

    return __version__


#: Total attempts per call: the first, plus exactly one retry (FR08 clause 1).
_MAX_ATTEMPTS = 2

#: How often the auto-start wait re-reads the discovery file.
_DISCOVERY_POLL_SECONDS = 0.05

#: Tools a retry may repeat after the request was sent: a second run lands the
#: same way as the first. ``memory_store`` qualifies because ``call_tool``
#: fixes its ``entry_id`` before the first attempt, so a replay updates the row
#: the lost attempt wrote instead of adding a second one. ``memory_update`` (a
#: correction) sets values, ``tags_add`` dedups and a closed prior is skipped, so a
#: replay leaves the row as the first run did; only its audit event repeats. A
#: conditional one (``if_revision``; ``memory_sync_apply`` always is) replays as
#: ``conflict``: its caller re-reads.
#: ``memory_reembed`` skips rows already in the active space.
_REPLAYABLE_TOOLS = frozenset(
    {
        "memory_assertion_health",
        "memory_audit",
        "memory_find_duplicate",
        "memory_list_page",
        "memory_get",
        "memory_graph_related",
        "memory_namespace_diagnose",
        "memory_quarantine_list",
        "memory_recall",
        "memory_reembed",
        "memory_search",
        "memory_status",
        "memory_store",
        "memory_sync_apply",
        "memory_sync_apply_many",
        "memory_sync_dirty_page",
        "memory_sync_find",
        "memory_sync_find_many",
        "memory_sync_mark_synced",
        "memory_update",
        "memory_vectors",
        "memory_anchored",
        "memory_similar",
    }
)


def _stop_unpublished(spawned: object) -> bool:
    """Stop a daemon this client started that never published; whether one was stopped.

    A daemon stuck before ``serve()`` never reaches its idle timer, so leaving it
    means it lives until someone kills it, and the next call spawns another beside
    it. A stub that spawned nothing (tests pass ``lambda _paths: None``) is not a process.
    """
    if not isinstance(spawned, SpawnedDaemon) or not spawned.stop():
        return False
    logger.warning("daemon_auto_start_stopped", pid=spawned.pid)
    return True


async def probe_endpoint(info: DaemonInfo, token: str | None, timeout: float) -> Literal["answered", "grant_refused"]:
    """Send one MCP ping to *info*'s endpoint, never starting a daemon (PRD-CORE-310 FR05).

    ``"grant_refused"`` still proves the endpoint serves: it authenticated the request and said no.

    Raises:
        DaemonUnreachableError: Nothing answered within *timeout* seconds.
    """

    async def _ping() -> None:
        async with open_session(info, token, _package_version(), timeout=timeout, init_timeout=timeout) as client:
            await client.ping()

    try:
        await asyncio.wait_for(_ping(), timeout)
    except Exception as exc:  # classified here: a 401 is an answer, anything else is none
        if is_unauthorized(exc):
            return "grant_refused"
        raise DaemonUnreachableError(f"{info.url} did not answer a ping ({type(exc).__name__})") from exc
    return "answered"


#: Optional arguments added after 8.0's first daemon shipped, sent only when set: a daemon started before the
#: upgrade refuses an unknown argument, so an unset one must not reach it (an unfiltered call keeps working),
#: while a set one is refused explicitly rather than silently ignored (PRD-CORE-334 review r1).
_OMIT_WHEN_UNSET: frozenset[str] = frozenset({"types"})


def _forward(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Replace a stub method's ``...`` body with a ``call_tool`` request (FR04).

    Applied by name, after the class body, to every method in
    ``_FORWARDED_METHODS`` -- mypy --strict still type-checks every caller
    against the method exactly as written in the class body; only the
    runtime function object is swapped, once, per name.

    *fn* (the still-real stub ``async def``, body ``...``) is read once with
    ``inspect.signature``: its name gives the served tool (``"memory_" +
    fn.__name__``), and its parameter list -- including declared defaults --
    gives the payload shape, so that shape is stated exactly once, in the
    stub's own signature, rather than duplicated a second time in a
    hand-written dict literal.
    """
    tool = f"memory_{fn.__name__}"
    params = tuple(inspect.signature(fn).parameters.values())[1:]  # drop ``self``
    names = tuple(p.name for p in params)
    defaults = {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}

    @functools.wraps(fn)
    async def wrapper(self: DaemonClient, *args: Any, **kwargs: Any) -> Any:
        payload = dict(defaults)
        payload.update(zip(names, args, strict=False))
        payload.update(kwargs)
        for late in _OMIT_WHEN_UNSET:
            if late in payload and payload[late] is None:
                del payload[late]
        return await self.call_tool(tool, payload)

    return wrapper


class DaemonClient:
    """Calls daemon-served tools, failing closed when the daemon is absent."""

    def __init__(
        self,
        token: str,
        config: MemoryConfig | None = None,
        paths: DaemonPaths | None = None,
        *,
        instance: tuple[int, str] | None = None,
        keep_session: bool = False,
        launcher: Callable[[DaemonPaths], SpawnedDaemon] | None = None,
    ) -> None:
        """Args: token: the checkout's grant. config: source of the startup deadline. paths: daemon files.

        instance: the ``(pid, started_at)`` a caller checked; every call to any other daemon is refused.
        keep_session: hold one MCP session open across calls made on the same event loop (W27).
        Only for a caller whose loop outlives its calls: a session opened on a loop that
        ``asyncio.run`` then closes is abandoned, not reused.
        launcher: starts the daemon when none runs, in place of this interpreter's
        :func:`start_daemon_detached`; for a caller that knows another installation must serve.
        """
        require_supported_platform()  # before explicit paths skip the resolver's own check (C12)
        self._token = token
        self._config = config or MemoryConfig()
        self._paths = paths or DaemonPaths.resolve()
        self._instance = instance
        self._keep_session = keep_session
        self._launcher = launcher
        self._sessions = HeldSessions()

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

    def _compatible(self, info: DaemonInfo, why: str = "") -> DaemonInfo:
        """*info*, unless it serves another major version than this client's (PRD-CORE-302 C7).

        *why* names the reason the client did not replace it itself (DAEMON-AUTO-RESTART-ON-UPGRADE).
        """
        if not majors_differ(info.version, _package_version()):
            return info
        not_replaced = f" The client did not replace it automatically: {why}." if why else ""
        raise DaemonVersionMismatchError(
            f"daemon_version_mismatch: the trw-memory daemon (pid {info.pid}) serves {info.version}, but this client "
            f"is {_package_version()}; their tool signatures differ. No memory was read or written.{not_replaced} "
            f"The user should stop the old daemon (process {info.pid}); restarting the MCP client alone leaves it "
            f"running. The next call then starts a daemon from this installation. {AGENT_MUST_NOT_STOP}"
        )

    def _attach(self) -> DaemonInfo:
        """Return a live daemon, auto-starting one only if the slot is free.

        Auto-start is gated on :class:`DiscoveryAbsent` specifically. An
        untrusted record is not an absent one: spawning on it would bind a
        second endpoint over a daemon that may still be serving.
        """
        result = read_live_discovery(self._paths)
        if isinstance(result, DaemonInfo) and majors_differ(result.version, _package_version()):
            pinned = self._instance is not None  # a pinned client refuses before any restart action
            why = replace_older_daemon(
                result, self._paths, self._config, self._token, pinned=pinned, mine=_package_version()
            )
            result = self._compatible(result, why) if why else read_live_discovery(self._paths)
        if isinstance(result, DaemonInfo):
            return self._compatible(result)
        if isinstance(result, DiscoveryInvalid):
            raise self._refuse_invalid(result)
        if not self._config.memory_daemon_autostart:  # PRD-CORE-310 FR04: the one spawn site honours it
            raise self._unreachable(f"{result.reason}, and auto-start is off (MEMORY_DAEMON_AUTOSTART=false)")
        spawned = (
            self._launcher(self._paths)
            if self._launcher is not None
            else launch_from_record(self._paths) or start_daemon_detached(self._paths)
        )
        deadline = time.monotonic() + self._config.memory_daemon_startup_timeout_seconds
        while time.monotonic() < deadline:
            result = read_live_discovery(self._paths)
            if isinstance(result, DaemonInfo):
                return self._compatible(result)
            if isinstance(result, DiscoveryInvalid):
                raise self._refuse_invalid(result)
            time.sleep(_DISCOVERY_POLL_SECONDS)
        reason = (
            f"auto-start did not publish a discovery file within {self._config.memory_daemon_startup_timeout_seconds}s"
        )
        if _stop_unpublished(spawned):
            reason += f"; the client stopped the daemon it started, whose stderr is in {self._paths.start_log}"
        raise self._unreachable(reason)

    async def retire(self) -> None:
        """Stop holding a session: close it now if idle, else when its last in-flight call returns.

        For an owner that replaced this client (a restarted daemon, a changed setting).
        A caller still holding it keeps working, on a session per call.
        """
        await self._sessions.retire()

    async def _call_once(self, info: DaemonInfo, name: str, arguments: dict[str, Any]) -> Any:
        if name in _direct.DIRECT_TOOLS:
            return await _direct.post_tool(info, self._token, _package_version(), name, arguments)
        if self._keep_session and not self._sessions.retired:
            held = await self._sessions.acquire(
                (info.url, info.pid, info.started_at), lambda: open_session(info, self._token, _package_version())
            )
            try:
                return (await held.client.call_tool(name, arguments)).data
            except Exception as exc:
                if not _direct.answered(exc):  # the daemon answering leaves the session fine
                    # A transport failure: the retry must not reuse this session. Released, not
                    # closed: another call may still be using it, and closes it when it returns.
                    await self._sessions.release(held)
                raise
            finally:
                await self._sessions.done_with(held)
        async with open_session(info, self._token, _package_version()) as client:
            return (await client.call_tool(name, arguments)).data

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a daemon-served tool, or fail closed.

        Args:
            name: Registered tool name, e.g. ``"memory_recall"``.
            arguments: Tool arguments.

        Returns:
            The tool's structured result.

        Raises:
            DaemonAuthError: The daemon rejected the token. Not retried, and
                nothing is re-minted.
            DaemonRecordInvalidError: ``daemon.json`` exists but cannot be
                trusted. Nothing was spawned and no file was rewritten.
            DaemonUnreachableError: The daemon could not be reached after the
                first attempt and its single retry, or a write that cannot be
                replayed lost its response after it was sent.
            ToolError: The daemon answered and the tool refused. Not retried.
        """
        arguments = dict(arguments or {})
        if name == "memory_store" and not arguments.get("entry_id"):
            arguments["entry_id"] = "M-" + uuid4().hex[:16]
        last_error: BaseException | None = None
        holder = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Re-resolve on the retry. NFR02 requires that a daemon killed
            # mid-request results in either a restarted daemon serving the retry
            # or a fail-closed error; reusing the first attempt's endpoint could
            # only ever produce the second, because the retry would dial the
            # dead port again. Re-attaching lets the auto-start in ``_attach``
            # do its job, so the one retry is a real second chance.
            info = self._attach()
            if self._instance is not None and (info.pid, info.started_at) != self._instance:
                raise self._unreachable(
                    f"daemon {info.pid} started {info.started_at} replaced {self._instance[0]} started "
                    f"{self._instance[1]}, which this client was checked against; attach again"
                )
            try:
                return await self._call_once(info, name, arguments)
            except Exception as exc:  # transport failures are classified immediately below
                if _direct.answered(exc) or isinstance(exc, DaemonProtocolError):
                    # The daemon answered: the tool itself refused (a namespace outside
                    # the grant, invalid input), or a direct read's reply was malformed.
                    # Retrying cannot change that answer, and a retry's "unreachable"
                    # would hide the reason.
                    raise
                if is_unauthorized(exc):
                    logger.warning("daemon_token_rejected_by_server", tool=name)
                    raise DaemonAuthError(
                        "the trw-memory daemon rejected this checkout's grant; nothing was re-minted. "
                        "Run `trw-mcp memory token` in this checkout, then retry."
                    ) from exc
                last_error = exc
                logger.warning(
                    "daemon_call_failed",
                    tool=name,
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=type(exc).__name__,
                )
                if name not in _REPLAYABLE_TOOLS and not never_sent(exc):
                    raise DaemonUnreachableError(
                        f"the connection to the trw-memory daemon failed after {name} was sent "
                        f"({type(exc).__name__}), so it may have been applied. It was not retried: "
                        f"check its effect before running it again."
                    ) from exc
                # A client pinned to one daemon (``instance``) cannot use a successor, so it does not wait for one.
                refused = never_sent(exc) and attempt < _MAX_ATTEMPTS and self._instance is None
                if refused and not await self._await_withdrawal(info):
                    remedy = info.stop_remedy(self._paths.discovery)
                    holder = f"; daemon {info.pid} holds the record but refuses {info.url}: {remedy}"
        reason = type(last_error).__name__ if last_error else "unknown error"
        raise self._unreachable(reason + holder) from last_error

    async def _await_withdrawal(self, refused: DaemonInfo) -> bool:
        """Wait until the record that refused a connection is withdrawn, replaced or dead; whether it was.

        A daemon draining after its idle window closes its socket before it withdraws
        its record, so a retry at once dials the same closed port. Waiting, bounded by
        the startup deadline, lets the retry attach to whatever serves next, a
        successor started under the claim lock included (PRD-CORE-310 FR02).
        """
        deadline = time.monotonic() + self._config.memory_daemon_startup_timeout_seconds
        while time.monotonic() < deadline:
            if withdrawn(self._paths, refused):
                return True
            await asyncio.sleep(_DISCOVERY_POLL_SECONDS)
        logger.warning("daemon_record_still_refusing", pid=refused.pid, url=refused.url)
        return False

    # Below this line, every method is a typed stub whose ``...`` body never
    # runs: ``_forward`` (FR04) replaces each one, by name, right after the
    # class body ends, deriving the served tool and the payload shape from the
    # stub's own signature. A method stays hand-written only when it carries
    # logic beyond "shape the arguments and name the tool" -- ``namespace_move``
    # picks its tool name from an argument, which no generic forwarder can express.

    async def store(self, content: str, namespace: str, **kwargs: Any) -> Any:
        """Write a memory entry through the daemon, or fail closed."""

    async def recall(self, query: str, namespace: str, **kwargs: Any) -> Any:
        """Read memory entries through the daemon, or fail closed.

        Failing closed on a READ is the deliberate part: an empty-but-truthful
        error beats a partial view the caller cannot tell is partial.
        """

    async def get(self, memory_id: str, namespace: str) -> Any:
        """Read one entry by ``(namespace, id)`` through the daemon, or fail closed."""

    async def find_duplicate(self, namespace: str, content: str, detail: str) -> Any:
        """Id of an ACTIVE exact-content copy in *namespace* through the daemon, or fail closed."""

    async def update(self, entry_id: str, namespace: str, patch: dict[str, Any]) -> Any:
        """Correct one entry with a ``memory_update`` *patch* through the daemon, or fail closed."""

    async def vectors(self, namespace: str, ids: list[str]) -> Any:
        """Active-space vectors of *ids* in *namespace*, with that space and its collapse threshold, or fail closed."""

    async def anchored(self, namespace: str, file: str, limit: int, status: str | None = None) -> Any:
        """*namespace*'s rows anchored to the repo-relative *file* in ``memory_recall``'s row shape, or fail closed."""

    async def verify(self, namespace: str, project_root: str | None, settings: dict[str, object] | None = None) -> Any:
        """Run the maintain-verify sweep over *namespace* against *project_root*, or fail closed."""

    async def assertion_health(self, namespace: str, stale_days: int) -> Any:
        """*namespace*'s cached assertion verdicts as counts (``health`` is ``None`` when none exist), or fail closed."""

    async def graph_related(
        self, namespace: str, learning_id: str, depth: int, edge_types: list[str] | None, limit: int
    ) -> Any:
        """*learning_id*'s active graph neighbours in *namespace*, or fail closed."""

    async def graph_backfill(
        self, namespace: str, after: dict[str, str] | None, limit: int, deadline_seconds: float | None
    ) -> Any:
        """Graph one page of *namespace*'s existing rows after the *after* cursor, or fail closed."""

    async def record_surfaced(self, namespace: str, ids: list[str], *, session_start: bool = False) -> Any:
        """Count *ids* of *namespace* as accessed (and surfaced at session start), or fail closed."""

    async def maintain(self, namespace: str, consolidation: dict[str, object] | None = None) -> Any:
        """Run the daemon's maintenance passes for *namespace* (decay, consolidation, verification, WAL).

        *consolidation* is the caller's project policy for the consolidation pass.
        """

    async def reembed(self, namespace: str, cursor: str | None = None) -> Any:
        """One bounded pass re-encoding *namespace*'s vectors outside the active space; resume with its ``cursor``."""

    async def import_checkout(self, namespace: str, source_path: str, ids: list[str]) -> Any:
        """Merge the checkout's project store at *source_path* into *namespace*; counts what it holds of *ids*."""

    async def similar(
        self, namespace: str, text: str, skip_threshold: float, merge_threshold: float, top_k: int = 10
    ) -> Any:
        """The daemon's skip/merge/store verdict for *text* in *namespace*, or fail closed."""

    async def list_page(
        self,
        namespace: str,
        limit: int,
        after: dict[str, str] | None,
        *,
        status: str | None = None,
        tags: list[str] | None = None,
        types: list[str] | None = None,
    ) -> Any:
        """One keyset page of *namespace*'s rows as ``MemoryEntry`` JSON, or fail closed. *types* is sent only when set."""

    async def status(self, namespace: str) -> Any:
        """Count *namespace*'s rows through the daemon, or fail closed."""

    async def sync_dirty_page(self, namespace: str, limit: int, cursor: str | None = None) -> Any:
        """The oldest *limit* rows of *namespace* that still need a push (behind *cursor*, ``"<sync_seq>:<id>"``, when given), or fail closed."""

    async def sync_mark_synced(self, namespace: str, acks: dict[str, int]) -> Any:
        """Mark pushed rows of *namespace* synced, each at the ``sync_seq`` it was paged at, or fail closed."""

    async def sync_find(self, namespace: str, remote_id: str, ids: list[str]) -> Any:
        """The row in *namespace* a pulled learning maps to, or fail closed."""

    async def sync_find_many(self, namespace: str, remote_ids: list[str], ids: list[str]) -> Any:
        """Every row of *namespace* a whole pulled page maps to, in one call, or fail closed."""

    async def sync_apply(
        self, namespace: str, entry: dict[str, Any], *, if_revision: str | None, synced: bool = True
    ) -> Any:
        """Write a merged pulled row into *namespace* over the revision it was read at, or fail closed."""

    async def sync_apply_many(self, namespace: str, items: list[dict[str, Any]]) -> Any:
        """Write a pulled page into *namespace*, each row over the revision it was read at, in one call, or fail closed."""

    async def search(self, namespace: str, **kwargs: Any) -> Any:
        """Filter one namespace's entries through the daemon, or fail closed."""

    async def forget(self, memory_id: str, namespace: str) -> Any:
        """Delete one entry through the daemon, or fail closed."""

    async def consolidate(self, namespace: str, *, dry_run: bool = False) -> Any:
        """Run one consolidation pass through the daemon, or fail closed."""

    async def namespace_diagnose(self, namespace: str | None) -> Any:
        """Report whether a checkout looks moved, through the daemon; read-only."""

    async def namespace_move(self, action: Literal["merge", "rename"], source: str, destination: str) -> Any:
        """Merge or rename one namespace's rows into another through the daemon."""
        return await self.call_tool(f"memory_namespace_{action}", {"source": source, "destination": destination})


#: Every stub method above whose body is generated by ``_forward`` (FR04). Kept as
#: one list rather than a decorator per method: mypy --strict type-checks each stub
#: exactly as written in the class body above, and applying ``_forward`` here,
#: after the class exists, changes nothing that mypy's static pass ever sees.
_FORWARDED_METHODS = (
    "store",
    "recall",
    "get",
    "find_duplicate",
    "update",
    "vectors",
    "anchored",
    "verify",
    "assertion_health",
    "graph_related",
    "graph_backfill",
    "record_surfaced",
    "maintain",
    "reembed",
    "import_checkout",
    "similar",
    "list_page",
    "status",
    "sync_dirty_page",
    "sync_mark_synced",
    "sync_find",
    "sync_find_many",
    "sync_apply",
    "sync_apply_many",
    "search",
    "forget",
    "consolidate",
    "namespace_diagnose",
)
for _name in _FORWARDED_METHODS:
    setattr(DaemonClient, _name, _forward(getattr(DaemonClient, _name)))
del _name
