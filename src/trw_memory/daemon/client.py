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
   lost fails at once, saying it may have been applied.
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

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any, Literal
from uuid import uuid4

import httpx
import structlog
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from trw_memory.daemon._discovery import DaemonInfo, DiscoveryInvalid, read_live_discovery
from trw_memory.daemon._paths import DaemonPaths, open_private_log
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

#: The auto-started daemon's argv after the interpreter: the module entry point,
#: so auto-start does not depend on the console script being on ``PATH``.
_DAEMON_ARGV = ("-m", "trw_memory.server", "serve", "http")

#: How long a daemon that never published gets to exit on SIGTERM before SIGKILL.
_STOP_GRACE_SECONDS = 2.0

#: HTTP status the daemon returns for a missing or wrong bearer token.
_UNAUTHORIZED_STATUS = 401

#: Tools a retry may repeat after the request was sent: a second run lands the
#: same way as the first (``memory_sync_apply`` rewrites the same id, synced). ``memory_store`` qualifies because ``call_tool``
#: fixes its ``entry_id`` before the first attempt, so a replay updates the row
#: the lost attempt wrote instead of adding a second one. ``memory_update`` (a
#: correction) sets values, ``tags_add`` dedups and a closed prior is skipped, so a
#: replay leaves the row as the first run did; only its audit event repeats.
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
        "memory_search",
        "memory_status",
        "memory_store",
        "memory_sync_apply",
        "memory_sync_dirty_page",
        "memory_sync_find",
        "memory_sync_mark_synced",
        "memory_update",
        "memory_vectors",
        "memory_similar",
    }
)


def start_daemon_detached(paths: DaemonPaths) -> subprocess.Popen[bytes]:
    """Spawn a daemon in its own session, detached from this process, and return it.

    Its stderr goes to :attr:`DaemonPaths.start_log`, emptied on each start, so a
    start that stalls or crashes before publishing leaves a reason behind. The
    daemon installs no log handlers, so stderr carries warnings and tracebacks only.
    """
    logger.info("daemon_auto_start", discovery=str(paths.discovery))
    log = open_private_log(paths.start_log)
    try:
        return subprocess.Popen(  # noqa: S603 -- fixed argv: this interpreter and a module constant
            [sys.executable, *_DAEMON_ARGV],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )
    finally:
        os.close(log)


def _stop_unpublished(spawned: object) -> bool:
    """Stop a daemon this client started that never published; whether one was stopped.

    A daemon stuck before ``serve()`` never reaches its idle timer, so leaving it
    means it lives until someone kills it, and the next call spawns another beside
    it. A stub that spawned nothing (tests pass ``lambda _paths: None``) is not a process.
    """
    if not isinstance(spawned, subprocess.Popen) or spawned.poll() is not None:
        return False
    spawned.terminate()
    try:
        spawned.wait(timeout=_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):  # it exited between the check and the kill
            spawned.kill()
        spawned.wait()
    logger.warning("daemon_auto_start_stopped", pid=spawned.pid)
    return True


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """*exc*, its causes and contexts, and the members of any exception group among them."""
    seen: set[int] = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        members = getattr(current, "exceptions", None)  # an exception group; Python 3.10 has no builtin
        if isinstance(members, tuple):
            pending.extend(member for member in members if isinstance(member, BaseException))


def _is_unauthorized(exc: BaseException) -> bool:
    """Whether *exc* (or a cause in its chain) is a 401 rejection."""
    return any(
        getattr(getattr(current, "response", None), "status_code", None) == _UNAUTHORIZED_STATUS
        for current in _chain(exc)
    )


def _never_sent(exc: BaseException) -> bool:
    """Whether *exc* failed while connecting, so the daemon never saw the request."""
    return any(isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout)) for current in _chain(exc))


class DaemonClient:
    """Calls daemon-served tools, failing closed when the daemon is absent."""

    def __init__(
        self,
        token: str,
        config: MemoryConfig | None = None,
        paths: DaemonPaths | None = None,
        *,
        instance: tuple[int, str] | None = None,
    ) -> None:
        """Args: token: the checkout's grant. config: source of the startup deadline. paths: daemon files.

        instance: the ``(pid, started_at)`` a caller checked; every call to any other daemon is refused.
        """
        self._token = token
        self._config = config or MemoryConfig()
        self._paths = paths or DaemonPaths.resolve()
        self._instance = instance

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
        spawned = start_daemon_detached(self._paths)
        deadline = time.monotonic() + self._config.memory_daemon_startup_timeout_seconds
        while time.monotonic() < deadline:
            result = read_live_discovery(self._paths)
            if isinstance(result, DaemonInfo):
                return result
            if isinstance(result, DiscoveryInvalid):
                raise self._refuse_invalid(result)
            time.sleep(_DISCOVERY_POLL_SECONDS)
        reason = (
            f"auto-start did not publish a discovery file within {self._config.memory_daemon_startup_timeout_seconds}s"
        )
        if _stop_unpublished(spawned):
            reason += f"; the client stopped the daemon it started, whose stderr is in {self._paths.start_log}"
        raise self._unreachable(reason)

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
                transport = StreamableHttpTransport(url=info.url, auth=self._token)
                async with Client(transport) as client:
                    result = await client.call_tool(name, arguments)
                return result.data
            except ToolError:
                # The daemon answered: the tool itself refused (a namespace outside
                # the grant, invalid input). Retrying cannot change that answer, and
                # reporting it as "unreachable" would hide the refusal's reason.
                raise
            except Exception as exc:  # transport failures are classified immediately below
                if _is_unauthorized(exc):
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
                if name not in _REPLAYABLE_TOOLS and not _never_sent(exc):
                    raise DaemonUnreachableError(
                        f"the connection to the trw-memory daemon failed after {name} was sent "
                        f"({type(exc).__name__}), so it may have been applied. It was not retried: "
                        f"check its effect before running it again."
                    ) from exc
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

    async def get(self, memory_id: str, namespace: str) -> Any:
        """Read one entry by ``(namespace, id)`` through the daemon, or fail closed."""
        return await self.call_tool("memory_get", {"memory_id": memory_id, "namespace": namespace})

    async def find_duplicate(self, namespace: str, content: str, detail: str) -> Any:
        """Id of an ACTIVE exact-content copy in *namespace* through the daemon, or fail closed."""
        return await self.call_tool(
            "memory_find_duplicate", {"namespace": namespace, "content": content, "detail": detail}
        )

    async def update(self, entry_id: str, namespace: str, patch: dict[str, Any]) -> Any:
        """Correct one entry with a ``memory_update`` *patch* through the daemon, or fail closed."""
        return await self.call_tool("memory_update", {"entry_id": entry_id, "namespace": namespace, "patch": patch})

    async def admit_shared(self, namespace: str, results: list[dict[str, object]]) -> Any:
        """Admit fetched shared results through *namespace*'s gate (not replayed: the gate quarantines), or fail closed."""
        return await self.call_tool("memory_admit_shared", {"namespace": namespace, "results": results})

    async def vectors(self, namespace: str, ids: list[str], space: dict[str, object]) -> Any:
        """Stored vectors of *ids* in *namespace* encoded in *space*, or fail closed."""
        return await self.call_tool("memory_vectors", {"namespace": namespace, "ids": ids, "space": space})

    async def verify(self, namespace: str, project_root: str | None, settings: dict[str, object] | None = None) -> Any:
        """Run the maintain-verify sweep over *namespace* against *project_root*, or fail closed."""
        return await self.call_tool(
            "memory_verify", {"namespace": namespace, "project_root": project_root, "settings": settings}
        )

    async def assertion_health(self, namespace: str, stale_days: int) -> Any:
        """*namespace*'s cached assertion verdicts as counts (``health`` is ``None`` when none exist), or fail closed."""
        return await self.call_tool("memory_assertion_health", {"namespace": namespace, "stale_days": stale_days})

    async def graph_related(
        self, namespace: str, learning_id: str, depth: int, edge_types: list[str] | None, limit: int
    ) -> Any:
        """*learning_id*'s active graph neighbours in *namespace*, or fail closed."""
        return await self.call_tool(
            "memory_graph_related",
            {
                "namespace": namespace,
                "learning_id": learning_id,
                "depth": depth,
                "edge_types": edge_types,
                "limit": limit,
            },
        )

    async def graph_backfill(
        self, namespace: str, after: dict[str, str] | None, limit: int, deadline_seconds: float | None
    ) -> Any:
        """Graph one page of *namespace*'s existing rows after the *after* cursor, or fail closed."""
        return await self.call_tool(
            "memory_graph_backfill",
            {"namespace": namespace, "after": after, "limit": limit, "deadline_seconds": deadline_seconds},
        )

    async def record_surfaced(self, namespace: str, ids: list[str], *, session_start: bool = False) -> Any:
        """Count *ids* of *namespace* as accessed (and surfaced at session start), or fail closed."""
        return await self.call_tool(
            "memory_record_surfaced", {"namespace": namespace, "ids": ids, "session_start": session_start}
        )

    async def maintain(self, namespace: str) -> Any:
        """Run the daemon's maintenance passes for *namespace* (decay, consolidation, verification, WAL)."""
        return await self.call_tool("memory_maintain", {"namespace": namespace})

    async def import_checkout(self, namespace: str, source_path: str, ids: list[str]) -> Any:
        """Merge the checkout's project store at *source_path* into *namespace*; counts what it holds of *ids*."""
        return await self.call_tool(
            "memory_import_checkout", {"namespace": namespace, "source_path": source_path, "ids": ids}
        )

    async def similar(self, namespace: str, vector: list[float], space: dict[str, Any] | None, top_k: int = 10) -> Any:
        """The dedup KNN window for *vector* (encoded in *space*) in *namespace*, or fail closed."""
        return await self.call_tool(
            "memory_similar", {"namespace": namespace, "vector": vector, "space": space, "top_k": top_k}
        )

    async def list_page(
        self,
        namespace: str,
        limit: int,
        after: dict[str, str] | None,
        *,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """One keyset page of *namespace*'s rows as ``MemoryEntry`` JSON, or fail closed."""
        arguments = {"namespace": namespace, "limit": limit, "after": after, "status": status, "tags": tags}
        return await self.call_tool("memory_list_page", arguments)

    async def status(self, namespace: str) -> Any:
        """Count *namespace*'s rows through the daemon, or fail closed."""
        return await self.call_tool("memory_status", {"namespace": namespace})

    async def sync_dirty_page(self, namespace: str, limit: int) -> Any:
        """The oldest *limit* rows of *namespace* that still need a push, or fail closed."""
        return await self.call_tool("memory_sync_dirty_page", {"namespace": namespace, "limit": limit})

    async def sync_mark_synced(self, namespace: str, acks: dict[str, int]) -> Any:
        """Mark pushed rows of *namespace* synced, each at the ``sync_seq`` it was paged at, or fail closed."""
        return await self.call_tool("memory_sync_mark_synced", {"namespace": namespace, "acks": acks})

    async def sync_find(self, namespace: str, remote_id: str, ids: list[str]) -> Any:
        """The row in *namespace* a pulled learning maps to, or fail closed."""
        return await self.call_tool("memory_sync_find", {"namespace": namespace, "remote_id": remote_id, "ids": ids})

    async def sync_apply(self, namespace: str, entry: dict[str, Any], *, synced: bool = True) -> Any:
        """Write a merged pulled row into *namespace* through the write gate, or fail closed."""
        return await self.call_tool("memory_sync_apply", {"namespace": namespace, "entry": entry, "synced": synced})

    async def search(self, namespace: str, **kwargs: Any) -> Any:
        """Filter one namespace's entries through the daemon, or fail closed."""
        return await self.call_tool("memory_search", {"namespace": namespace, **kwargs})

    async def forget(self, memory_id: str, namespace: str) -> Any:
        """Delete one entry through the daemon, or fail closed."""
        return await self.call_tool("memory_forget", {"memory_id": memory_id, "namespace": namespace})

    async def consolidate(self, namespace: str, *, dry_run: bool = False) -> Any:
        """Run one consolidation pass through the daemon, or fail closed."""
        return await self.call_tool("memory_consolidate", {"namespace": namespace, "dry_run": dry_run})

    async def namespace_diagnose(self, namespace: str | None) -> Any:
        """Report whether a checkout looks moved, through the daemon; read-only."""
        return await self.call_tool("memory_namespace_diagnose", {"namespace": namespace})

    async def namespace_move(self, action: Literal["merge", "rename"], source: str, destination: str) -> Any:
        """Merge or rename one namespace's rows into another through the daemon."""
        return await self.call_tool(f"memory_namespace_{action}", {"source": source, "destination": destination})
