"""The daemon discovery record -- PRD-CORE-253 FR03 property 2.

``<user_memory_dir>/daemon.json`` is how a client finds the daemon: nothing
hardcodes a port, because the daemon asks the operating system for an ephemeral
one by default. The record carries no credential (PRD-CORE-298 FR02) and is
still written through the hardened 0600 path in :mod:`trw_memory.daemon._paths`.

Every read is defensive, but defensive is not the same as permissive, and a
read answers one of THREE things rather than two:

``DaemonInfo``
    a record this build understands, naming a process. The only answer that
    authorises attaching.
``DiscoveryAbsent``
    no record, or one naming a process that is gone. The only answer that
    authorises starting a daemon; a dead pid is reapable under the claim lock.
``DiscoveryInvalid``
    a record that is unreadable, malformed, or from a schema this build does
    not know. It is evidence of NOTHING. Folding it into "there is no daemon"
    -- which this module used to do -- is what let a second daemon bind a fresh
    port and overwrite the record while the first was still serving, leaving
    two writers on one ``memory.db``. Deciding callers refuse and name the file.

A partially trusted endpoint is never returned either: a client that connected
to a stale URL would hang, or reach a *different* process that inherited the
port.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from pydantic import BaseModel, Field, field_validator

from trw_memory.daemon._paths import DaemonPaths, read_secret_file, write_secret_file
from trw_memory.exceptions import DaemonSecretUnreadableError
from trw_memory.storage._pid_liveness import is_process_live, process_start

__all__ = [
    "DAEMON_CAPABILITIES",
    "DISCOVERY_SCHEMA_VERSION",
    "DRAINING_HEADER",
    "DRAINING_STATUS",
    "DRAIN_CAPABILITY",
    "VERSION_HEADER",
    "DaemonInfo",
    "DiscoveryAbsent",
    "DiscoveryInvalid",
    "DiscoveryRead",
    "read_discovery_result",
    "read_live_discovery",
    "refused_while_draining",
    "write_discovery",
]

logger = structlog.get_logger(__name__)

#: Schema generation of ``daemon.json``. A client that reads a record it does
#: not understand treats the daemon as absent rather than guessing, so bumping
#: this is a safe way to make older clients re-start a daemon they can talk to.
DISCOVERY_SCHEMA_VERSION = 1

#: The header a ``DaemonClient`` carries its trw-memory version in (PLAN W45).
#: Defined here, beside the rest of what a client needs to reach the daemon, so
#: the client never imports the server-side gate that enforces it: that module
#: pulls in ``fastmcp.server``, which cost every edit hook ~0.2 s at import.
VERSION_HEADER = "x-trw-memory-version"

#: Marks the 503 a draining daemon answers at its door, before the app sees the request: the call was never
#: applied, so a client may retry it on the successor (DRAIN-503-NEVER-SENT). An unmarked 503 (the MCP
#: session manager's own, a pre-5.0.1 daemon's) proves nothing and keeps the "may have been applied" answer.
DRAINING_HEADER = "x-trw-memory-draining"
DRAINING_STATUS = 503


#: JSON-RPC requests an MCP session sends before any tool runs. A refusal on one of these, or on the
#: ``tools/call`` itself, applied nothing. mcp's ``ClientSession.call_tool`` sends ``tools/list`` AFTER a
#: successful call whose output schema it has not cached, so a refused ``tools/list`` (or any other
#: request) may follow an applied write (DRAIN-503-REQUEST-SCOPE).
_BEFORE_ANY_EFFECT = frozenset({"initialize", "notifications/initialized", "tools/call"})


def _refused_method(request: object) -> str | None:
    """The JSON-RPC method of the refused *request*, or None when its body does not say."""
    try:
        body = json.loads(getattr(request, "content", b"") or b"null")
    except (
        ValueError,
        RuntimeError,
    ):  # trw-fail-silent-allow: an unreadable body proves nothing, so the refusal stays "may have been applied"
        return None
    method = body.get("method") if isinstance(body, dict) else None
    return method if isinstance(method, str) else None


def refused_while_draining(error: BaseException) -> bool:
    """Whether *error* is a draining daemon's marked door refusal of a request that precedes any tool effect."""
    response = getattr(error, "response", None)
    marked = getattr(response, "status_code", None) == DRAINING_STATUS and DRAINING_HEADER in getattr(
        response, "headers", {}
    )
    return marked and _refused_method(getattr(error, "request", None)) in _BEFORE_ANY_EFFECT


#: The drain handshake: ``memory_drain`` finishes in-flight calls, withdraws the record and exits.
DRAIN_CAPABILITY = "drain"
#: The drain tool's name. A wire constant beside its capability, so the client side (``_upgrade``) can
#: name it without importing ``_drain``, whose server code loads fastmcp (PRD-CORE-333 S3b).
DRAIN_TOOL = "memory_drain"
#: What a daemon of this build advertises in its record, so a client can tell before it calls.
DAEMON_CAPABILITIES: tuple[str, ...] = (DRAIN_CAPABILITY,)


class DaemonInfo(BaseModel):
    """A running daemon's advertised endpoint."""

    schema_version: int = Field(default=DISCOVERY_SCHEMA_VERSION, description="Discovery record generation")
    pid: int = Field(gt=0, description="Process id of the serving daemon")
    url: str = Field(description="Loopback MCP endpoint, e.g. http://127.0.0.1:41234/mcp")
    started_at: str = Field(description="ISO-8601 UTC timestamp of the bind")
    version: str = Field(description="trw-memory version serving this endpoint")
    process_start: str | None = Field(
        default=None, description="The daemon's OS process start (PRD-CORE-310 FR01); absent from 4.0 records"
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Handshakes this daemon serves, e.g. 'drain' (DAEMON-AUTO-RESTART-ON-UPGRADE). Absent from 5.0.0 and "
            "older records, which read as none; an older reader ignores the field, so the schema stays 1."
        ),
    )

    @field_validator("url")
    @classmethod
    def _url_must_be_loopback(cls, value: str) -> str:
        """Refuse any URL a forged record could name that the daemon itself never emits.

        The only production writer, ``endpoint_url()``, always emits
        ``http://127.0.0.1:<port>/mcp`` (PRD-FIX-157-FR01), so this rejects
        nothing today's code produces -- only a record an attacker with
        same-user filesystem access could plant. ``urlsplit`` is used instead
        of a regex so scheme and host are parsed the same way a client would
        resolve them, and userinfo (``user@host``) never contributes to the
        host check: ``.hostname`` is always the part after ``@``.

        Refused, deliberately, rather than accepted:
          - any scheme other than exactly ``http``/``https`` (blocks
            ``file:``, and blocks scheme confusion generally);
          - a host that fails to parse at all (``urlsplit`` raising, or an
            empty/`` None`` hostname) -- treated as untrusted, not as
            "no opinion";
          - a malformed port (``.port`` raises ``ValueError`` for a
            non-numeric or out-of-range port) -- refused rather than ignored,
            since a malformed port is exactly the kind of thing a forged
            record would carry;
          - any hostname that is not exactly one of the three trusted forms
            once IPv6 brackets are stripped and case is folded: the fold is
            applied only for the ``localhost`` hostname comparison (DNS names
            are case-insensitive; ``LOCALHOST`` is refused only because the
            comparison target is lower-cased, not because case is otherwise
            ignored), and no fold or fuzzy match is applied to
            ``127.0.0.1``/``::1`` -- ``127.1`` (a valid but non-canonical
            shorthand some resolvers accept for ``127.0.0.1``) is refused
            because it is not the literal string ``127.0.0.1``;
          - ``127.0.0.1.evil.com`` (a suffix trick) and ``127.0.0.1@evil.com``
            (a userinfo trick) are both refused by the same mechanism: the
            hostname parsed by ``urlsplit`` for the first is the whole
            ``127.0.0.1.evil.com`` string (not a member of the trusted set),
            and for the second is ``evil.com`` (the part after ``@``).
        """
        try:
            parts = urlsplit(value)
        except ValueError as exc:
            raise ValueError(f"daemon discovery URL is not a parseable URL: {value!r} ({exc})") from exc
        if parts.scheme not in {"http", "https"}:
            raise ValueError(f"daemon discovery URL scheme must be http or https, got {parts.scheme!r} in {value!r}")
        try:
            hostname = parts.hostname
        except ValueError as exc:
            raise ValueError(f"daemon discovery URL host is not parseable: {value!r} ({exc})") from exc
        if not hostname:
            raise ValueError(f"daemon discovery URL has no host: {value!r}")
        try:
            _ = parts.port
        except ValueError as exc:
            raise ValueError(f"daemon discovery URL has a malformed port: {value!r} ({exc})") from exc
        candidate = hostname.strip("[]")
        if candidate.lower() == "localhost":
            candidate = "localhost"
        if candidate not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError(
                f"daemon discovery URL host must be one of 127.0.0.1, ::1, or localhost, got {hostname!r} in {value!r}"
            )
        return value

    def is_live(self, lock_file: Path) -> bool:
        """Whether the recorded process is still running: THE liveness decision for a daemon record.

        The pid must run and not be a zombie, and, when the record carries the
        daemon's start, the process now at that pid must have started then. A
        record outlives its daemon across a crash or a reboot, and a pid reused by
        an unrelated process answers ``kill(pid, 0)``: read as live, it held the
        slot so no successor could claim it (PRD-CORE-310 FR01). *lock_file* is
        the mtime fallback on platforms without ``/proc`` or signals.
        """
        return is_process_live(self.pid, self.process_start, lock_file)

    def stop_remedy(self, discovery: Path) -> str:
        """How the USER clears a live holder that serves nothing: stop only a pid its start proves is the daemon.

        A 4.0 record carries no start, and its pid may now be any process (a reboot reuses pids).
        """
        if self.process_start is not None:
            return f"the user can stop process {self.pid} (the old daemon). {AGENT_MUST_NOT_STOP}"
        return (
            f"if pid {self.pid} is not trw_memory.server, the user can remove {discovery}; otherwise the user can "
            f"stop process {self.pid}. {AGENT_MUST_NOT_STOP}"
        )


#: Every remedy that names a process to stop is addressed to the USER. An agent that reads it must not act on it:
#: CONSTITUTION HB-2 (agents never kill a process they did not start). An opencode agent once ran ``kill <pid>``
#: from the earlier imperative wording, stopped only by a bash:ask permission (DoD-5 run, 2026-09-26).
AGENT_MUST_NOT_STOP = "Agents must not stop or remove it themselves; report this to the user."


@dataclass(frozen=True)
class DiscoveryAbsent:
    """Nobody holds the daemon slot; starting one is safe.

    Covers "no file" and "a file naming a process that is gone" alike -- the
    second is reapable under the claim lock, so a caller deciding whether to
    start gets the same answer. *reason* keeps them apart for diagnostics.
    """

    reason: str = "no discovery record"


@dataclass(frozen=True)
class DiscoveryInvalid:
    """A record exists that cannot be trusted -- and cannot be dismissed.

    The distinction from :class:`DiscoveryAbsent` is the whole point of this
    type. An unreadable, malformed or schema-mismatched record says nothing
    about whether a daemon is serving, so a caller that reads it as "no daemon"
    binds a second port and overwrites the record while the first daemon is
    live: two writers on one ``memory.db``. Every deciding caller refuses here.
    """

    path: Path
    reason: str


#: The three answers a discovery read can support. Only :class:`DaemonInfo`
#: authorises attaching; only :class:`DiscoveryAbsent` authorises starting.
DiscoveryRead = DaemonInfo | DiscoveryAbsent | DiscoveryInvalid


#: The record THIS process published, so its answers can say which daemon gave them.
_published: DaemonInfo | None = None


def this_daemon() -> tuple[int, str] | None:
    """This process's published ``(pid, started_at)``, or ``None`` when it is not a daemon."""
    return (_published.pid, _published.started_at) if _published is not None else None


def offers_drain() -> bool:
    """Whether this build's daemon serves the drain handshake (read at call time: tests override it)."""
    return DRAIN_CAPABILITY in DAEMON_CAPABILITIES


def write_discovery(paths: DaemonPaths, *, url: str, version: str, drain_key: bool = False) -> DaemonInfo:
    """Write the discovery record for THIS process at mode 0600; ``drain`` is advertised only with a *drain_key*."""
    global _published
    info = _published = DaemonInfo(
        pid=os.getpid(),
        url=url,
        started_at=datetime.now(timezone.utc).isoformat(),
        version=version,
        process_start=process_start(os.getpid()),
        capabilities=[c for c in DAEMON_CAPABILITIES if c != DRAIN_CAPABILITY or drain_key],
    )
    write_secret_file(paths.discovery, info.model_dump_json())
    logger.info("daemon_discovery_written", path=str(paths.discovery), pid=info.pid, url=url)
    return info


def read_discovery_result(paths: DaemonPaths) -> DiscoveryRead:
    """Read ``daemon.json`` and say which of the three answers it supports."""
    try:
        raw = read_secret_file(paths.discovery)
    except DaemonSecretUnreadableError as exc:
        return _invalid(paths, str(exc), "daemon_discovery_unreadable")
    if raw is None:
        return DiscoveryAbsent()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _invalid(paths, f"the record is not valid JSON ({exc})", "daemon_discovery_malformed")
    if not isinstance(payload, dict):
        return _invalid(paths, "the record is not a JSON object", "daemon_discovery_malformed")
    if payload.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        return _invalid(
            paths,
            f"the record is schema {payload.get('schema_version')!r}, and this build writes "
            f"schema {DISCOVERY_SCHEMA_VERSION}",
            "daemon_discovery_schema_mismatch",
        )
    try:
        return DaemonInfo.model_validate(payload)
    except ValueError as exc:
        return _invalid(paths, f"the record failed field validation ({exc})", "daemon_discovery_invalid")


def _invalid(paths: DaemonPaths, reason: str, event: str) -> DiscoveryInvalid:
    """Build the untrusted-record answer, logging it once where it is decided."""
    logger.warning(event, path=str(paths.discovery), reason=reason)
    return DiscoveryInvalid(path=paths.discovery, reason=reason)


def read_live_discovery(paths: DaemonPaths) -> DiscoveryRead:
    """Read the record and fold liveness into the same three answers.

    A record naming a process that is gone answers :class:`DiscoveryAbsent`:
    the slot is reapable under the claim lock, so for a caller deciding whether
    to start a daemon it is the same answer as no file at all. An INVALID
    record is not folded away -- nothing about it says the slot is free.
    """
    result = read_discovery_result(paths)
    if isinstance(result, DaemonInfo) and not result.is_live(paths.lock):
        logger.info("daemon_discovery_stale", path=str(paths.discovery), pid=result.pid)
        return DiscoveryAbsent(reason=f"the record names pid {result.pid}, which is no longer running")
    return result
