"""The daemon discovery record -- PRD-CORE-253 FR03 property 2.

``<user_memory_dir>/daemon.json`` is how a client finds the daemon: nothing
hardcodes a port, because the daemon asks the operating system for an ephemeral
one by default. The record is a secret (it carries the bearer token) and is
written through the hardened 0600 path in :mod:`trw_memory.daemon._paths`.

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

import structlog
from pydantic import BaseModel, Field

from trw_memory.daemon._paths import DaemonPaths, read_secret_file, write_secret_file
from trw_memory.exceptions import DaemonSecretUnreadableError
from trw_memory.storage._writer_registry import _pid_is_live

__all__ = [
    "DISCOVERY_SCHEMA_VERSION",
    "DaemonInfo",
    "DiscoveryAbsent",
    "DiscoveryInvalid",
    "DiscoveryRead",
    "read_discovery",
    "read_discovery_result",
    "read_live_discovery",
    "write_discovery",
]

logger = structlog.get_logger(__name__)

#: Schema generation of ``daemon.json``. A client that reads a record it does
#: not understand treats the daemon as absent rather than guessing, so bumping
#: this is a safe way to make older clients re-start a daemon they can talk to.
DISCOVERY_SCHEMA_VERSION = 1


class DaemonInfo(BaseModel):
    """A running daemon's advertised endpoint."""

    schema_version: int = Field(default=DISCOVERY_SCHEMA_VERSION, description="Discovery record generation")
    pid: int = Field(gt=0, description="Process id of the serving daemon")
    url: str = Field(description="Loopback MCP endpoint, e.g. http://127.0.0.1:41234/mcp")
    token: str = Field(description="Bearer token every request must carry; never logged")
    started_at: str = Field(description="ISO-8601 UTC timestamp of the bind")
    version: str = Field(description="trw-memory version serving this endpoint")

    def __repr__(self) -> str:
        """Redact the token so no diagnostic path can print it (NFR03)."""
        return f"DaemonInfo(pid={self.pid}, url={self.url!r}, started_at={self.started_at!r}, version={self.version!r})"

    __str__ = __repr__

    def is_live(self, lock_file: Path) -> bool:
        """Whether the recorded process is still running.

        Reuses the package's existing liveness predicate rather than adding a
        third one (FR03 property 4); *lock_file* is the mtime fallback it uses
        on platforms without ``/proc`` or POSIX signals.
        """
        return _pid_is_live(self.pid, lock_file)


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


def write_discovery(paths: DaemonPaths, *, url: str, token: str, version: str) -> DaemonInfo:
    """Write the discovery record for THIS process at mode 0600."""
    info = DaemonInfo(
        pid=os.getpid(),
        url=url,
        token=token,
        started_at=datetime.now(timezone.utc).isoformat(),
        version=version,
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


def read_discovery(paths: DaemonPaths) -> DaemonInfo | None:
    """Return the recorded daemon, or ``None`` when no record can be trusted.

    A PROBE, for read-only diagnostics that report on the record and act on
    nothing. It collapses "absent" and "invalid" into ``None``, which is
    exactly the conflation that must never reach a caller deciding whether to
    bind a port, spawn a daemon or delete a file -- those use
    :func:`read_discovery_result` and refuse on
    :class:`DiscoveryInvalid`.
    """
    result = read_discovery_result(paths)
    return result if isinstance(result, DaemonInfo) else None


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
