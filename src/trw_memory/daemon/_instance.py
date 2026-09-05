"""Single-instance claim and stale-record reaping -- FR03 property 4, NFR02.

Two daemons on one store would defeat the whole point of the daemon, so a start
is a claim, not a hope. The claim is a read-modify-write over the discovery
record, serialised by the package's existing advisory file lock
(``lock_for_rmw`` at ``<user_memory_dir>/daemon.lock``) so no third locking
mechanism enters the tree:

1. take the lock;
2. read the recorded daemon. If its pid is **alive**, release and refuse --
   without binding a port and without touching the discovery file. If the
   record exists but is **untrusted**, refuse the same way: it is not evidence
   the slot is free, and claiming on no evidence is how a second daemon binds
   over a live one;
3. if the record names a **dead** pid, reap it. Liveness is the package's
   existing predicate, so reaping is conditional on evidence, never on age;
4. bind the loopback socket while still holding the lock, so two simultaneous
   starts cannot both get past step 2; and
5. write the discovery record naming this pid and the bound port, then release.

Ownership is the discovery record rather than a held file lock, because the
lock has to be released for the daemon's whole serving life anyway (a client
generating its first token takes the same lock). The record is strictly better
evidence: it carries the pid that liveness is checked against.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import structlog

from trw_memory.daemon._discovery import (
    DaemonInfo,
    DiscoveryInvalid,
    read_discovery_result,
    write_discovery,
)
from trw_memory.daemon._loopback import bind_loopback_socket, endpoint_url
from trw_memory.daemon._paths import DaemonPaths
from trw_memory.exceptions import DaemonAlreadyRunningError, DaemonRecordInvalidError
from trw_memory.storage.persistence import lock_for_rmw

__all__ = ["InstanceClaim", "claim_single_instance", "release_single_instance"]

logger = structlog.get_logger(__name__)


@dataclass
class InstanceClaim:
    """A won single-instance claim: the bound socket and what was advertised."""

    paths: DaemonPaths
    sock: socket.socket
    info: DaemonInfo


def claim_single_instance(paths: DaemonPaths, *, port: int, token: str, version: str) -> InstanceClaim:
    """Claim sole ownership of the store's daemon slot and bind its socket.

    Args:
        paths: Resolved daemon file locations.
        port: TCP port to bind, or 0 for an operating-system assignment.
        token: The per-user bearer token to advertise.
        version: trw-memory version string to advertise.

    Returns:
        The claim, holding the listening socket and the written record.

    Raises:
        DaemonAlreadyRunningError: If a live daemon already holds the claim.
            Nothing was bound and the discovery file was not modified.
        DaemonRecordInvalidError: If a record exists that cannot be trusted.
            Nothing was bound and the discovery file was not modified.
        ConfigError: If the bind address is not loopback.
    """
    with lock_for_rmw(paths.lock_anchor):
        existing = read_discovery_result(paths)
        if isinstance(existing, DiscoveryInvalid):
            # Step 2 has no answer, so there is no step 3. An untrusted record
            # is not evidence the slot is free: binding here would publish a
            # second endpoint over a daemon that may well still be serving.
            logger.warning("daemon_start_refused_invalid_record", path=str(existing.path), reason=existing.reason)
            raise DaemonRecordInvalidError(
                f"refusing to start: {existing.path} exists but cannot be trusted -- {existing.reason}. "
                f"It may name a daemon that is still serving this store, and starting a second one "
                f"would put two writers on {paths.store}. Inspect the file and remove it if no daemon "
                f"is running, then retry."
            )
        if isinstance(existing, DaemonInfo) and existing.pid != os.getpid():
            if existing.is_live(paths.lock):
                logger.info("daemon_start_refused_already_running", pid=existing.pid, url=existing.url)
                raise DaemonAlreadyRunningError(
                    f"a memory daemon is already running (pid {existing.pid}) at {existing.url}; "
                    f"its record is {paths.discovery}"
                )
            logger.info("daemon_stale_record_reaped", pid=existing.pid, path=str(paths.discovery))
            paths.discovery.unlink(missing_ok=True)

        sock = bind_loopback_socket(port)
        try:
            info = write_discovery(paths, url=endpoint_url(sock), token=token, version=version)
        except BaseException:
            sock.close()
            raise
    return InstanceClaim(paths=paths, sock=sock, info=info)


def release_single_instance(paths: DaemonPaths) -> None:
    """Remove this process's discovery record, if it is still ours.

    Called on idle shutdown and on any exit path. The pid check keeps a slow
    exit from deleting a *successor* daemon's record: whoever reaped us has
    already written their own, and this must not undo it. A record we cannot
    read is left alone for the same reason -- deleting on no evidence is how a
    successor's record disappears.
    """
    with lock_for_rmw(paths.lock_anchor):
        existing = read_discovery_result(paths)
        if isinstance(existing, DaemonInfo) and existing.pid == os.getpid():
            paths.discovery.unlink(missing_ok=True)
            logger.info("daemon_discovery_removed", path=str(paths.discovery), pid=existing.pid)
