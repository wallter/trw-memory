"""Which interpreter starts a store's daemon: the launcher record (DAEMON-AUTOSTART-VERSION-RACE).

After a daemon drain, the FIRST memory client to call starts the replacement, and left alone it starts it from its
OWN interpreter: on 2026-09-30 a repo-``.venv`` client (an older trw-memory) won that race and published an older
daemon beside the 8.1.2 stable server. ``<user_memory_dir>/launcher.json`` names the interpreter that serves this
store (and its ``PYTHONPATH`` for a source worktree), written by whoever points the store at one (``trw-mcp swap``)
through :func:`write_launcher_record`. ``DaemonClient`` starts the daemon from it, whichever client it is
(trw-mcp, the ``trw-memory`` CLI), so no client reads another package's records.

Fail closed: a record whose interpreter is gone, or no longer serves the version the record names, or that cannot
be read, refuses the start naming the fix. Starting from the client's own interpreter instead is the bug being
closed. A store with NO record is one nobody pointed at an interpreter: the client's own start stands.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.daemon._paths import DaemonPaths, read_secret_file, write_secret_file
from trw_memory.daemon._spawn import SpawnedDaemon, start_daemon_detached
from trw_memory.daemon._versions import is_older
from trw_memory.exceptions import DaemonSecretUnreadableError, DaemonUnreachableError
from trw_memory.user_paths import user_memory_dir_path

__all__ = [
    "LauncherRecord",
    "launch_from_record",
    "read_launcher_record",
    "register_launcher_record",
    "write_launcher_record",
]

#: The interpreter-selecting variables a record decides; the client's shell never does.
_RECORD_DECIDES = ("PYTHONPATH", "PYTHONHOME", "TRW_USER_DIR")
_PROBE = "from trw_memory._version import __version__; print(__version__)"
_PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class LauncherRecord:
    """The interpreter (and source path) that starts a store's daemon, and the trw-memory it must serve."""

    python: str
    version: str
    written_at: str
    pythonpath: str | None = None


def write_launcher_record(paths: DaemonPaths, python: Path, version: str, *, pythonpath: str | None = None) -> None:
    """Record that *python* (with *pythonpath*, when serving a source tree) starts this store's daemon."""
    record = LauncherRecord(str(python), version, datetime.now(timezone.utc).isoformat(), pythonpath or None)
    write_secret_file(paths.launcher, json.dumps(record.__dict__, sort_keys=True))


def register_launcher_record(paths: DaemonPaths, python: Path, version: str, *, pythonpath: str | None = None) -> bool:
    """Write the record for a server that IS *python* serving *version*, unless the store's names this or newer.

    A server calls this at boot, so a store pointed at an interpreter before records existed (or by an older
    build) gets one without waiting for the next ``swap``. A record that cannot be read is replaced: the server
    knows its own interpreter, which is the very fact the unreadable file failed to state. Returns whether it wrote.
    """
    try:
        current = read_launcher_record(paths)
    except DaemonUnreachableError:  # trw-fail-silent-allow: an unreadable record is replaced by the server's own facts
        current = None
    if current is not None and not is_older(current.version, version):
        return False
    write_launcher_record(paths, python, version, pythonpath=pythonpath)
    return True


def read_launcher_record(paths: DaemonPaths) -> LauncherRecord | None:
    """The store's record, ``None`` when it has none; a record that cannot be read or understood raises."""
    try:
        raw = read_secret_file(paths.launcher)
        if raw is None:
            return None
        data = json.loads(raw)
        pythonpath = data.get("pythonpath")
        if not all(isinstance(data.get(k), str) and data[k] for k in ("python", "version", "written_at")) or not (
            pythonpath is None or isinstance(pythonpath, str)
        ):
            raise ValueError("a field is missing or not a string")
        return LauncherRecord(data["python"], data["version"], data["written_at"], pythonpath)
    except (ValueError, AttributeError, DaemonSecretUnreadableError) as exc:
        raise DaemonUnreachableError(
            f"{paths.launcher} cannot be read ({type(exc).__name__}), so no memory daemon was started: this client's "
            f"own interpreter might not be the one that serves the store. Repair or remove the file, or point the "
            f"store at an interpreter again with `trw-mcp swap --python <path>`."
        ) from exc


def launch_from_record(paths: DaemonPaths) -> SpawnedDaemon | None:
    """Start the store's daemon from its recorded interpreter; ``None`` when the store has no record."""
    record = read_launcher_record(paths)
    if record is None:
        return None
    served = _served_version(record)
    if served != record.version:
        raise DaemonUnreachableError(
            f"{paths.launcher} names interpreter {record.python} for trw-memory {record.version}, but it "
            f"{'does not run' if served is None else f'serves {served}'}, so no memory daemon was started from this "
            f"client's own interpreter. Point the store at a current one: `trw-mcp swap --python <path>`."
        )
    return start_daemon_detached(paths, python=record.python, environ=_environ(paths, record))


def _served_version(record: LauncherRecord) -> str | None:
    """The trw-memory version the recorded interpreter serves, ``None`` when it is gone or cannot say."""
    env = {k: v for k, v in os.environ.items() if k not in _RECORD_DECIDES}
    if record.pythonpath:
        env["PYTHONPATH"] = record.pythonpath
    try:
        shown = subprocess.run(  # noqa: S603 -- the interpreter is the one the user's swap recorded
            [record.python, "-B", "-c", _PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # trw-fail-silent-allow: None is "stale", and the caller refuses
        return None
    return shown.stdout.strip() if shown.returncode == 0 else None


def _environ(paths: DaemonPaths, record: LauncherRecord) -> dict[str, str]:
    """The client's environment with the interpreter-selecting variables replaced by the record's."""
    child = {k: v for k, v in os.environ.items() if k not in _RECORD_DECIDES}
    if record.pythonpath:
        child["PYTHONPATH"] = record.pythonpath
    if user_memory_dir_path(child).resolve() != paths.user_memory_dir.resolve():  # not the default store
        child["TRW_USER_DIR"] = str(paths.user_memory_dir.parent)
    return child
