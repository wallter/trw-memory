"""Machine-wide admission lock for wide pytest runs (test-perf P1e, heavy-suite admission).

The per-process xdist cap in ``conftest.py`` bounds ONE run. It cannot stop two agents
each starting a capped full suite at once, which stacks 2x the workers on one host and
is how load-sensitive timing tests went red. This lock admits one wide run at a time:

- A pytest CONTROLLER whose xdist worker count is above :data:`ADMISSION_WORKERS` takes
  an exclusive ``flock`` on :func:`lock_path` before collection. A run at ``-n 2`` or
  less, a serial run, and ``-p no:xdist`` never touch it.
- xdist workers (``PYTEST_XDIST_WORKER``) and any pytest launched from inside an admitted
  run (``HELD_ENV``, which the holder exports) skip it. Both would otherwise wait on
  their own parent forever.
- While another run holds it, the controller waits up to ``WAIT_ENV`` seconds (default
  :data:`DEFAULT_WAIT_SECONDS`), then fails with the holder's pid and command line, read
  from a sidecar file the holder writes. ``SKIP_ENV=1`` bypasses the lock entirely.
- ``flock`` belongs to the open file description, so the lock is released the moment the
  holding process dies, however it dies. The descriptor is not inheritable (PEP 446),
  so a surviving child never keeps it.

The path is fixed when this module is imported, i.e. when the conftest loads: before any
fixture redirects ``HOME``. ``LOCK_ENV`` overrides it.

This module is duplicated verbatim in trw-mcp/tests and trw-memory/tests, like the xdist
cap it sits beside: the two packages publish as separate mirrors and neither test tree may
import the other's.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

#: A controller above this many xdist workers needs admission.
ADMISSION_WORKERS = 2
LOCK_ENV = "TRW_PYTEST_ADMISSION_LOCK"
SKIP_ENV = "TRW_PYTEST_ADMISSION_SKIP"
WAIT_ENV = "TRW_PYTEST_ADMISSION_WAIT_SECONDS"
#: Exported by the holder (its pid), so a pytest it launches does not wait on it.
HELD_ENV = "TRW_PYTEST_ADMISSION_HELD"
#: Long enough to queue behind one full trw-mcp suite (~17 min at -n 4 on a loaded host).
DEFAULT_WAIT_SECONDS = 1800.0
_POLL_SECONDS = 0.5

#: The real home, captured at import (conftest load), before any fixture redirects HOME.
_DEFAULT_LOCK = Path.home() / ".cache" / "trw-pytest" / "heavy.lock"


class AdmissionRefused(RuntimeError):
    """Another wide run held the lock for the whole wait."""


def lock_path(env: Mapping[str, str] = os.environ) -> Path:
    override = env.get(LOCK_ENV)
    return Path(override) if override else _DEFAULT_LOCK


def holder_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".holder")


def needs_admission(numprocesses: object, env: Mapping[str, str] = os.environ) -> bool:
    """Whether this pytest process must take the lock.

    ``numprocesses`` is ``config.option.numprocesses``: ``None`` without ``-n``, an int,
    or the string ``"auto"``/``"logical"`` when xdist sizes itself (always wide).
    """
    if env.get("PYTEST_XDIST_WORKER") or env.get(HELD_ENV) or env.get(SKIP_ENV) == "1":
        return False
    if isinstance(numprocesses, str):
        return True
    return isinstance(numprocesses, int) and numprocesses > ADMISSION_WORKERS


def wait_seconds(env: Mapping[str, str] = os.environ) -> float:
    raw = env.get(WAIT_ENV, "")
    try:
        value = float(raw) if raw else DEFAULT_WAIT_SECONDS
    except ValueError:
        return DEFAULT_WAIT_SECONDS
    return max(value, 0.0)


def read_holder(lock: Path) -> str:
    """The recorded holder as ``pid N: <command line>``, or a placeholder when unrecorded."""
    try:
        data = json.loads(holder_path(lock).read_text(encoding="utf-8"))
        return f"pid {int(data['pid'])}: {data['cmdline']}"
    except (OSError, ValueError, KeyError, TypeError):  # trw-fail-silent-allow: the message still names the lock
        return "an unrecorded holder"


def acquire(lock: Path, *, wait: float, cmdline: str, poll: float = _POLL_SECONDS) -> int:
    """Take the lock, waiting up to *wait* seconds; return the held descriptor.

    Raises :class:`AdmissionRefused` naming the holder when the wait runs out.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + wait
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise AdmissionRefused(
                    f"another wide pytest run holds {lock} ({read_holder(lock)}); waited {wait:g} s. "
                    f"Wait for it, or set {SKIP_ENV}=1 to run beside it anyway"
                ) from None
            if not announced:
                print(f"pytest admission: waiting for {lock} ({read_holder(lock)})", file=sys.stderr, flush=True)
                announced = True
            time.sleep(poll)
    record = {"pid": os.getpid(), "cmdline": cmdline, "started": time.time()}
    sidecar = holder_path(lock)
    staging = sidecar.with_name(f"{sidecar.name}.{os.getpid()}")
    staging.write_text(json.dumps(record), encoding="utf-8")
    os.replace(staging, sidecar)
    return fd


def release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def admit(numprocesses: object, env: dict[str, str] | None = None) -> int | None:
    """The controller's entry point: take the lock when needed and mark the env as held.

    Returns the held descriptor (release it with :func:`release`), or ``None`` when this
    process needs no admission. *env* defaults to ``os.environ``, which xdist workers and
    any nested pytest inherit.
    """
    target = os.environ if env is None else env
    if not needs_admission(numprocesses, target):
        return None
    fd = acquire(lock_path(target), wait=wait_seconds(target), cmdline=" ".join([sys.executable, *sys.argv]))
    target[HELD_ENV] = str(os.getpid())
    return fd


def dismiss(fd: int, env: dict[str, str] | None = None) -> None:
    """Undo :func:`admit`: clear this process's held marker, then release the lock."""
    target = os.environ if env is None else env
    if target.get(HELD_ENV) == str(os.getpid()):
        del target[HELD_ENV]
    release(fd)
