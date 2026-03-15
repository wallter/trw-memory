"""Atomic YAML/JSONL read/write with advisory file locks.

All file-level persistence goes through this module.  Writes are atomic
(write to temp file, then rename) to prevent corruption on interrupts.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path

import structlog
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from trw_memory.exceptions import StorageError

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Internal YAML factory
# ---------------------------------------------------------------------------


def _new_yaml() -> YAML:
    """Create a fresh, thread-safe YAML instance.

    ruamel.yaml's YAML class maintains internal emitter state that is
    NOT thread-safe.  Creating a fresh instance per operation prevents
    concurrent write corruption.
    """
    yml = YAML()
    yml.default_flow_style = False
    yml.preserve_quotes = True
    return yml


# ---------------------------------------------------------------------------
# JSON serializer helper
# ---------------------------------------------------------------------------


def json_serializer(obj: object) -> str:
    """JSON serializer for objects not handled by default json code.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-compatible string representation.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# YAML read/write
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> dict[str, object]:
    """Read and parse a YAML file with a shared lock.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.  Returns ``{}`` for empty files.

    Raises:
        StorageError: If the file cannot be read or parsed.
    """
    if not path.exists():
        raise StorageError(f"YAML file not found: {path}", path=str(path))
    try:
        with path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                data = _new_yaml().load(fh)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except (OSError, YAMLError, ValueError, TypeError) as exc:
        raise StorageError(
            f"Failed to read YAML: {exc}",
            path=str(path),
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise StorageError(
            f"YAML root must be a mapping, got {type(data).__name__}",
            path=str(path),
        )
    result: dict[str, object] = dict(data)
    return result


def write_yaml(path: Path, data: dict[str, object]) -> None:
    """Atomically write *data* to a YAML file.

    Writes to a temporary file in the same directory, then renames.
    This prevents corruption if the process is interrupted mid-write.

    Args:
        path: Target YAML file path.
        data: Dictionary to serialise as YAML.

    Raises:
        StorageError: If the write fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(path.parent),
            suffix=".yaml.tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    _new_yaml().dump(data, fh)
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            # fd is now closed by os.fdopen context manager
            tmp_path.rename(path)
        except Exception:  # broad catch: must clean temp file on any failure
            tmp_path.unlink(missing_ok=True)
            raise
        logger.debug("yaml_written", path=str(path))
    except StorageError:
        raise
    except (OSError, YAMLError, ValueError, TypeError, RuntimeError) as exc:
        raise StorageError(
            f"Failed to write YAML: {exc}",
            path=str(path),
        ) from exc


# ---------------------------------------------------------------------------
# JSONL append
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append a JSON record to a JSONL file with an exclusive lock.

    Args:
        path: Target JSONL file path (created if absent).
        record: Dictionary to serialise as a single JSON line.

    Raises:
        StorageError: If the append fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=json_serializer) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        logger.debug("jsonl_appended", path=str(path))
    except (OSError, ValueError, TypeError) as exc:
        raise StorageError(
            f"Failed to append JSONL: {exc}",
            path=str(path),
        ) from exc


# ---------------------------------------------------------------------------
# Advisory lock context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def lock_for_rmw(path: Path) -> Generator[Path, None, None]:
    """Advisory exclusive lock for read-modify-write cycles.

    Acquires an exclusive lock on ``{path}.lock`` before yielding,
    releases after the block completes (or on exception).  This prevents
    concurrent R-M-W races on the same file.

    Args:
        path: The file being protected.  A sibling ``.lock`` file is used.

    Yields:
        The original *path* (unchanged) for caller convenience.

    Example::

        with lock_for_rmw(entry_path) as p:
            data = read_yaml(p)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_yaml(p, data)
    """
    lock_path = path.parent / f"{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        yield path
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()
