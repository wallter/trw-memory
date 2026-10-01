"""Off-machine-ready local backup archive (PRD-CORE-311 FR01).

Builds on the existing :func:`create_snapshot` (PRD-INFRA-065): a fresh
VACUUM INTO snapshot is taken to a uniquely-named temp location, gzipped, and
published to ``<base_dir>/memory/backups/<UTC timestamp>-<random>.db.gz``. The
timestamp carries microsecond precision AND a random suffix, and the final
name is published via ``os.link`` (fails closed with ``FileExistsError`` on a
name collision, retried with a fresh suffix) rather than ``os.replace`` — an
existing completed archive is NEVER silently overwritten, unlike a second-
resolution timestamp would allow for two calls in the same second or two
concurrent threads. A ``.sha256`` sidecar is written alongside it, naming the
DECOMPRESSED file the digest actually covers (``sha256sum`` format:
``"<hex>  <stem>.db"``, not the compressed ``.db.gz``) so ``sha256sum -c``
against a decompressed copy verifies correctly. The intermediate snapshot and
staging archive are removed on every path, success or failure.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import os
import secrets
import sqlite3
import tempfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory._live_stores import connect_registered
from trw_memory.exceptions import StorageError
from trw_memory.storage._snapshot import SnapshotError, create_snapshot, restore_from_snapshot, snapshots_base_dir

__all__ = [
    "BackupArchive",
    "BackupArchiveError",
    "backups_base_dir",
    "create_backup_archive",
    "restore_from_archive",
    "verified_archive",
]

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 1024 * 1024


class BackupArchiveError(StorageError):
    """Raised when a backup archive cannot be created."""


@dataclass(frozen=True)
class BackupArchive:
    """Return shape for :func:`create_backup_archive`."""

    path: Path
    sha256: str
    size_bytes: int
    source_snapshot: Path


def backups_base_dir(base_dir: Path) -> Path:
    """Return the backups root directory for a given base dir."""
    return base_dir / "memory" / "backups"


#: Bounded retries when a name collision is hit publishing the final archive
#: name (see module docstring) — collisions are practically impossible given
#: the microsecond timestamp + random suffix, but the loop still fails
#: closed rather than looping forever.
_PUBLISH_NAME_ATTEMPTS = 5


def create_backup_archive(base_dir: Path, db_path: Path) -> BackupArchive:
    """Gzip a fresh VACUUM INTO snapshot of ``db_path`` into a checksummed archive.

    Every call publishes a uniquely-named archive — concurrent or same-second
    calls never collide, and an existing completed archive is never
    overwritten (see module docstring).

    Raises:
        BackupArchiveError: the underlying snapshot, gzip, or publish step
            failed, or a unique name could not be published after
            ``_PUBLISH_NAME_ATTEMPTS`` retries.
    """
    backups_dir = backups_base_dir(base_dir)
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%f")

    tmp_snapshot_fd, tmp_snapshot_name = tempfile.mkstemp(dir=backups_dir, prefix=f".{stamp}-", suffix=".snapshot.tmp")
    os.close(tmp_snapshot_fd)
    tmp_snapshot = Path(tmp_snapshot_name)
    tmp_archive_fd, tmp_archive_name = tempfile.mkstemp(dir=backups_dir, prefix=f".{stamp}-", suffix=".db.gz.tmp")
    os.close(tmp_archive_fd)
    tmp_archive = Path(tmp_archive_name)
    try:
        try:
            create_snapshot(db_path, tmp_snapshot)
        except SnapshotError as exc:
            raise BackupArchiveError(f"backup snapshot failed: {exc}") from exc

        digest = hashlib.sha256()
        try:
            with open(tmp_snapshot, "rb") as source, gzip.open(tmp_archive, "wb") as gz_out:
                while chunk := source.read(_CHUNK_SIZE):
                    gz_out.write(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise BackupArchiveError(f"backup archive failed: {exc}") from exc

        sha256_hex = digest.hexdigest()

        # Publish under a unique name via os.link: FileExistsError on a
        # collision is retried with a fresh random suffix rather than ever
        # falling back to an overwrite (os.replace would silently delete an
        # earlier, still-referenced completed archive).
        dest: Path | None = None
        for _ in range(_PUBLISH_NAME_ATTEMPTS):
            candidate = backups_dir / f"{stamp}-{secrets.token_hex(4)}.db.gz"
            try:
                os.link(tmp_archive, candidate)
            except FileExistsError:  # trw-fail-silent-allow: a name collision is retried with a fresh suffix below,
                # not swallowed — the loop raises BackupArchiveError if every attempt collides (see below).
                continue
            dest = candidate
            break
        if dest is None:
            raise BackupArchiveError("backup archive failed: could not publish a unique archive name")

        # Sidecar names the DECOMPRESSED file the digest actually covers, in
        # sha256sum format, so `sha256sum -c` against a decompressed copy
        # verifies correctly (the digest is NOT over the compressed bytes).
        decompressed_name = dest.name.removesuffix(".gz")
        dest.with_name(dest.name + ".sha256").write_text(f"{sha256_hex}  {decompressed_name}\n", encoding="utf-8")

        size = dest.stat().st_size
        logger.info("backup_archive_created", dest=str(dest), size_bytes=size)
        return BackupArchive(path=dest, sha256=sha256_hex, size_bytes=size, source_snapshot=tmp_snapshot)
    finally:
        with contextlib.suppress(OSError):
            tmp_snapshot.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            tmp_archive.unlink(missing_ok=True)


def _sidecar_sha256(archive_path: Path) -> str | None:
    """Read the ``.sha256`` sidecar written by :func:`create_backup_archive`, if present.

    The sidecar format is ``"<hex>  <filename>\\n"``. A sidecar that does not
    EXIST returns ``None`` — the check is skip-if-absent (a pre-fix client's
    archive has none), never required. A sidecar that DOES exist but cannot be
    read or parsed is a distinct condition from "absent" and must never
    collapse into the same "no check" outcome: it raises, because a caller
    that silently skipped verification here could restore corrupted or
    tampered bytes believing the sidecar had cleared them.

    Raises:
        BackupArchiveError: the sidecar exists but is unreadable or empty.
    """
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    if not sidecar.exists():
        return None
    try:
        text = sidecar.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BackupArchiveError(f"backup archive sidecar exists but is unreadable: {sidecar}: {exc}") from exc
    first_token = text.split()[0] if text else ""
    if not first_token:
        raise BackupArchiveError(f"backup archive sidecar exists but is empty/malformed: {sidecar}")
    return first_token


_SQLITE_MAGIC = b"SQLite format 3\x00"
_REQUIRED_TABLE = "memories"


def _assert_restorable_store(path: Path) -> None:
    """Refuse *path* unless it is a healthy TRW memory store (INC-127).

    A gzip of anything decompresses; only the bytes say whether it is a store. Checks the SQLite header, runs
    ``PRAGMA integrity_check`` on a read-only connection, and requires the ``memories`` table. Raises
    :class:`BackupArchiveError` naming the failed check and never the file's content.
    """
    with open(path, "rb") as handle:
        if handle.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
            raise BackupArchiveError(
                "backup archive is not a SQLite database (the decompressed bytes have no SQLite header)"
            )
    try:
        conn = connect_registered(path, sqlite3, f"{path.as_uri()}?mode=ro&immutable=1", uri=True, store_lock=False)
        try:
            verdict = [row[0] for row in conn.execute("PRAGMA integrity_check")]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise BackupArchiveError(f"backup archive is not a readable SQLite database ({type(exc).__name__})") from exc
    if verdict != ["ok"]:
        raise BackupArchiveError("backup archive failed the SQLite integrity_check")
    if _REQUIRED_TABLE not in tables:
        raise BackupArchiveError(
            f"backup archive is a SQLite database but has no '{_REQUIRED_TABLE}' table: not a TRW memory store"
        )


@contextlib.contextmanager
def verified_archive(archive_path: Path, db_path: Path, *, expected_sha256: str | None = None) -> Iterator[Path]:
    """Yield a staged, fully verified decompressed copy of *archive_path*; it is removed on exit.

    Nothing about ``db_path`` is read or changed here: the archive is refused (``BackupArchiveError``) while the
    live store is still byte-identical. Refused: a missing or symlinked archive, a truncated or corrupt gzip, a
    sha256 that differs from *expected_sha256* or the ``.sha256`` sidecar beside the archive (when one exists),
    bytes that are not a healthy TRW memory store.
    """
    if archive_path.is_symlink():
        raise BackupArchiveError(f"backup archive is a symlink, which restore does not follow: {archive_path}")
    if not archive_path.is_file():
        raise BackupArchiveError(f"backup archive does not exist: {archive_path}")

    sha_to_check = expected_sha256 or _sidecar_sha256(archive_path)
    snap_dir = snapshots_base_dir(db_path.parent)
    snap_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    tmp_snapshot: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=snap_dir, prefix=".restore-", suffix=".db", delete=False) as tmp_handle:
            tmp_snapshot = Path(tmp_handle.name)
        try:
            with gzip.open(archive_path, "rb") as gz_in, open(tmp_snapshot, "wb") as out:
                while chunk := gz_in.read(_CHUNK_SIZE):
                    out.write(chunk)
                    digest.update(chunk)
        except (OSError, EOFError, zlib.error) as exc:
            raise BackupArchiveError(
                f"backup archive decompress failed ({type(exc).__name__}): the archive is truncated or not gzip"
            ) from exc

        actual_sha256 = digest.hexdigest()
        if sha_to_check is not None and actual_sha256 != sha_to_check:
            raise BackupArchiveError(f"backup archive sha256 mismatch: expected {sha_to_check}, got {actual_sha256}")
        _assert_restorable_store(tmp_snapshot)
        yield tmp_snapshot
    finally:
        if tmp_snapshot is not None:
            for suffix in ("", "-wal", "-shm", "-journal"):
                with contextlib.suppress(OSError):
                    tmp_snapshot.with_name(tmp_snapshot.name + suffix).unlink(missing_ok=True)


def restore_from_archive(archive_path: Path, db_path: Path, *, expected_sha256: str | None = None) -> None:
    """Verify *archive_path* (see :func:`verified_archive`), then replace ``db_path`` with it atomically.

    Delegates the replace to :func:`~trw_memory.storage._snapshot.restore_from_snapshot`, which holds the store's
    ``restore`` op: a store another process (the daemon) has open is refused with :class:`StoreBusyError`, nothing
    changed, identically to ``trw-memory restore --from-snapshot``.

    Raises:
        BackupArchiveError: the archive failed verification; the store is left untouched.
        StoreBusyError: a running daemon holds the store open.
    """
    with verified_archive(archive_path, db_path, expected_sha256=expected_sha256) as staged:
        try:
            restore_from_snapshot(db_path.parent, staged, db_path)
        except SnapshotError as exc:
            raise BackupArchiveError(f"backup restore failed: {exc}") from exc
