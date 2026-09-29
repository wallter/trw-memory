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
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.storage._snapshot import SnapshotError, create_snapshot, restore_from_snapshot, snapshots_base_dir

__all__ = [
    "BackupArchive",
    "BackupArchiveError",
    "backups_base_dir",
    "create_backup_archive",
    "restore_from_archive",
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


def restore_from_archive(archive_path: Path, db_path: Path, *, expected_sha256: str | None = None) -> None:
    """Gunzip *archive_path*, verify its sha256, then restore ``db_path`` from it.

    Verifies the decompressed bytes' sha256 against *expected_sha256* when
    given, else against a ``.sha256`` sidecar sitting beside *archive_path*
    when one exists (skip-if-absent otherwise). Delegates the actual
    replace to the existing :func:`~trw_memory.storage._snapshot.restore_from_snapshot`,
    which holds the store's ``restore`` op for the duration — a store another
    process (the daemon) has open is refused with :class:`StoreBusyError`,
    nothing changed, identically to ``trw-memory restore --from-snapshot``.

    Raises:
        BackupArchiveError: the archive is missing, fails to decompress, or
            its sha256 does not match the expected/sidecar value. The store
            is left untouched on any of these failures.
        StoreBusyError: a running daemon holds the store open.
    """
    if not archive_path.exists():
        raise BackupArchiveError(f"backup archive does not exist: {archive_path}")

    sha_to_check = expected_sha256 or _sidecar_sha256(archive_path)

    base_dir = db_path.parent
    snap_dir = snapshots_base_dir(base_dir)
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
        except OSError as exc:
            raise BackupArchiveError(f"backup archive decompress failed: {exc}") from exc

        actual_sha256 = digest.hexdigest()
        if sha_to_check is not None and actual_sha256 != sha_to_check:
            raise BackupArchiveError(f"backup archive sha256 mismatch: expected {sha_to_check}, got {actual_sha256}")

        try:
            restore_from_snapshot(base_dir, tmp_snapshot, db_path)
        except SnapshotError as exc:
            raise BackupArchiveError(f"backup restore failed: {exc}") from exc
    finally:
        if tmp_snapshot is not None:
            with contextlib.suppress(OSError):
                tmp_snapshot.unlink(missing_ok=True)
