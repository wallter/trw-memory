"""Tests for PRD-CORE-311 FR01: local gzip backup archive built on create_snapshot."""

from __future__ import annotations

import gzip
import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from trw_memory.exceptions import StoreBusyError
from trw_memory.storage._backup_archive import (
    BackupArchiveError,
    backups_base_dir,
    create_backup_archive,
    restore_from_archive,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    finally:
        conn.close()


def test_create_backup_archive_roundtrips_and_hashes(tmp_path: Path) -> None:
    """Gunzipping the archive yields a byte-identical VACUUM INTO snapshot; the sidecar hash matches."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    from tests.conftest import make_entry

    entry = make_entry(content="the backup drill must recall this exact sentence")
    backend.store(entry)
    backend.close()

    archive = create_backup_archive(tmp_path, db_path)

    assert archive.path.exists()
    assert archive.path.parent == backups_base_dir(tmp_path)
    sidecar = archive.path.with_name(archive.path.name + ".sha256")
    assert sidecar.exists()
    assert archive.sha256 in sidecar.read_text(encoding="utf-8")

    with gzip.open(archive.path, "rb") as gz_in:
        decompressed = gz_in.read()
    assert hashlib.sha256(decompressed).hexdigest() == archive.sha256
    assert archive.size_bytes == archive.path.stat().st_size

    restored_db = tmp_path / "restored.db"
    restored_db.write_bytes(decompressed)
    restored = SQLiteBackend(restored_db)
    try:
        got = restored.get(entry.id, namespace=entry.namespace)
    finally:
        restored.close()
    assert got is not None
    assert got.content == "the backup drill must recall this exact sentence"

    # The intermediate uncompressed snapshot must never persist.
    assert not archive.source_snapshot.exists()
    assert not any(p.name.endswith(".tmp") for p in backups_base_dir(tmp_path).iterdir())


def test_create_backup_archive_raises_on_missing_source(tmp_path: Path) -> None:
    missing = tmp_path / "never_created.db"
    with pytest.raises(BackupArchiveError):
        create_backup_archive(tmp_path, missing)


def test_create_backup_archive_leaves_no_partial_file_on_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure partway through gzip writing leaves no partial/truncated .gz at the final name."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    from tests.conftest import make_entry

    backend.store(make_entry(content="row that must not leak into a half-written archive"))
    backend.close()

    import trw_memory.storage._backup_archive as backup_archive_mod

    real_gzip_open = backup_archive_mod.gzip.open

    class _ExplodingGzipFile:
        def __init__(self, real_handle: object) -> None:
            self._real_handle = real_handle

        def write(self, data: bytes) -> int:
            raise OSError("simulated mid-write failure")

        def __enter__(self) -> _ExplodingGzipFile:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self._real_handle.__exit__(*exc_info)

    def _boom(*args: object, **kwargs: object) -> _ExplodingGzipFile:
        return _ExplodingGzipFile(real_gzip_open(*args, **kwargs))

    monkeypatch.setattr(backup_archive_mod.gzip, "open", _boom)

    with pytest.raises(BackupArchiveError):
        create_backup_archive(tmp_path, db_path)

    backups_dir = backups_base_dir(tmp_path)
    assert not any(backups_dir.glob("*.db.gz"))
    assert not any(backups_dir.glob("*.tmp"))


def test_create_backup_archive_sequential_calls_in_same_second_both_intact(tmp_path: Path) -> None:
    """P1 (sol review of S1): two creates in the same wall-clock second never collide/overwrite."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    first = create_backup_archive(tmp_path, db_path)
    second = create_backup_archive(tmp_path, db_path)

    assert first.path != second.path
    assert first.path.exists()
    assert second.path.exists()
    with gzip.open(first.path, "rb") as gz_in:
        assert hashlib.sha256(gz_in.read()).hexdigest() == first.sha256
    with gzip.open(second.path, "rb") as gz_in:
        assert hashlib.sha256(gz_in.read()).hexdigest() == second.sha256


def test_create_backup_archive_concurrent_calls_produce_distinct_archives(tmp_path: Path) -> None:
    """P1: concurrent threads publishing at the same instant never share a temp/final file."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    results: list[object] = [None] * 6
    errors: list[BaseException] = []

    def _worker(index: int) -> None:
        try:
            results[index] = create_backup_archive(tmp_path, db_path)
        except BaseException as exc:  # pragma: no cover - surfaced via `errors` assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent create_backup_archive calls raised: {errors}"
    paths = {r.path for r in results}  # type: ignore[union-attr]
    assert len(paths) == 6, "every concurrent call must publish a distinct archive path"
    for archive in results:
        assert archive.path.exists()  # type: ignore[union-attr]
        with gzip.open(archive.path, "rb") as gz_in:  # type: ignore[union-attr]
            assert hashlib.sha256(gz_in.read()).hexdigest() == archive.sha256  # type: ignore[union-attr]


def test_create_backup_archive_sidecar_names_decompressed_file(tmp_path: Path) -> None:
    """P2 (sol review of S1): the .sha256 sidecar names the decompressed .db, not the compressed .db.gz."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    archive = create_backup_archive(tmp_path, db_path)
    sidecar = archive.path.with_name(archive.path.name + ".sha256")
    hex_digest, named_file = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)

    assert hex_digest == archive.sha256
    assert named_file == archive.path.name.removesuffix(".gz")
    assert not named_file.endswith(".gz")

    with gzip.open(archive.path, "rb") as gz_in:
        decompressed = gz_in.read()
    assert hashlib.sha256(decompressed).hexdigest() == hex_digest


def test_restore_from_archive_roundtrips_into_fresh_store(tmp_path: Path) -> None:
    """restore_from_archive gunzips, verifies the sidecar hash, and restores via restore_from_snapshot."""
    from tests.conftest import make_entry

    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    entry = make_entry(content="the restore drill must recall this exact sentence")
    backend.store(entry)
    backend.close()

    archive = create_backup_archive(tmp_path, db_path)

    # Wipe the store, then restore from the archive alone.
    db_path.unlink()
    restore_from_archive(archive.path, db_path)

    restored = SQLiteBackend(db_path)
    try:
        got = restored.get(entry.id, namespace=entry.namespace)
    finally:
        restored.close()
    assert got is not None
    assert got.content == "the restore drill must recall this exact sentence"


def test_restore_from_archive_sha256_mismatch_refuses_and_leaves_store_untouched(tmp_path: Path) -> None:
    """A sha256 mismatch refuses the restore and never touches the existing store."""
    from tests.conftest import make_entry

    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    entry = make_entry(content="must survive a rejected restore")
    backend.store(entry)
    backend.close()
    original_bytes = db_path.read_bytes()

    archive = create_backup_archive(tmp_path, db_path)

    with pytest.raises(BackupArchiveError):
        restore_from_archive(archive.path, db_path, expected_sha256="0" * 64)

    assert db_path.read_bytes() == original_bytes
    still_there = SQLiteBackend(db_path)
    try:
        assert still_there.get(entry.id, namespace=entry.namespace) is not None
    finally:
        still_there.close()


def test_restore_from_archive_missing_archive_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    with pytest.raises(BackupArchiveError):
        restore_from_archive(tmp_path / "does-not-exist.db.gz", db_path)


def test_restore_from_archive_refuses_beside_running_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon-open refusal applies identically to restore_from_archive (reuses restore_from_snapshot's guard)."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()
    archive = create_backup_archive(tmp_path, db_path)

    import trw_memory.storage._snapshot as snapshot_mod

    def _busy_store_access(*args: object, **kwargs: object) -> object:
        raise StoreBusyError("store is busy")

    monkeypatch.setattr(snapshot_mod, "store_access", _busy_store_access)

    with pytest.raises(StoreBusyError):
        restore_from_archive(archive.path, db_path)
