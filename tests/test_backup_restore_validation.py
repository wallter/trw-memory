"""A restore refuses a source that is not a healthy SQLite store, and leaves the live store byte-identical (INC-127, INC-129 a)."""

from __future__ import annotations

import gzip
import hashlib
import sqlite3
from pathlib import Path

import pytest

from trw_memory.storage._backup_archive import BackupArchiveError, create_backup_archive, restore_from_archive
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _live_store(tmp_path: Path) -> Path:
    from tests.conftest import make_entry

    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(make_entry(content="the live store must survive a bad restore"))
    backend.close()
    return db_path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gz(tmp_path: Path, payload: bytes, name: str = "src.db.gz") -> Path:
    path = tmp_path / name
    path.write_bytes(gzip.compress(payload))
    return path


def _assert_refused_and_untouched(tmp_path: Path, archive: Path, reason: str) -> None:
    db_path = _live_store(tmp_path)
    before = _digest(db_path)
    with pytest.raises(BackupArchiveError, match=reason):
        restore_from_archive(archive, db_path)
    assert _digest(db_path) == before
    snapshots = tmp_path / "memory" / "snapshots"
    leftovers = list(snapshots.glob(".restore-*")) if snapshots.exists() else []
    assert leftovers == []


def test_a_gzip_of_plain_text_is_refused(tmp_path: Path) -> None:
    _assert_refused_and_untouched(tmp_path, _gz(tmp_path, b"not sqlite at all"), "not a SQLite database")


def test_a_sqlite_file_without_the_memories_table_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.commit()
    conn.close()
    _assert_refused_and_untouched(tmp_path, _gz(tmp_path, other.read_bytes()), "memories")


def test_a_corrupt_sqlite_file_is_refused(tmp_path: Path) -> None:
    source = _live_store(tmp_path / "donor") if (tmp_path / "donor").mkdir() is None else None
    assert source is not None
    data = bytearray(source.read_bytes())
    data[4096:8192] = b"\xff" * 4096  # a valid header, damaged pages
    _assert_refused_and_untouched(tmp_path, _gz(tmp_path, bytes(data)), "integrity|not a SQLite|not a readable SQLite")


def test_a_truncated_archive_is_a_clean_error_not_a_traceback(tmp_path: Path) -> None:
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    archive = create_backup_archive(donor_dir, _live_store(donor_dir))
    cut = tmp_path / "cut.db.gz"
    cut.write_bytes(archive.path.read_bytes()[:40])
    _assert_refused_and_untouched(tmp_path, cut, "decompress")


def test_a_sidecar_that_does_not_match_is_refused(tmp_path: Path) -> None:
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    archive = create_backup_archive(donor_dir, _live_store(donor_dir))
    archive.path.with_name(archive.path.name + ".sha256").write_text("0" * 64 + "  x.db\n", encoding="utf-8")
    _assert_refused_and_untouched(tmp_path, archive.path, "sha256 mismatch")


def test_a_healthy_archive_still_restores(tmp_path: Path) -> None:
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    donor = _live_store(donor_dir)
    archive = create_backup_archive(donor_dir, donor)
    target = tmp_path / "target"
    target.mkdir()
    db_path = target / "memory.db"
    db_path.write_bytes(b"placeholder of the store being replaced")
    restore_from_archive(archive.path, db_path)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()
