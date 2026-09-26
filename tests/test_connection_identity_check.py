"""PRD-SEC-016 round-2 finding 4 -- ``connect()`` brackets the SQLite open with an identity check.

The residual (SQLite-opens-by-path, not by-fd) TOCTOU gap is checked in
``trw_memory._live_stores.connect_registered``, which every file-backed open
(fresh open, reconnect, recovery, probe) funnels through: the store's inode is
pinned before the connect and compared after it. The swap test below fails on
Linux without the pin, because the replacement gets the freed inode number back.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.storage._connection import connect
from trw_memory.storage.sqlite_backend import SQLiteBackend


def test_a_fresh_database_creation_is_not_flagged_as_an_identity_mismatch(tmp_path: Path) -> None:
    """The file does not exist BEFORE connect (SQLite creates it): this must not look like a swap."""
    db_path = tmp_path / "new.db"
    assert not db_path.exists()

    conn = connect(db_path, dbapi=sqlite3, timeout=5.0, check_same_thread=True)
    try:
        assert db_path.exists()
    finally:
        conn.close()


def test_reopening_an_unchanged_existing_database_succeeds(tmp_path: Path) -> None:
    """The ordinary, non-raced reopen of an existing file must not be refused."""
    db_path = tmp_path / "existing.db"
    sqlite3.connect(str(db_path)).close()  # create it first, exactly like a real prior session

    conn = connect(db_path, dbapi=sqlite3, timeout=5.0, check_same_thread=True)
    conn.close()


def test_an_in_memory_database_is_never_identity_checked() -> None:
    """``:memory:`` has no file to stat; the check must not fire (and must not raise) for it."""
    conn = connect(Path(":memory:"), dbapi=sqlite3, timeout=5.0, check_same_thread=True)
    conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink/unlink-and-replace timing is POSIX-specific here")
def test_a_database_swapped_for_a_different_file_between_stat_and_connect_is_refused(tmp_path: Path) -> None:
    """The exact scenario the residual-window docstring describes: identity changes mid-open.

    ``dbapi.connect`` is patched to swap the file for a DIFFERENT one (same
    path, different inode) as a side effect of the mocked call itself --
    simulating a race won between this function's own before-stat and the
    real driver's open.
    """
    db_path = tmp_path / "target.db"
    sqlite3.connect(str(db_path)).close()  # the file connect() will see BEFORE

    real_connect = sqlite3.connect

    def _swap_then_connect(path: str, **kwargs: object) -> object:
        Path(path).unlink()
        real_connect(str(Path(path)) + ".seed").close()  # create a DIFFERENT file's connection target first
        Path(f"{path}.seed").rename(path)  # then move it into place: a new inode at the same name
        return real_connect(path, **kwargs)  # type: ignore[arg-type]

    with patch("trw_memory.storage._connection.sqlite3.connect", side_effect=_swap_then_connect):
        with pytest.raises(StorageError, match="identity changed"):
            connect(db_path, dbapi=sqlite3, timeout=5.0, check_same_thread=True)


def test_sqlite_backend_construction_surfaces_the_identity_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wired all the way to the public constructor: SQLiteBackend(...) itself raises, not just connect()."""
    db_path = tmp_path / "store.db"
    sqlite3.connect(str(db_path)).close()

    from tests._swap_after_pin import swap_after_pin

    swap_after_pin(monkeypatch, db_path)

    with pytest.raises(StorageError, match="identity changed"):
        SQLiteBackend(db_path)


def test_file_backed_excludes_memory_uris() -> None:
    from trw_memory.storage._connection import _file_backed

    assert _file_backed(Path(":memory:")) is False
    assert _file_backed(Path("file::memory:?cache=shared")) is False
    assert _file_backed(Path("/tmp/real.db")) is True


def test_connect_calls_dbapi_connect_exactly_once_per_call(tmp_path: Path) -> None:
    """Regression control: the identity check must not cause a second connect attempt on the happy path."""
    mock_conn = MagicMock()
    mock_conn.row_factory = None
    db_path = tmp_path / "counted.db"

    with patch("trw_memory.storage._connection.sqlite3.connect", return_value=mock_conn) as mocked:
        connect(db_path, dbapi=sqlite3, timeout=5.0, check_same_thread=True)

    assert mocked.call_count == 1
