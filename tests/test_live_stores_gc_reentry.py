"""A registered connection the garbage collector finalizes while the live-store registry is being read.

The finalizer runs on whatever thread triggered the collection. FD_LOCK is an RLock, so a finalizer
that released the connection itself re-entered the lock from inside the scan of ``_OPEN`` and
deleted the entry mid-iteration: "dictionary changed size during iteration" (rc7 C2).
"""

from __future__ import annotations

import gc
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory import _live_stores
from trw_memory._inode_pin import current_identity


@pytest.fixture
def collect_only_on_demand() -> Iterator[None]:
    """Keep the cycle collector from freeing the dropped connection before the scan asks it to."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.filterwarnings("ignore::ResourceWarning")  # the dropped connection is collected unclosed on purpose
def test_a_connection_collected_mid_scan_is_released_after_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collect_only_on_demand: None
) -> None:
    kept_path, dropped_path = tmp_path / "kept.db", tmp_path / "dropped.db"
    kept = _live_stores.connect_registered(kept_path, sqlite3, str(kept_path))
    dropped = _live_stores.connect_registered(dropped_path, sqlite3, str(dropped_path))
    kept_id, dropped_id = current_identity(kept_path), current_identity(dropped_path)
    assert kept_id in _live_stores._OPEN and dropped_id in _live_stores._OPEN
    cycle: list[object] = [dropped]
    cycle.append(cycle)  # only the cycle collector can free it now
    del dropped, cycle

    scan_sidecars = _live_stores._record_sidecars

    def collect_then_scan(identity: tuple[int, int], store: object) -> None:
        gc.collect()  # the dropped connection's finalizer runs here, inside the loop over _OPEN
        scan_sidecars(identity, store)  # type: ignore[arg-type]

    monkeypatch.setattr(_live_stores, "_record_sidecars", collect_then_scan)
    plain = tmp_path / "plain.txt"
    plain.write_text("not a store")
    fd = os.open(plain, os.O_RDONLY)
    try:
        assert _live_stores.admit_reader_fd(fd)
    finally:
        _live_stores.close_reader_fd(fd)

    # The next registry access applied the release: the dropped store is no longer live.
    assert dropped_id not in _live_stores._OPEN
    dir_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        assert not _live_stores.is_known_live(dir_fd, dropped_path.name)
        assert _live_stores.is_known_live(dir_fd, kept_path.name)
    finally:
        os.close(dir_fd)
    kept.close()
    assert kept_id not in _live_stores._OPEN
