"""Tests for the ``checkpoint_wal()`` maintenance seam.

``checkpoint_wal`` already exists on :class:`SQLiteBackend` (it is the
single-connection WAL-reset corruption fix). These tests cover the *public
maintenance contract* of that method plus the new no-op stub added to the
:class:`StorageBackend` interface so non-WAL backends (YAML) expose the seam
uniformly.

The SQLite implementation returns a :class:`CheckpointResult` TypedDict
(``{busy, checkpointed, mode}``) — the exported contract consumed by
trw-mcp's ``maybe_checkpoint_wal`` — not a raw ``pages_*`` dict, so these
assertions target the real returned keys.
"""

from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


def _make_entry(content: str) -> MemoryEntry:
    now = datetime.datetime.now(datetime.timezone.utc)
    return MemoryEntry(
        id=str(uuid.uuid4()),
        content=content,
        namespace="default",
        status=MemoryStatus.ACTIVE,
        importance=0.5,
        created_at=now,
        updated_at=now,
    )


class TestWalCheckpointApi:
    def test_checkpoint_wal_returns_dict(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "wal.db")
        try:
            result = backend.checkpoint_wal()
            assert isinstance(result, dict)
            # CheckpointResult exposes busy/checkpointed/mode; "checkpointed" is
            # the count of WAL frames written back to the main DB.
            assert "checkpointed" in result
            assert "mode" in result
        finally:
            backend.close()

    def test_checkpoint_wal_passive_mode(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "wal2.db")
        try:
            # Write entries to generate WAL activity, then checkpoint it back.
            for i in range(10):
                backend.store(_make_entry(f"entry {i}"))
            result = backend.checkpoint_wal("PASSIVE")
            # A PASSIVE checkpoint never resets the WAL, so it runs as requested
            # and reports a non-negative frame count without raising.
            assert result["mode"] == "PASSIVE"
            assert result["checkpointed"] >= 0
        finally:
            backend.close()

    def test_checkpoint_wal_interface_stub_returns_dict(self) -> None:
        # The interface stub is a safe no-op returning {} — exercised through
        # YAMLBackend, which has no WAL and inherits the default.
        assert StorageBackend.checkpoint_wal is not None
        with tempfile.TemporaryDirectory() as d:
            backend = YAMLBackend(Path(d) / "mem.yaml")
            result = backend.checkpoint_wal()
            assert isinstance(result, dict)
            assert result == {}
