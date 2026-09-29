"""QUARANTINE-DB-VERSIONING: the review queue (quarantine.db) is versioned exactly like memory.db.

The backlog row asked whether ``quarantine.db`` stays at ``user_version`` 8 while ``memory.db``
moves to 13. It does not: the queue is opened through the same ``SQLiteBackend`` and
``ensure_schema``, so a real historical queue migrates to ``SCHEMA_VERSION`` alongside the store.
This pins that parity, so a future schema bump cannot leave the queue behind unnoticed.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_quarantine import open_quarantine_backend
from trw_memory.storage._schema import SCHEMA_VERSION

pytestmark = pytest.mark.integration

_HISTORICAL_QUEUE = Path(__file__).parent / "fixtures" / "pre_q3_quarantine" / "quarantine.db"


def _user_version(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def test_a_historical_review_queue_migrates_to_the_same_schema_version_as_the_store(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    config = MemoryConfig(storage_path=str(memory), memory_single_store_path=str(memory / "memory.db"))
    queue = Path(config.quarantine_db_path)
    queue.parent.mkdir(parents=True)
    shutil.copyfile(_HISTORICAL_QUEUE, queue)  # built by the real pre-v8 quarantine code
    assert _user_version(queue) < SCHEMA_VERSION, "fixture must predate the current schema (non-vacuity)"

    with open_quarantine_backend(config) as backend:  # the production opener of the review queue
        backend.store(MemoryEntry(id="Q-new", content="held for review", namespace="default"))
    with create_backend_from_config(config, "default") as store:
        store.store(MemoryEntry(id="L-new", content="ordinary guidance", namespace="default"))

    assert _user_version(queue) == _user_version(memory / "memory.db") == SCHEMA_VERSION
