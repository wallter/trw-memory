"""W10 (trw-memory 4.0.0): the ``wiki_refs`` sidecar table is retired at schema 7.

Proves the forward-only ``_migrate_v7_retire_wiki_refs`` delta against **real**
pre-W10 databases: ``fixtures/pre_w10_wiki/memory.db`` was built by the actual historical
``SQLiteBackend``/``WikiPage`` code (``make_store.py`` extracts it via ``git archive`` from the
last commit before this retirement), so the fixture's
``wiki_refs`` rows, both of its indexes, and each entry's real
``wiki.page`` / ``wiki.slug`` / ``wiki.kind`` metadata payload are produced by
the genuine pre-retirement implementation — never a hand-written guess at the
old shape. Covers a database stamped at schema 6 (the historical build's own
tip) and one rolled back to schema 5 (the last version before wiki_refs's
namespace-boundary rebuild; additive-only schema 6 does not touch wiki_refs
or `memories`, so this is a faithful simulation of "migrated only that far").

Each fixture must:

* open and migrate to :data:`SCHEMA_VERSION` without raising,
* keep every ``memories`` row readable, with its own ``metadata`` — including
  the real ``wiki.page``/``wiki.slug``/``wiki.kind`` payload — byte-for-byte
  intact, and entry ids/count equal before and after,
* end with ``wiki_refs`` gone (table absent, not merely empty).

Also proves the migration is idempotent (running it twice is a no-op) and
that a brand-new database never creates ``wiki_refs`` in the first place.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from trw_memory.storage._schema import SCHEMA_VERSION, ensure_schema
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.integration


#: A real pre-W10 store and the receipt of what built it (fixtures/pre_w10_wiki/make_store.py).
_PRE_W10 = Path(__file__).parent / "fixtures" / "pre_w10_wiki"


def build_historical_wiki_store(db_path: Path) -> dict[str, object]:
    """Copy the committed pre-W10 store to *db_path*; returns its build receipt."""
    shutil.copyfile(_PRE_W10 / "memory.db", db_path)
    return dict(json.loads((_PRE_W10 / "receipt.json").read_text()))


def _wiki_refs_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'wiki_refs'").fetchone()
    return row is not None


@pytest.mark.parametrize("stamped_version", [5, 6])
def test_wiki_refs_carrying_store_migrates_cleanly(tmp_path: Path, stamped_version: int) -> None:
    """A real schema-5 or schema-6 store with wiki_refs rows migrates to SCHEMA_VERSION.

    Every entry stays readable with its wiki metadata intact, and wiki_refs
    (rows AND both indexes) is gone.
    """
    db_path = tmp_path / f"legacy_v{stamped_version}.db"
    receipt = build_historical_wiki_store(db_path)
    assert receipt["wiki_rows"] > 0, "fixture setup must actually carry real wiki_refs rows"
    assert receipt["indexes"] == ["idx_wiki_refs_source", "idx_wiki_refs_target"]
    entry_ids: list[str] = list(receipt["ids"])  # type: ignore[arg-type]

    conn = sqlite3.connect(db_path)
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == receipt["user_version"]
    if stamped_version == 5:
        # v5 -> v6 is purely additive to vec_index (unrelated to wiki_refs or
        # memories), so rolling the historical build's own v6 stamp back to 5
        # is a faithful simulation of "migrated only that far" without
        # fabricating any wiki_refs mechanics.
        before_metadata = {
            row[0]: row[1] for row in conn.execute("SELECT id, metadata FROM memories ORDER BY id").fetchall()
        }
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
    else:
        before_metadata = {
            row[0]: row[1] for row in conn.execute("SELECT id, metadata FROM memories ORDER BY id").fetchall()
        }
    assert _wiki_refs_exists(conn)

    ensure_schema(conn)

    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    assert not _wiki_refs_exists(conn), "wiki_refs must be dropped after the full migration chain"

    # Data-preservation: entry count, ids, and each row's real wiki metadata
    # payload are equal before and after.
    after_rows = conn.execute("SELECT id, metadata FROM memories ORDER BY id").fetchall()
    after_metadata = {row[0]: row[1] for row in after_rows}
    assert sorted(after_metadata) == sorted(entry_ids) == sorted(before_metadata)
    assert after_metadata == before_metadata
    for entry_id in entry_ids:
        assert '"wiki.page"' in after_metadata[entry_id]
        assert '"wiki.slug"' in after_metadata[entry_id]
        assert '"wiki.kind"' in after_metadata[entry_id]
    conn.close()

    # Every entry is still readable through the CURRENT public backend API,
    # and its real wiki metadata payload survived the retirement untouched.
    backend = SQLiteBackend(db_path)
    try:
        for entry_id in entry_ids:
            index = int(entry_id.rsplit("-", 1)[-1])
            reread = backend.get(entry_id, namespace="project:legacy")
            assert reread is not None
            assert "wiki.page" in reread.metadata
            assert reread.metadata["wiki.slug"] == f"topic/legacy-{index}"
            assert reread.metadata["wiki.kind"] == "topic"
    finally:
        backend.close()


def test_migration_is_idempotent_and_a_fresh_db_never_creates_wiki_refs(tmp_path: Path) -> None:
    """Re-running the migration is a no-op, and a fresh DB never creates wiki_refs."""
    db_path = tmp_path / "fresh.db"
    backend = SQLiteBackend(db_path)
    try:
        assert not _wiki_refs_exists(backend._conn), "a fresh DB must never create the retired wiki_refs table"
        assert int(backend._conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    finally:
        backend.close()

    # Re-running ensure_schema against the same file is a no-op: still no
    # table, still stamped, no error.
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    ensure_schema(conn)
    assert not _wiki_refs_exists(conn)
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    conn.close()


def test_dropping_wiki_refs_a_second_time_is_a_no_op(tmp_path: Path) -> None:
    """A store already migrated through v7 tolerates a second full ensure_schema pass."""
    db_path = tmp_path / "legacy_replay.db"
    receipt = build_historical_wiki_store(db_path)
    entry_ids = sorted(receipt["ids"])  # type: ignore[arg-type]

    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    assert not _wiki_refs_exists(conn)
    before = conn.execute("SELECT id FROM memories ORDER BY id").fetchall()

    # Idempotent replay: nothing raises, nothing changes.
    ensure_schema(conn)
    assert not _wiki_refs_exists(conn)
    after = conn.execute("SELECT id FROM memories ORDER BY id").fetchall()
    assert before == after == [(entry_id,) for entry_id in entry_ids]
    conn.close()


def test_refs_only_store_still_gets_a_pre_migration_snapshot(tmp_path: Path) -> None:
    """P1 fix: an empty ``memories`` table with populated ``wiki_refs`` must still snapshot.

    ``_has_rows`` used to probe only ``memories``; a store whose entries were
    already removed (leaving orphaned ``wiki_refs`` rows behind, e.g. by a
    direct SQL delete that bypassed the cascade) would then skip the
    pre-migration snapshot entirely, and the v7 drop discarded the only copy
    of that data with no way back.
    """
    from trw_memory.storage._schema_backup import BACKUP_DIR_NAME

    db_path = tmp_path / "refs_only.db"
    receipt = build_historical_wiki_store(db_path)
    assert receipt["wiki_rows"] > 0

    conn = sqlite3.connect(db_path)
    # Simulate the orphan case directly: empty memories, wiki_refs untouched.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM memories")
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    wiki_rows_before = conn.execute("SELECT COUNT(*) FROM wiki_refs").fetchone()[0]
    assert wiki_rows_before > 0

    ensure_schema(conn)
    conn.close()

    snapshots = sorted((tmp_path / BACKUP_DIR_NAME).glob(f"refs_only.db.pre-schema-{SCHEMA_VERSION}.*"))
    assert len(snapshots) == 1, "a refs-only store must still get a pre-migration snapshot"

    restored = sqlite3.connect(f"file:{snapshots[0]}?mode=ro", uri=True)
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert restored.execute("SELECT COUNT(*) FROM wiki_refs").fetchone()[0] == wiki_rows_before
    finally:
        restored.close()
