"""PRD-CORE-194 NFR01 — additive-only migration, zero existing-row mutation.

Opening a pre-migration ``memory.db`` (rows written before the validity columns
existed) with the new schema SHALL gain the columns with open-validity defaults
and mutate ZERO existing rows on read.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _make_pre_migration_db(db_path: Path) -> tuple[str, str]:
    """Create a full pre-194 memories table (CREATE_MEMORIES minus the 3 validity
    columns) and insert one row. Returns (entry_id, created_at_iso).

    Built from the real ``CREATE_MEMORIES`` DDL with the validity column lines
    stripped, so it is a faithful "DB written by the prior schema version".
    """
    from trw_memory.storage._schema import CREATE_MEMORIES

    pre_ddl_lines = [
        line
        for line in CREATE_MEMORIES.splitlines()
        if not line.strip().startswith(("valid_from", "invalid_from", "invalidated_by"))
    ]
    pre_ddl = "\n".join(pre_ddl_lines)
    assert "valid_from" not in pre_ddl

    created = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    created_iso = created.isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(pre_ddl)
        conn.execute(
            "INSERT INTO memories (id, content, detail, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("M-old", "legacy content", "legacy detail", created_iso, created_iso),
        )
        conn.commit()
    finally:
        conn.close()
    return "M-old", created_iso


def test_migration_is_additive_and_backfills_open_validity(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    entry_id, created_iso = _make_pre_migration_db(db_path)

    # Snapshot the pre-migration content (everything except the 3 added columns).
    pre_conn = sqlite3.connect(str(db_path))
    pre_row = pre_conn.execute(
        "SELECT id, content, detail, created_at, updated_at FROM memories WHERE id = ?",
        (entry_id,),
    ).fetchone()
    pre_conn.close()

    # Open with the new schema — ensure_schema runs the additive ALTER migration.
    backend = SQLiteBackend(db_path)
    loaded = backend.get(entry_id, namespace="default")

    assert loaded is not None
    # Open-validity defaults: absent valid_from back-fills to created_at.
    assert loaded.valid_from == datetime.fromisoformat(created_iso)
    assert loaded.invalid_from is None
    assert loaded.invalidated_by is None

    # NFR01: the three columns now exist (additive ALTER) ...
    post_conn = sqlite3.connect(str(db_path))
    cols = {r[1] for r in post_conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert {"valid_from", "invalid_from", "invalidated_by"} <= cols

    # ... and the original row content is byte-identical (no destructive rewrite).
    post_row = post_conn.execute(
        "SELECT id, content, detail, created_at, updated_at FROM memories WHERE id = ?",
        (entry_id,),
    ).fetchone()
    # The added columns default to NULL on the existing row (zero mutation on read).
    added = post_conn.execute(
        "SELECT valid_from, invalid_from, invalidated_by FROM memories WHERE id = ?",
        (entry_id,),
    ).fetchone()
    post_conn.close()

    assert post_row == pre_row
    assert added == (None, None, None)


def test_no_graph_db_dependency_in_pyproject() -> None:
    """NFR03: no neo4j/falkordb/cypher dependency introduced."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text().lower()
    assert "neo4j" not in text
    assert "falkordb" not in text
    assert "cypher" not in text


def test_round_trip_closed_window_survives_store_get(tmp_path: Path) -> None:
    """A closed-window entry persists + reloads with validity intact."""
    backend = SQLiteBackend(tmp_path / "m.db")
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    close = datetime(2026, 1, 2, tzinfo=timezone.utc)
    entry = MemoryEntry(
        id="M-closed",
        content="c",
        created_at=created,
        valid_from=created,
        invalid_from=close,
        invalidated_by="M-new",
    )
    backend.store(entry)
    got = backend.get("M-closed", namespace="default")
    assert got is not None
    assert got.invalid_from == close
    assert got.invalidated_by == "M-new"
    assert got.validity_state() == "superseded"


def test_anchor_validity_null_when_no_anchors(tmp_path: Path) -> None:
    """PRD-CORE-244 FR01 — an unanchored entry stores SQL NULL, and reads back None.

    The old ``1.0`` default made "nothing was ever assessed" identical to "every
    anchor still resolves", which is the score that feeds the recall ranking
    boost. The round trip has to hold at every layer: the model default, the
    column, and the row mapper that reads it back.
    """
    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(MemoryEntry(id="M-unanchored", content="no anchors here", namespace="default"))

    loaded = backend.get("M-unanchored", namespace="default")
    assert loaded is not None
    assert loaded.anchors == []
    assert loaded.anchor_validity is None

    # The column itself is NULL — not 1.0, and not 0.0 (which would be a real,
    # and false, "every anchor is broken" claim).
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    try:
        stored = conn.execute("SELECT anchor_validity FROM memories WHERE id = ?", ("M-unanchored",)).fetchone()
    finally:
        conn.close()
    assert stored == (None,)


def test_anchor_validity_survives_when_a_score_was_actually_computed(tmp_path: Path) -> None:
    """The falsification: a real score is still persisted and read back verbatim."""
    from trw_memory.models.memory import Anchor

    backend = SQLiteBackend(tmp_path / "m.db")
    backend.store(
        MemoryEntry(
            id="M-anchored",
            content="anchored",
            namespace="default",
            anchors=[Anchor(file="src/main.py", symbol_name="my_func")],
            anchor_validity=0.5,
        )
    )

    loaded = backend.get("M-anchored", namespace="default")
    assert loaded is not None
    assert loaded.anchor_validity == 0.5
