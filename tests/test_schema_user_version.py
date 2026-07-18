"""PRAGMA user_version gating for ensure_schema (adoption-safety contract).

trw-memory is a public PyPI package with real user databases in the wild, so
the migration-adoption path is the whole risk. These tests pin the three
adoption cases the version gate must handle:

  * FRESH db (user_version=0, no tables) migrates 0 -> current and is stamped.
  * EXISTING fully-migrated wild db (user_version=0, all columns, live rows)
    is stamped to current WITHOUT re-migrating or mutating any row.
  * A NEWER-than-code db (user_version > SCHEMA_VERSION) fails LOUDLY with
    SchemaDowngradeError, BEFORE any DDL runs.
"""

from __future__ import annotations

import sqlite3

import pytest

from trw_memory.storage._schema import (
    CREATE_MEMORIES,
    SCHEMA_VERSION,
    SchemaDowngradeError,
    ensure_schema,
)


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Fresh database
# ---------------------------------------------------------------------------


def test_fresh_db_migrates_and_stamps_current_version() -> None:
    """A brand-new db (user_version=0) migrates and is stamped to SCHEMA_VERSION."""
    conn = sqlite3.connect(":memory:")
    assert _user_version(conn) == 0  # sqlite default

    ensure_schema(conn)

    assert _user_version(conn) == SCHEMA_VERSION
    assert "memories" in _table_names(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    # A representative column from the newest migration batch must be present.
    assert "valid_from" in cols
    assert "protection_tier" in cols


def test_stamped_db_takes_fast_path_and_skips_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """When user_version already equals SCHEMA_VERSION, the DDL storm is skipped.

    Observable effect: _bootstrap_and_backfill must NOT run. We stamp an EMPTY
    db to the current version, then assert no 'memories' table is created — the
    fast path returned before any DDL.
    """
    calls: list[int] = []

    import trw_memory.storage._schema as schema_mod

    real_bootstrap = schema_mod._bootstrap_and_backfill

    def _spy(cursor: sqlite3.Cursor) -> None:
        calls.append(1)
        real_bootstrap(cursor)

    monkeypatch.setattr(schema_mod, "_bootstrap_and_backfill", _spy)

    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    ensure_schema(conn)

    assert calls == []  # bootstrap skipped
    assert "memories" not in _table_names(conn)  # no DDL ran


def test_second_open_does_not_re_migrate(monkeypatch: pytest.MonkeyPatch) -> None:
    """First open migrates; a second open takes the fast path (no re-bootstrap)."""
    import trw_memory.storage._schema as schema_mod

    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)  # migrates 0 -> current
    assert _user_version(conn) == SCHEMA_VERSION

    calls: list[int] = []
    real_bootstrap = schema_mod._bootstrap_and_backfill

    def _spy(cursor: sqlite3.Cursor) -> None:
        calls.append(1)
        real_bootstrap(cursor)

    monkeypatch.setattr(schema_mod, "_bootstrap_and_backfill", _spy)
    ensure_schema(conn)  # second open

    assert calls == []


# ---------------------------------------------------------------------------
# Existing fully-migrated wild database (the critical adoption path)
# ---------------------------------------------------------------------------


def test_existing_fully_migrated_db_stamped_without_row_mutation() -> None:
    """A wild db at user_version=0 with EVERY column + live rows is stamped, not rewritten.

    This is the dominant real-world case: today's released build never sets
    user_version, so an already-current db opens at 0. ensure_schema must run
    the (now no-op) storm, stamp to SCHEMA_VERSION, and leave every row
    byte-identical.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(CREATE_MEMORIES)  # current full schema, user_version still 0
    conn.execute(
        "INSERT INTO memories (id, content, detail, importance, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("L-wild", "wild content", "wild detail", 0.7, "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
    )
    conn.commit()
    assert _user_version(conn) == 0

    pre = conn.execute(
        "SELECT id, content, detail, importance, created_at, updated_at FROM memories WHERE id='L-wild'"
    ).fetchone()

    ensure_schema(conn)

    assert _user_version(conn) == SCHEMA_VERSION
    post = conn.execute(
        "SELECT id, content, detail, importance, created_at, updated_at FROM memories WHERE id='L-wild'"
    ).fetchone()
    assert post == pre  # zero row mutation
    # Newly-added-since columns default to NULL on the pre-existing row.
    validity = conn.execute(
        "SELECT valid_from, invalid_from, invalidated_by FROM memories WHERE id='L-wild'"
    ).fetchone()
    assert validity == (None, None, None)


def test_legacy_partial_db_migrates_and_stamps() -> None:
    """A legacy db missing recent columns is upgraded AND stamped to current."""
    conn = sqlite3.connect(":memory:")
    # Realistic legacy shape: the earliest real schema always carried the core
    # columns (namespace/status/importance) that the pre-column-add index
    # creation depends on, but lacked the later migration batches.
    conn.execute(
        """
        CREATE TABLE memories (
            id         TEXT PRIMARY KEY,
            content    TEXT NOT NULL,
            detail     TEXT DEFAULT '',
            importance REAL DEFAULT 0.5,
            status     TEXT DEFAULT 'active',
            namespace  TEXT DEFAULT 'default',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO memories (id, content, created_at, updated_at) VALUES (?,?,?,?)",
        ("L-legacy", "legacy", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    conn.commit()

    ensure_schema(conn)

    assert _user_version(conn) == SCHEMA_VERSION
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "protection_tier" in cols
    assert "valid_from" in cols
    row = conn.execute("SELECT content FROM memories WHERE id='L-legacy'").fetchone()
    assert row[0] == "legacy"


def test_v3_legacy_enum_migration_is_lossless_and_idempotent() -> None:
    """Real legacy producer values become readable without losing their exact form."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    conn.execute("PRAGMA user_version = 2")
    conn.execute(
        "INSERT INTO memories (id, content, type, confidence, metadata, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "L-legacy-enums",
            "legacy",
            "gotcha",
            "0.97",
            '{"existing":"retained"}',
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    )
    conn.commit()

    ensure_schema(conn)
    first = conn.execute("SELECT type, confidence, metadata FROM memories WHERE id = 'L-legacy-enums'").fetchone()
    assert first == (
        "incident",
        "unverified",
        '{"existing": "retained", "legacy_confidence": "0.97", "legacy_memory_type": "gotcha"}',
    )
    assert _user_version(conn) == SCHEMA_VERSION

    ensure_schema(conn)
    second = conn.execute("SELECT type, confidence, metadata FROM memories WHERE id = 'L-legacy-enums'").fetchone()
    assert second == first


def test_v1_gotcha_rows_cross_v2_without_data_loss() -> None:
    """Known historical audit rows must not be blocked by the stricter v2 gate."""
    conn = sqlite3.connect(":memory:")
    _full_memories_at_version(conn, 1)
    conn.execute(
        "INSERT INTO memories (id, content, importance, type, confidence, metadata, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "L-gotcha",
            "audit finding",
            0.8,
            "gotcha",
            "verified",
            "{}",
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    )
    conn.commit()

    ensure_schema(conn)

    type_value, metadata = conn.execute("SELECT type, metadata FROM memories WHERE id = 'L-gotcha'").fetchone()
    assert type_value == "incident"
    assert '"legacy_memory_type": "gotcha"' in metadata
    assert _user_version(conn) == SCHEMA_VERSION


def test_v3_unknown_enums_are_preserved_without_semantic_inflation() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    conn.execute("PRAGMA user_version = 2")
    conn.execute(
        "INSERT INTO memories (id, content, type, confidence, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        ("L-future", "future", "novel_type", "0.75", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    conn.commit()

    ensure_schema(conn)

    type_value, confidence, metadata = conn.execute(
        "SELECT type, confidence, metadata FROM memories WHERE id = 'L-future'"
    ).fetchone()
    assert type_value == "pattern"
    assert confidence == "unverified"
    assert '"legacy_memory_type": "novel_type"' in metadata
    assert '"legacy_confidence": "0.75"' in metadata


# ---------------------------------------------------------------------------
# Newer-than-code database (downgrade guard)
# ---------------------------------------------------------------------------


def test_newer_db_raises_downgrade_error() -> None:
    """user_version > SCHEMA_VERSION raises SchemaDowngradeError."""
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(SchemaDowngradeError):
        ensure_schema(conn)


def test_downgrade_error_raised_before_any_ddl() -> None:
    """The downgrade guard fails loud BEFORE creating tables — no partial writes."""
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")

    with pytest.raises(SchemaDowngradeError):
        ensure_schema(conn)

    # No DDL ran: the memories table must not exist, and the version is untouched.
    assert "memories" not in _table_names(conn)
    assert _user_version(conn) == SCHEMA_VERSION + 5


def test_downgrade_error_is_runtime_error_subclass() -> None:
    """SchemaDowngradeError is a RuntimeError so existing broad handlers still catch it."""
    assert issubclass(SchemaDowngradeError, RuntimeError)


# ---------------------------------------------------------------------------
# PRD-CORE-181-FR06: memory_model_v2_importance_type cutover (schema gate)
# ---------------------------------------------------------------------------


def _full_memories_at_version(conn: sqlite3.Connection, version: int) -> None:
    """Create the current full ``memories`` shape and stamp ``user_version``."""
    conn.execute(CREATE_MEMORIES)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def test_prd_core_181_fr06() -> None:
    """FR06 schema gate: v0->v2, v1 migrates once, v2 no-ops, >2 rejects, and
    missing/invalid ``type`` normalise-or-block with no partial writes.
    """
    from trw_memory.storage._memory_model_v2 import (
        MIGRATION_KEY,
        MigrationBlocked,
        migrate_sqlite_importance_type,
    )
    from trw_memory.storage._schema import _MIGRATIONS

    # Migration key is registered at _MIGRATIONS[2] and SCHEMA_VERSION advanced.
    assert SCHEMA_VERSION >= 2
    assert _MIGRATIONS.get(2) is migrate_sqlite_importance_type
    assert MIGRATION_KEY == "memory_model_v2_importance_type"

    # (1) user_version 0 fresh -> ends at 2.
    fresh = sqlite3.connect(":memory:")
    ensure_schema(fresh)
    assert _user_version(fresh) == SCHEMA_VERSION

    # (2) user_version 1 -> migrates exactly once to 2; second open no-ops.
    v1 = sqlite3.connect(":memory:")
    _full_memories_at_version(v1, 1)
    v1.execute(
        "INSERT INTO memories (id, content, importance, type, created_at, updated_at) "
        "VALUES ('L-v1', 'c', 0.4, 'pattern', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
    )
    v1.commit()
    ensure_schema(v1)
    assert _user_version(v1) == SCHEMA_VERSION

    calls: list[int] = []
    real = migrate_sqlite_importance_type

    def _spy(cursor: sqlite3.Cursor) -> None:
        calls.append(1)
        real(cursor)

    _MIGRATIONS[2] = _spy
    try:
        ensure_schema(v1)  # already at 2 -> fast path, no migration call
        assert calls == []
    finally:
        _MIGRATIONS[2] = real

    # (3) user_version newer than this build -> typed rejection BEFORE any DDL.
    newer = sqlite3.connect(":memory:")
    newer.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(SchemaDowngradeError):
        ensure_schema(newer)
    assert "memories" not in _table_names(newer)
    assert _user_version(newer) == SCHEMA_VERSION + 1

    # (4) missing/empty type -> pattern.
    missing = sqlite3.connect(":memory:")
    _full_memories_at_version(missing, 1)
    missing.execute(
        "INSERT INTO memories (id, content, importance, type, created_at, updated_at) "
        "VALUES ('L-empty', 'c', 0.5, '', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
    )
    missing.commit()
    ensure_schema(missing)
    assert _user_version(missing) == SCHEMA_VERSION
    assert missing.execute("SELECT type FROM memories WHERE id='L-empty'").fetchone()[0] == "pattern"

    # (5) invalid type -> BLOCK with version still 1 and no partial writes.
    invalid = sqlite3.connect(":memory:")
    _full_memories_at_version(invalid, 1)
    invalid.execute(
        "INSERT INTO memories (id, content, importance, type, created_at, updated_at) "
        "VALUES ('L-bad', 'c', 0.5, 'not_a_type', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
    )
    invalid.commit()
    with pytest.raises(MigrationBlocked) as blocked:
        ensure_schema(invalid)
    assert _user_version(invalid) == 1  # no version bump
    assert invalid.execute("SELECT type FROM memories WHERE id='L-bad'").fetchone()[0] == "not_a_type"
    report = blocked.value.report
    assert any(entry.ref == "L-bad" and "invalid type" in entry.reason for entry in report)

    # (6) FR06 completion requires the source census: the external impact/min_impact
    # data-key vocabulary must be contained to the versioned mapper + migrations.
    # The mapped evidence explicitly includes the census (audit requirement).
    from tests.test_memory_model_v2_cutover import (
        test_source_census_impact_only_in_versioned_mapper_and_migration as _census,
    )

    _census()
