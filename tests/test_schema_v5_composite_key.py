"""PRD-CORE-245 FR01/FR02/FR03 + NFR02 — namespace as part of a row's identity.

These run against a REAL SQLite file through the shipped ``ensure_schema``, not
a hand-built fixture: the defect this PRD fixes lived in the DDL, so a test that
declares its own table proves nothing about what ships.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

import trw_memory.storage._dbapi  # noqa: F401  — installs pysqlite3 as ``sqlite3``
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage import _schema as schema_module
from trw_memory.storage._schema import SCHEMA_VERSION, ensure_fts_table, ensure_schema
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit

#: The schema-5 ``memories`` column set, written out rather than derived.
#:
#: This literal IS the consolidated migration table in PRD-CORE-245 section 4.
#: PRD-CORE-245 owns the re-key; PRD-CORE-244 owns ``verification_checked_at``
#: (FR03) and the absence of ``sessions_surfaced`` / ``avg_rework_delta`` /
#: ``outcome_correlation`` (FR08). Deriving it from ``ENTRY_COLUMNS`` would make
#: the test agree with whatever the code did; spelling it out is what makes a
#: divergent second delta fail the build instead of passing review.
EXPECTED_MEMORIES_COLUMNS = [
    "id",
    "content",
    "detail",
    "tags",
    "evidence",
    "importance",
    "status",
    "recurrence",
    "namespace",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "valid_from",
    "invalid_from",
    "invalidated_by",
    "access_count",
    "session_count",
    "q_value",
    "q_observations",
    "source",
    "source_identity",
    "client_profile",
    "model_id",
    "merged_from",
    "consolidated_from",
    "consolidated_into",
    "metadata",
    "vector_clock",
    "remote_id",
    "published_to_platform",
    "pending_delete",
    "cross_validated",
    "outcome_history",
    "assertions",
    "anchors",
    "anchor_validity",
    "type",
    "nudge_line",
    "expires_at",
    "confidence",
    "task_type",
    "domain",
    "phase_origin",
    "phase_affinity",
    "team_origin",
    "protection_tier",
    "sync_hash",
    "sync_seq",
    "last_synced_at",
    "recall_count",
    "helpful_count",
    "unhelpful_count",
    "verification_status",
    "verification_checked_at",
]

EXPECTED_TABLES = [
    "memories",
    "memories_fts",
    "memory_graph_edges",
    "memory_namespaces",
    "memory_tags",
    "vec_index",
    "wiki_refs",
]


def _v4_fixture(path: Path) -> None:
    """Write a database with the full sidecar set, stamped back to user_version 4.

    Opened through the real backend so ``vec_index`` and ``memories_fts`` exist
    exactly as they do in production; the version stamp is then rewound so the
    schema-5 delta has something to migrate.
    """
    backend = SQLiteBackend(path)
    backend.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    # FTS5 and vec0 shadow tables are engine-internal storage for the virtual
    # tables above them, not schema objects this migration declares.
    return sorted(str(r[0]) for r in rows if not str(r[0]).startswith(("memories_fts_", "vec_memories", "sqlite_")))


def test_schema_5_is_registered_exactly_once() -> None:
    """The single-registration tripwire (FR01).

    PRD-CORE-244 drops three columns in this same version, so two authors can
    each assign ``_MIGRATIONS[5]`` and the second silently wins — plain dict
    assignment, no error. This asserts against the imported module AND the
    source text, because only the second catches a duplicate assignment.
    """
    assert SCHEMA_VERSION == 5
    assert sorted(schema_module._MIGRATIONS) == [2, 3, 4, 5]

    source = Path(schema_module.__file__).read_text()
    assignments = re.findall(r"^_MIGRATIONS\[5\]\s*=", source, flags=re.MULTILINE)
    assert len(assignments) == 1, f"expected exactly one _MIGRATIONS[5] assignment, found {len(assignments)}"


def test_schema_5_delta_matches_the_consolidated_table(tmp_path: Path) -> None:
    """The delta changes exactly what the consolidated migration table says (FR01)."""
    db = tmp_path / "v4.db"
    _v4_fixture(db)

    conn = sqlite3.connect(db)
    ensure_schema(conn)
    ensure_fts_table(conn)

    columns = [str(r[1]) for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert columns == EXPECTED_MEMORIES_COLUMNS
    assert _tables(conn) == EXPECTED_TABLES
    # ENTRY_COLUMNS is the read/write projection; it must agree with the table
    # or every row mapping is off by a position.
    assert list(ENTRY_COLUMNS) == EXPECTED_MEMORIES_COLUMNS

    tag_edges = conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = 'tag_cooccurrence'").fetchone()[
        0
    ]
    assert tag_edges == 0
    conn.close()


def test_composite_key_admits_same_id_in_two_namespaces(tmp_path: Path) -> None:
    """FR01: a colliding id in a second namespace is a second row, not a replacement.

    Before schema 5 the ``INSERT OR REPLACE`` in ``_crud_ops.store`` replaced the
    first row outright, taking its namespace, its FTS row and its vector with it.
    """
    db = tmp_path / "collide.db"
    _v4_fixture(db)
    backend = SQLiteBackend(db)
    try:
        assert backend._conn.execute("PRAGMA user_version").fetchone()[0] == 5
        ddl = backend._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'memories'").fetchone()[0]
        assert "PRIMARY KEY (namespace, id)" in ddl

        backend.store(MemoryEntry(id="M-shared", content="owner content", namespace="project:owner"))
        backend.store(MemoryEntry(id="M-shared", content="other content", namespace="project:other"))

        rows = backend._conn.execute(
            "SELECT namespace, content FROM memories WHERE id = ? ORDER BY namespace", ("M-shared",)
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("project:other", "other content"),
            ("project:owner", "owner content"),
        ]
    finally:
        backend.close()


def test_sidecars_do_not_alias_across_namespaces(tmp_path: Path) -> None:
    """FR02: each namespace keeps its own FTS row and vector, and a delete takes only its own."""
    db = tmp_path / "sidecars.db"
    _v4_fixture(db)
    backend = SQLiteBackend(db)
    try:
        backend.store(MemoryEntry(id="M-shared", content="owner content", namespace="project:owner"))
        backend.store(MemoryEntry(id="M-shared", content="other content", namespace="project:other"))
        if backend._vec_available:
            vector = [0.5] * backend._dim
            backend.upsert_vector("M-shared", vector, namespace="project:owner")
            backend.upsert_vector("M-shared", vector, namespace="project:other")
            assert (
                backend._conn.execute("SELECT COUNT(*) FROM vec_index WHERE entry_id = ?", ("M-shared",)).fetchone()[0]
                == 2
            )

        assert backend._conn.execute("SELECT COUNT(*) FROM memories_fts WHERE id = ?", ("M-shared",)).fetchone()[0] == 2

        assert backend.delete("M-shared", namespace="project:owner") is True

        surviving_fts = backend._conn.execute(
            "SELECT namespace FROM memories_fts WHERE id = ?", ("M-shared",)
        ).fetchall()
        assert [str(r[0]) for r in surviving_fts] == ["project:other"]
        if backend._vec_available:
            surviving_vec = backend._conn.execute(
                "SELECT namespace FROM vec_index WHERE entry_id = ?", ("M-shared",)
            ).fetchall()
            assert [str(r[0]) for r in surviving_vec] == ["project:other"]
        assert backend.get("M-shared", namespace="project:other") is not None
    finally:
        backend.close()


def test_get_and_delete_require_namespace(tmp_path: Path) -> None:
    """FR03: identity operations name the namespace they mean, and refuse without one."""
    db = tmp_path / "identity.db"
    _v4_fixture(db)
    backend = SQLiteBackend(db)
    try:
        backend.store(MemoryEntry(id="M-x", content="a", namespace="project:a"))
        backend.store(MemoryEntry(id="M-x", content="b", namespace="project:b"))

        assert backend.get("M-x", namespace="project:a").content == "a"  # type: ignore[union-attr]
        assert backend.get("M-x", namespace="project:b").content == "b"  # type: ignore[union-attr]
        assert backend.get("M-x", namespace="project:absent") is None

        with pytest.raises(TypeError):
            backend.get("M-x")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            backend.delete("M-x")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            backend.update("M-x", content="anything")  # type: ignore[call-arg]

        assert backend.delete("M-x", namespace="project:a") is True
        assert backend.get("M-x", namespace="project:b") is not None
    finally:
        backend.close()


def test_migration_is_atomic_and_idempotent(tmp_path: Path) -> None:
    """NFR02: an interrupted delta leaves version 4 intact; a second run does nothing."""
    db = tmp_path / "atomic.db"
    _v4_fixture(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO memories (id, content, namespace, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("M-keep", "content", "default", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()

    # Interrupt the delta partway through and prove nothing was stamped.
    boom = sqlite3.connect(db)
    original = schema_module._MIGRATIONS[5]

    def _explode(cursor: sqlite3.Cursor) -> None:
        original(cursor)
        raise RuntimeError("interrupted mid-rebuild")

    schema_module._MIGRATIONS[5] = _explode
    try:
        with pytest.raises(RuntimeError, match="interrupted mid-rebuild"):
            ensure_schema(boom)
    finally:
        schema_module._MIGRATIONS[5] = original
    assert boom.execute("PRAGMA user_version").fetchone()[0] == 4
    assert boom.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == before
    boom.close()

    # Now let it complete, then run it again and prove the second run is a no-op.
    conn = sqlite3.connect(db)
    ensure_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == before
    ensure_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == before
    conn.close()
