"""SQLite delta matrix for the PRD-CORE-181-FR06 ``memory_model_v2_importance_type`` cutover.

Covers the SQLite schema gate and its ambiguity/invalid-type blocking with a
path/row classification report, the first-party remote protocol boundary, and
a source census proving the external ``impact`` vocabulary is contained to the
versioned mapper + migration. (The one-time YAML + backup-API orchestrator
this delta shipped alongside had no upgrade-path caller and was removed in the
trw-memory deletion wave — see UPGRADE-NOTES-8.0.0.md.)

Every fixture uses ``tmp_path`` databases — no real store is touched.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

import trw_memory
from trw_memory.storage._memory_model_v2 import (
    ClassificationEntry,
    MigrationBlocked,
    migrate_sqlite_importance_type,
)
from trw_memory.storage._schema import (
    CREATE_MEMORIES,
    SCHEMA_VERSION,
    SchemaDowngradeError,
    ensure_schema,
)

_TS = "2026-01-01T00:00:00+00:00"
_SRC_ROOT = Path(trw_memory.__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _make_v1_db(path: Path, rows: list[tuple[object, ...]]) -> None:
    """Create a full v1 ``memories`` shape (user_version=1) with *rows*."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(CREATE_MEMORIES)
        conn.executemany(
            "INSERT INTO memories (id, content, importance, type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SQLite schema gate matrix
# ---------------------------------------------------------------------------


def test_user_version_0_wild_and_fresh_end_at_current(tmp_path: Path) -> None:
    """A wild uv=0 db and a fresh uv=0 db both end at the current version."""
    assert SCHEMA_VERSION >= 2  # FR06 introduced v2; later migrations may advance it.
    fresh = sqlite3.connect(":memory:")
    ensure_schema(fresh)
    assert _user_version(fresh) == SCHEMA_VERSION

    wild = sqlite3.connect(":memory:")
    wild.execute(CREATE_MEMORIES)  # full current schema, uv still 0
    wild.execute(
        "INSERT INTO memories (id, content, importance, type, created_at, updated_at) "
        "VALUES ('L-wild', 'wild', 0.7, 'pattern', ?, ?)",
        (_TS, _TS),
    )
    wild.commit()
    assert _user_version(wild) == 0
    ensure_schema(wild)
    assert _user_version(wild) == SCHEMA_VERSION
    # Row preserved (v0 bootstrap through v1 then v2 is stamp, not rewrite).
    assert wild.execute("SELECT content, importance FROM memories WHERE id='L-wild'").fetchone() == ("wild", 0.7)


def test_v1_migrates_once_then_v2_is_noop(tmp_path: Path) -> None:
    """v1 -> v2 runs the delta exactly once; re-open at v2 takes the fast path."""
    import trw_memory.storage._schema as schema_mod

    db = tmp_path / "memory.db"
    _make_v1_db(db, [("L-1", "c", 0.4, "pattern", _TS, _TS)])
    conn = sqlite3.connect(str(db))

    calls: list[int] = []
    real = migrate_sqlite_importance_type

    def _spy(cursor: sqlite3.Cursor) -> None:
        calls.append(1)
        real(cursor)

    schema_mod._MIGRATIONS[2] = _spy
    try:
        ensure_schema(conn)  # v1 -> v2, one delta call
        assert _user_version(conn) == SCHEMA_VERSION
        assert calls == [1]
        ensure_schema(conn)  # already v2 -> fast path, no further delta
        assert calls == [1]
    finally:
        schema_mod._MIGRATIONS[2] = real
        conn.close()


def test_newer_user_version_typed_rejection_before_ddl() -> None:
    """A newer user_version raises SchemaDowngradeError before any DDL."""
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(SchemaDowngradeError):
        ensure_schema(conn)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert "memories" not in {r[0] for r in rows}
    assert _user_version(conn) == SCHEMA_VERSION + 1


def test_missing_type_becomes_pattern(tmp_path: Path) -> None:
    """A row with empty/NULL type is normalised to 'pattern' during the delta."""
    db = tmp_path / "memory.db"
    _make_v1_db(db, [("L-empty", "c", 0.5, "", _TS, _TS)])
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET type = NULL WHERE id='L-empty'")
    conn.execute(
        "INSERT INTO memories (id, content, importance, type, created_at, updated_at) "
        "VALUES ('L-blank', 'c', 0.5, '', ?, ?)",
        (_TS, _TS),
    )
    conn.commit()

    ensure_schema(conn)
    assert _user_version(conn) == SCHEMA_VERSION
    assert conn.execute("SELECT type FROM memories WHERE id='L-empty'").fetchone()[0] == "pattern"
    assert conn.execute("SELECT type FROM memories WHERE id='L-blank'").fetchone()[0] == "pattern"


def test_invalid_type_blocks_with_report_and_version_still_1(tmp_path: Path) -> None:
    """Invalid enum type blocks with a classification report; version stays 1, no partial writes."""
    db = tmp_path / "memory.db"
    _make_v1_db(
        db,
        [
            ("L-ok", "c", 0.5, "pattern", _TS, _TS),
            ("L-bad", "c", 0.5, "not_a_type", _TS, _TS),
        ],
    )
    conn = sqlite3.connect(str(db))
    with pytest.raises(MigrationBlocked) as blocked:
        ensure_schema(conn)

    assert _user_version(conn) == 1  # no bump
    # No partial writes: the invalid row is untouched.
    assert conn.execute("SELECT type FROM memories WHERE id='L-bad'").fetchone()[0] == "not_a_type"
    report = blocked.value.report
    assert any(e.kind == "sqlite_row" and e.ref == "L-bad" and "invalid type" in e.reason for e in report)


# ---------------------------------------------------------------------------
# Conflict / ambiguity blocking
# ---------------------------------------------------------------------------


def test_conflicting_sqlite_impact_importance_blocks_no_partial_writes(tmp_path: Path) -> None:
    """A SQLite row with both impact and importance columns disagreeing blocks and rolls back."""
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    conn.execute(CREATE_MEMORIES)  # has importance
    conn.execute("ALTER TABLE memories ADD COLUMN impact REAL")  # legacy column coexists
    conn.execute(
        "INSERT INTO memories (id, content, importance, impact, type, created_at, updated_at) "
        "VALUES ('L-c', 'c', 0.9, 0.2, 'pattern', ?, ?)",
        (_TS, _TS),
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()

    with pytest.raises(MigrationBlocked) as blocked:
        ensure_schema(conn)

    assert any(e.kind == "sqlite_row" and "conflicting" in e.reason for e in blocked.value.report)
    assert _user_version(conn) == 1
    assert conn.execute("SELECT importance, impact FROM memories WHERE id='L-c'").fetchone() == (0.9, 0.2)
    conn.close()


# ---------------------------------------------------------------------------
# First-party remote boundary
# ---------------------------------------------------------------------------


def test_remote_boundary_encoder_decoder_round_trips_impact_importance() -> None:
    """The learning_api_v1 mapper round-trips canonical importance <-> external impact."""
    from trw_memory.sync._remote_common import (
        decode_learning_api_v1_result,
        encode_learning_api_v1,
        encode_learning_api_v1_search,
    )

    encoded = encode_learning_api_v1(
        summary="s",
        detail=None,
        tags=["t"],
        importance=0.83,
        source_project="p",
        source_learning_id="L-1",
    )
    # Wire vocabulary on the way out.
    assert encoded["impact"] == 0.83

    decoded = decode_learning_api_v1_result(dict(encoded))
    # Canonical vocabulary on the way back in.
    assert decoded["importance"] == 0.83
    assert "impact" not in decoded

    params = encode_learning_api_v1_search(query="q", limit=5, min_importance=0.4)
    assert params["min_impact"] == 0.4


def test_remote_publish_fetch_route_through_mapper() -> None:
    """publish/_anonymize_entry and fetch params go through the versioned boundary."""
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.sync._remote_publish import _anonymize_entry

    entry = MemoryEntry(
        id="L-1",
        content="hello",
        importance=0.77,
        created_at=_TS,
        updated_at=_TS,
    )
    payload = _anonymize_entry(entry)
    # Publish still emits the external 'impact' field (produced by the mapper).
    assert payload["impact"] == 0.77


# ---------------------------------------------------------------------------
# Source census
# ---------------------------------------------------------------------------

# Local storage / lifecycle readers + remote publish/fetch: canonical
# importance only, no 'impact' vocabulary token anywhere.
#
# NOTE: storage/_cold_rebuild.py is deliberately NOT in this blanket list. It
# carries ONE blessed exception to the impact-vocabulary census — a legacy-key
# disaster-recovery fallback on the cold-archive rebuild path (see
# _COLD_REBUILD_DR_FALLBACK below) — so it is censused separately, by exact
# string, rather than by a blanket "no 'impact' substring" rule.
_READER_FILES = (
    "storage/_row_mapper.py",
    "storage/yaml_backend.py",
    "lifecycle/tiers/_scoring.py",
    "sync/_remote_publish.py",
    "sync/_remote_fetch.py",
)

# The single blessed impact-vocabulary occurrence in the whole cutover surface.
# recover_db(rebuild_from_cold=True) auto-invokes storage/_cold_rebuild.py on a
# corrupt DB, and pre-cutover cold archives may still be keyed only on the legacy
# ``impact`` field. Without this fallback every un-migrated cold-archive entry
# would silently recover as importance 0.5 (real data loss — release-verify
# 2026-07-17 P0). The census exempts it BY EXACT STRING so that any *new*
# impact-keyed read still trips the invariant.
_COLD_REBUILD_REL = "storage/_cold_rebuild.py"
_COLD_REBUILD_DR_FALLBACK = 'y.get("importance", y.get("impact", 0.5))'

# Files permitted to carry the external/legacy vocabulary.
_VOCAB_ALLOWED = frozenset(
    {
        "sync/_remote_common.py",  # versioned learning_api_v1 mapper
        "storage/_schema.py",  # v0 impact->importance rename migration
        "storage/_memory_model_v2.py",  # v2 cutover migration code
    }
)

_VOCAB_LITERAL = re.compile(r"""(["'])impact\1|min_impact""")

# A runtime READ/WRITE of the external ``impact``/``min_impact`` *data key* — the
# actual reader-fallback defect class (as opposed to prose, which the PRD-CORE-181
# FR06 census rule explicitly permits anywhere). Matches ``x.get("impact"...)``,
# ``x["impact"]``, and ``{"impact": ...}`` for both ``impact`` and ``min_impact``.
_VOCAB_DATA_KEY = re.compile(
    r"""\.get\(\s*["'](?:min_)?impact["']"""
    r"""|\[\s*["'](?:min_)?impact["']\s*\]"""
    r"""|["'](?:min_)?impact["']\s*:"""
)

# Only these files may perform a data-key read/write of the external vocabulary:
# the versioned mapper, the two migrations, and the explicit YAML importer.
_DATA_KEY_ALLOWED = frozenset(
    {
        "sync/_remote_common.py",
        "storage/_schema.py",
        "storage/_memory_model_v2.py",
    }
)


def test_source_census_impact_only_in_versioned_mapper_and_migration() -> None:
    """The 'impact' vocabulary appears only in the versioned mapper + migration code."""
    # (1) Named readers carry no 'impact' token at all.
    for rel in _READER_FILES:
        text = (_SRC_ROOT / rel).read_text(encoding="utf-8")
        assert "impact" not in text, f"reader {rel} must be importance-only"

    # (1b) storage/_cold_rebuild.py is the ONE blessed exception: a legacy-key
    #      disaster-recovery fallback (see _COLD_REBUILD_DR_FALLBACK). Assert the
    #      fallback is STILL present (regression guard — removing it silently
    #      resets un-migrated cold-archive importances to 0.5, real data loss)
    #      and that it is the ONLY external impact data-key read in the file, so a
    #      *new* impact-keyed read there still trips the census.
    cold = (_SRC_ROOT / _COLD_REBUILD_REL).read_text(encoding="utf-8")
    assert _COLD_REBUILD_DR_FALLBACK in cold, (
        "the blessed cold-rebuild legacy-DR fallback must not be removed — it prevents cold-archive data loss"
    )
    assert not _VOCAB_DATA_KEY.search(cold.replace(_COLD_REBUILD_DR_FALLBACK, "")), (
        f"{_COLD_REBUILD_REL} may read the external impact data key ONLY via the blessed DR fallback"
    )

    # (2) The versioned mapper DOES centralise the external vocabulary.
    mapper = (_SRC_ROOT / "sync/_remote_common.py").read_text(encoding="utf-8")
    assert "impact" in mapper
    assert "min_impact" in mapper

    # (3) Broad grep-style scan over the cutover surface: any quoted 'impact'
    #     data-key or 'min_impact' literal must live in an allowlisted file.
    scan_dirs = (
        _SRC_ROOT / "storage",
        _SRC_ROOT / "sync",
        _SRC_ROOT / "lifecycle" / "tiers",
    )
    offenders: list[str] = []
    for scan_dir in scan_dirs:
        for path in scan_dir.rglob("*.py"):
            rel = path.relative_to(_SRC_ROOT).as_posix()
            if rel in _VOCAB_ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            if rel == _COLD_REBUILD_REL:
                # Bless exactly the legacy-DR fallback; any OTHER impact literal
                # in this file still counts as an offender.
                text = text.replace(_COLD_REBUILD_DR_FALLBACK, "")
            if _VOCAB_LITERAL.search(text):
                offenders.append(rel)
    assert offenders == [], f"external impact vocabulary leaked outside the mapper/migration: {offenders}"

    # (4) WHOLE-tree scan (PRD-CORE-181-FR06 audit finding): NO file outside the
    #     versioned mapper / migration allowlist may READ or WRITE the external
    #     impact data key. This catches reader fallbacks like the historical
    #     _client_org_shared.py ``result.get("impact", ...)`` census-blind spot,
    #     while leaving PRD-permitted prose (comments, docstrings, field
    #     descriptions) alone anywhere in the tree.
    data_key_offenders = []
    migration_dir = _SRC_ROOT / "migration"
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(_SRC_ROOT).as_posix()
        if rel in _DATA_KEY_ALLOWED:
            continue
        if migration_dir.exists() and path.is_relative_to(migration_dir):
            continue
        text = path.read_text(encoding="utf-8")
        if rel == _COLD_REBUILD_REL:
            # Bless exactly the legacy-DR fallback (checked precisely in (1b));
            # any OTHER impact data-key read in this file still counts.
            text = text.replace(_COLD_REBUILD_DR_FALLBACK, "")
        if _VOCAB_DATA_KEY.search(text):
            data_key_offenders.append(rel)
    assert data_key_offenders == [], (
        f"external impact data-key read/write leaked outside the versioned mapper/migration: {data_key_offenders}"
    )


def test_shared_result_routes_external_impact_through_decoder() -> None:
    """A remote/org shared result's external ``impact`` is mapped to canonical
    ``importance`` through the versioned decoder, not a local fallback read."""
    from trw_memory._client_org_shared import shared_result_to_result

    result = shared_result_to_result({"memory_id": "R-1", "content": "shared", "impact": 0.71, "namespace": "org:acme"})
    assert result["importance"] == 0.71  # external impact decoded to importance
    assert result["score"] == 0.71  # score falls back to the decoded importance
    # Remote wire scope is not an authoritative local identity. The public
    # recall projection uses its shared ingress namespace, independently of
    # the trusted org:shared namespace used by admission.
    assert result["namespace"] == "shared"
    assert result["source"] == "shared"


def test_classification_entry_is_typed() -> None:
    entry = ClassificationEntry(kind="sqlite_row", ref="L-1", reason="invalid type 'x'")
    assert entry.kind == "sqlite_row"
    assert entry.ref == "L-1"
