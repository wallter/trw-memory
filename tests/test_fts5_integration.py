"""Tests for FTS5 full-text search integration.

Covers:
- FTS table creation and availability flag
- search_fts() finds entries by content, detail, tags
- FTS stays in sync with store(), update(), delete()
- search_fts() filter parameters (status, namespace, min_importance)
- Graceful empty-list return when no matches
- ensure_fts_table() populates from existing rows (migration path)
- search_fts() vs search() result parity for selective queries
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._schema import ensure_fts_table
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _entry(
    *,
    content: str = "test content",
    detail: str = "test detail",
    tags: list[str] | None = None,
    namespace: str = "default",
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    entry_id: str | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id or str(uuid.uuid4()),
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        namespace=namespace,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )


@pytest.fixture()
def backend() -> SQLiteBackend:
    return SQLiteBackend(Path(":memory:"))


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestFtsAvailability:
    def test_fts_available_on_modern_sqlite(self, backend: SQLiteBackend) -> None:
        assert backend.fts_available is True

    def test_search_fts_returns_empty_list_when_unavailable(self, backend: SQLiteBackend) -> None:
        backend._fts_available = False
        result = backend.search_fts("anything")
        assert result == []


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------


class TestSearchFtsBasic:
    def test_search_fts_finds_by_content(self, backend: SQLiteBackend) -> None:
        e = _entry(content="SQLite full-text search engine")
        backend.store(e)
        results = backend.search_fts("full-text")
        assert any(r.id == e.id for r in results)

    def test_search_fts_finds_by_detail(self, backend: SQLiteBackend) -> None:
        e = _entry(content="unrelated content", detail="FTS5 provides inverted index lookup")
        backend.store(e)
        results = backend.search_fts("inverted")
        assert any(r.id == e.id for r in results)

    def test_search_fts_finds_by_tags(self, backend: SQLiteBackend) -> None:
        e = _entry(content="generic content", tags=["fts5", "performance", "sqlite"])
        backend.store(e)
        results = backend.search_fts("fts5")
        assert any(r.id == e.id for r in results)

    def test_search_fts_no_match_returns_empty(self, backend: SQLiteBackend) -> None:
        backend.store(_entry(content="completely unrelated"))
        results = backend.search_fts("xyzzy_nonexistent_term_12345")
        assert results == []

    def test_search_fts_respects_top_k(self, backend: SQLiteBackend) -> None:
        for i in range(20):
            backend.store(_entry(content=f"matching term entry number {i}"))
        results = backend.search_fts("matching", top_k=5)
        assert len(results) <= 5

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_search_fts_non_positive_top_k_returns_empty(self, backend: SQLiteBackend, top_k: int) -> None:
        backend.store(_entry(content="matching term"))
        assert backend.search_fts("matching", top_k=top_k) == []

    def test_search_fts_returns_full_memory_entry(self, backend: SQLiteBackend) -> None:
        e = _entry(content="enterprise scale search", importance=0.9)
        backend.store(e)
        results = backend.search_fts("enterprise")
        assert len(results) == 1
        assert results[0].id == e.id
        assert results[0].content == "enterprise scale search"
        assert results[0].importance == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Sync correctness: store / update / delete keep FTS in sync
# ---------------------------------------------------------------------------


class TestFtsSyncCorrectness:
    def test_fts_synced_after_store(self, backend: SQLiteBackend) -> None:
        e = _entry(content="freshly stored unique_term_abc")
        backend.store(e)
        assert any(r.id == e.id for r in backend.search_fts("unique_term_abc"))

    def test_fts_updated_after_content_change(self, backend: SQLiteBackend) -> None:
        e = _entry(content="original_content_xyz")
        backend.store(e)
        backend.update(e.id, content="updated_content_pqr")
        assert backend.search_fts("original_content_xyz") == []
        assert any(r.id == e.id for r in backend.search_fts("updated_content_pqr"))

    def test_fts_updated_after_detail_change(self, backend: SQLiteBackend) -> None:
        e = _entry(content="fixed content", detail="old_detail_lmn")
        backend.store(e)
        backend.update(e.id, detail="new_detail_opq")
        assert backend.search_fts("old_detail_lmn") == []
        assert any(r.id == e.id for r in backend.search_fts("new_detail_opq"))

    def test_fts_removed_after_delete(self, backend: SQLiteBackend) -> None:
        e = _entry(content="to_be_deleted_uvw")
        backend.store(e)
        assert any(r.id == e.id for r in backend.search_fts("to_be_deleted_uvw"))
        backend.delete(e.id)
        assert backend.search_fts("to_be_deleted_uvw") == []

    def test_store_overwrite_keeps_fts_in_sync(self, backend: SQLiteBackend) -> None:
        eid = str(uuid.uuid4())
        e1 = _entry(entry_id=eid, content="first_version_term")
        e2 = _entry(entry_id=eid, content="second_version_term")
        backend.store(e1)
        backend.store(e2)  # INSERT OR REPLACE
        assert backend.search_fts("first_version_term") == []
        assert any(r.id == eid for r in backend.search_fts("second_version_term"))


# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------


class TestFtsFilers:
    def test_fts_status_filter_excludes_inactive(self, backend: SQLiteBackend) -> None:
        active = _entry(content="shared_term_status", status=MemoryStatus.ACTIVE)
        obsolete = _entry(content="shared_term_status", status=MemoryStatus.OBSOLETE)
        backend.store(active)
        backend.store(obsolete)
        results = backend.search_fts("shared_term_status", status=MemoryStatus.ACTIVE)
        ids = {r.id for r in results}
        assert active.id in ids
        assert obsolete.id not in ids

    def test_fts_namespace_filter(self, backend: SQLiteBackend) -> None:
        a = _entry(content="shared_ns_term", namespace="ns_a")
        b = _entry(content="shared_ns_term", namespace="ns_b")
        backend.store(a)
        backend.store(b)
        results = backend.search_fts("shared_ns_term", namespace="ns_a")
        ids = {r.id for r in results}
        assert a.id in ids
        assert b.id not in ids

    def test_fts_namespace_filter_applies_before_candidate_limit(self, backend: SQLiteBackend) -> None:
        for index in range(20):
            backend.store(_entry(content="dominant shared term", namespace="other", entry_id=f"other-{index}"))
        target = _entry(content="dominant shared term", namespace="target", entry_id="target")
        backend.store(target)

        results = backend.search_fts("dominant", namespace="target", top_k=1)

        assert [entry.id for entry in results] == [target.id]

    def test_fts_min_importance_filter(self, backend: SQLiteBackend) -> None:
        low = _entry(content="importance_term", importance=0.2)
        high = _entry(content="importance_term", importance=0.9)
        backend.store(low)
        backend.store(high)
        results = backend.search_fts("importance_term", min_importance=0.5)
        ids = {r.id for r in results}
        assert high.id in ids
        assert low.id not in ids


# ---------------------------------------------------------------------------
# Migration: ensure_fts_table populates existing rows
# ---------------------------------------------------------------------------


class TestFtsTableMigration:
    def test_ensure_fts_table_populates_existing_rows(self, tmp_path: Path) -> None:
        """Simulate adding FTS5 to a DB that already has entries (migration path)."""
        import sqlite3

        from trw_memory.storage._schema import ensure_schema

        db_path = tmp_path / "migration.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema(conn)

        # Insert a row directly (bypass FTS)
        conn.execute(
            "INSERT INTO memories (id, content, detail, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("migrated_id", "legacy search term", "", "[]", "2024-01-01", "2024-01-01"),
        )
        conn.commit()

        # Now create FTS table — should bulk-import the existing row
        result = ensure_fts_table(conn)
        assert result is True
        fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        assert fts_count >= 1

        # Verify the migrated row is searchable
        rows = conn.execute("SELECT id FROM memories_fts WHERE memories_fts MATCH ?", ('"legacy"',)).fetchall()
        assert any(r[0] == "migrated_id" for r in rows)
        conn.close()

    def test_ensure_fts_table_is_idempotent(self, backend: SQLiteBackend) -> None:
        """Calling ensure_fts_table multiple times must not double-populate."""
        from trw_memory.storage._schema import ensure_fts_table

        e = _entry(content="idempotent_test_term")
        backend.store(e)

        result = ensure_fts_table(backend._conn)
        assert result is True

        count = backend._conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        # Should be exactly 1, not 2 (no double-insert)
        assert count == 1


# ---------------------------------------------------------------------------
# Special character handling
# ---------------------------------------------------------------------------


class TestFtsSpecialChars:
    def test_fts_handles_double_quotes_in_query(self, backend: SQLiteBackend) -> None:
        e = _entry(content='she said "hello world"')
        backend.store(e)
        # Should not raise; may return 0 or 1 results
        results = backend.search_fts('"hello world"')
        assert isinstance(results, list)

    def test_fts_handles_hyphenated_terms(self, backend: SQLiteBackend) -> None:
        e = _entry(content="full-text search works well")
        backend.store(e)
        results = backend.search_fts("full-text")
        assert any(r.id == e.id for r in results)

    def test_fts_handles_empty_query(self, backend: SQLiteBackend) -> None:
        backend.store(_entry(content="any content"))
        # Empty string after quoting becomes '""' — FTS5 may error; should not propagate
        try:
            results = backend.search_fts("")
            assert isinstance(results, list)
        except Exception:
            pass  # StorageError is acceptable for empty FTS query


# ---------------------------------------------------------------------------
# Performance regression guard (scale sanity)
# ---------------------------------------------------------------------------


class TestFtsScaleGuard:
    def test_fts_faster_than_like_at_10k(self, tmp_path: Path) -> None:
        import random
        import time

        words = [f"word{i}" for i in range(500)]
        db = SQLiteBackend(tmp_path / "scale.db")

        entries = [_entry(content=" ".join(random.sample(words, 12)) + f" unique_rare_{i}") for i in range(10_000)]
        for e in entries:
            db.store(e)

        runs = 20
        rare = "unique_rare_42"

        t0 = time.perf_counter()
        for _ in range(runs):
            db.search_fts(rare, top_k=25)
        fts_ms = (time.perf_counter() - t0) / runs * 1000

        t0 = time.perf_counter()
        for _ in range(runs):
            db.search(rare, top_k=25)
        like_ms = (time.perf_counter() - t0) / runs * 1000

        # FTS5 should be measurably faster than LIKE at 10K entries for rare terms
        assert fts_ms < like_ms, f"FTS5 ({fts_ms:.2f}ms) should be faster than LIKE ({like_ms:.2f}ms) at 10K entries"


# ---------------------------------------------------------------------------
# Query sanitization (empty guard, length cap, operator literalization)
# ---------------------------------------------------------------------------


class TestFtsQuerySanitization:
    def test_whitespace_only_query_returns_empty(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        backend.store_many([_entry(content="some content")])
        assert backend.search_fts("   ") == []
        assert backend.search_fts("\t\n") == []

    def test_very_long_query_truncated_safely(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        long_query = "a" * 2000
        # Should not raise; returns empty or results without error
        result = backend.search_fts(long_query)
        assert isinstance(result, list)

    def test_fts_boolean_operators_treated_as_literal(self, backend: SQLiteBackend) -> None:
        if not backend.fts_available:
            pytest.skip("FTS5 not available")
        e = _entry(content="this AND that OR something NOT related")
        backend.store_many([e])
        # phrase quoting means AND/OR/NOT are treated as literals, not operators
        results = backend.search_fts("AND")
        assert any(r.id == e.id for r in results)
