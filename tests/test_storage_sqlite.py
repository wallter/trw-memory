"""Tests for SQLiteBackend.

Covers:
- store / get round-trip with full field verification
- get on unknown id returns None
- store overwrites existing entry (INSERT OR REPLACE)
- delete removes entry, returns True/False appropriately
- count with and without namespace filter
- list_entries with and without status filter
- search keyword matching (content + detail + tags)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pytest

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    entry_id: str,
    content: str = "test content",
    *,
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    namespace: str = "default",
    source: str = "agent",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        status=status,
        namespace=namespace,
        source=source,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> SQLiteBackend:  # type: ignore[misc]
    db = SQLiteBackend(tmp_path / "test.db")
    yield db  # type: ignore[misc]
    db.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestStoreAndGet:
    def test_store_and_get_round_trip(self, backend: SQLiteBackend) -> None:
        entry = make_entry(
            "e1",
            "pydantic validation error",
            detail="Use model_config strict=True",
            tags=["pydantic", "validation"],
            importance=0.8,
            namespace="proj",
        )
        backend.store(entry)
        result = backend.get("e1")

        assert result is not None
        assert result.id == "e1"
        assert result.content == "pydantic validation error"
        assert result.detail == "Use model_config strict=True"
        assert result.tags == ["pydantic", "validation"]
        assert result.importance == pytest.approx(0.8)
        assert result.namespace == "proj"

    def test_get_preserves_status(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e2", status=MemoryStatus.RESOLVED)
        backend.store(entry)
        result = backend.get("e2")
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_get_preserves_metadata(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e3")
        entry = entry.model_copy(update={"metadata": {"sprint": "31", "pr": "42"}})
        backend.store(entry)
        result = backend.get("e3")
        assert result is not None
        assert result.metadata == {"sprint": "31", "pr": "42"}

    def test_get_preserves_evidence(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e4")
        entry = entry.model_copy(update={"evidence": ["log1", "log2"]})
        backend.store(entry)
        result = backend.get("e4")
        assert result is not None
        assert result.evidence == ["log1", "log2"]


class TestGetNonexistent:
    def test_get_nonexistent_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.get("does-not-exist")
        assert result is None

    def test_get_empty_string_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.get("")
        assert result is None


class TestStoreOverwrite:
    def test_store_overwrites_existing_entry(self, backend: SQLiteBackend) -> None:
        entry_v1 = make_entry("dup", "first version")
        backend.store(entry_v1)

        entry_v2 = make_entry("dup", "second version — updated")
        backend.store(entry_v2)

        result = backend.get("dup")
        assert result is not None
        assert result.content == "second version — updated"

    def test_overwrite_preserves_new_fields(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("dup2", importance=0.3))
        backend.store(make_entry("dup2", importance=0.9))
        result = backend.get("dup2")
        assert result is not None
        assert result.importance == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_entry(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del1"))
        assert backend.delete("del1")
        assert backend.get("del1") is None

    def test_delete_returns_true_when_existed(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del2"))
        assert backend.delete("del2")

    def test_delete_nonexistent_returns_false(self, backend: SQLiteBackend) -> None:
        assert not backend.delete("no-such-entry")

    def test_double_delete_returns_false_second_time(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del3"))
        backend.delete("del3")
        assert backend.delete("del3") is False


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_empty_store(self, backend: SQLiteBackend) -> None:
        assert backend.count() == 0

    def test_count_returns_total(self, backend: SQLiteBackend) -> None:
        for i in range(3):
            backend.store(make_entry(f"c{i}"))
        assert backend.count() == 3

    def test_count_with_namespace(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("ns1a", namespace="ns1"))
        backend.store(make_entry("ns1b", namespace="ns1"))
        backend.store(make_entry("ns2a", namespace="ns2"))

        assert backend.count("ns1") == 2
        assert backend.count("ns2") == 1

    def test_count_namespace_no_match_returns_zero(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("x", namespace="other"))
        assert backend.count("nonexistent") == 0

    def test_count_decrements_after_delete(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("d1"))
        backend.store(make_entry("d2"))
        backend.delete("d1")
        assert backend.count() == 1


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------


class TestListEntries:
    def test_list_entries_returns_all(self, backend: SQLiteBackend) -> None:
        for i in range(4):
            backend.store(make_entry(f"le{i}"))
        entries = backend.list_entries()
        assert len(entries) == 4

    def test_list_entries_filtered_by_status_active(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("active1", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("active2", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("resolved1", status=MemoryStatus.RESOLVED))

        results = backend.list_entries(status=MemoryStatus.ACTIVE)
        ids = {e.id for e in results}
        assert "active1" in ids
        assert "active2" in ids
        assert "resolved1" not in ids

    def test_list_entries_filtered_by_status_resolved(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("a", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("r", status=MemoryStatus.RESOLVED))

        results = backend.list_entries(status=MemoryStatus.RESOLVED)
        assert len(results) == 1
        assert results[0].id == "r"

    def test_list_entries_respects_limit(self, backend: SQLiteBackend) -> None:
        for i in range(10):
            backend.store(make_entry(f"lim{i}"))
        results = backend.list_entries(limit=3)
        assert len(results) == 3

    def test_list_entries_empty_store_returns_empty(self, backend: SQLiteBackend) -> None:
        assert backend.list_entries() == []


class TestEntriesWithAssertions:
    def test_count_with_assertions_returns_only_assertion_entries(self, backend: SQLiteBackend) -> None:
        with_assertions = make_entry("a1").model_copy(
            update={
                "assertions": [
                    Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/main.py")
                ]
            }
        )
        without_assertions = make_entry("a2")

        backend.store(with_assertions)
        backend.store(without_assertions)

        results = backend.count_with_assertions()
        assert len(results) == 1
        assert results[0].id == "a1"


class TestIncrementSessionCounts:
    def test_increment_session_counts_updates_existing_rows(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("L-sess001"))
        backend.store(make_entry("L-sess002"))

        updated = backend.increment_session_counts(["L-sess001", "L-sess002"])

        assert updated == 2
        assert backend.get("L-sess001") is not None
        assert backend.get("L-sess001").session_count == 1  # type: ignore[union-attr]
        assert backend.get("L-sess002") is not None
        assert backend.get("L-sess002").session_count == 1  # type: ignore[union-attr]

    def test_increment_session_counts_uses_single_commit(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("L-batch01"))
        backend.store(make_entry("L-batch02"))
        backend.store(make_entry("L-batch03"))

        statements: list[str] = []
        backend._conn.set_trace_callback(statements.append)
        try:
            backend.increment_session_counts(["L-batch01", "L-batch02", "L-batch03"])
        finally:
            backend._conn.set_trace_callback(None)

        commit_count = sum(1 for statement in statements if statement.upper().startswith("COMMIT"))
        assert commit_count == 1

    def test_increment_session_counts_stays_under_latency_budget_for_25_rows(self, backend: SQLiteBackend) -> None:
        for index in range(25):
            backend.store(make_entry(f"L-lat{index:04d}"))

        entry_ids = [f"L-lat{index:04d}" for index in range(25)]

        start = time.perf_counter()
        updated = backend.increment_session_counts(entry_ids)
        elapsed = time.perf_counter() - start

        assert updated == 25
        assert elapsed < 0.05


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_keyword_in_content(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("p1", "pydantic validation error with strict mode"))
        backend.store(make_entry("p2", "fastmcp middleware registration pattern"))
        backend.store(make_entry("p3", "structlog reserved keyword event"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "p1" in ids
        assert "p2" not in ids
        assert "p3" not in ids

    def test_search_keyword_in_detail(self, backend: SQLiteBackend) -> None:
        entry = make_entry("detail_match", "short content", detail="pydantic v2 strict mode")
        backend.store(entry)
        backend.store(make_entry("no_match", "something else"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "detail_match" in ids

    def test_search_keyword_in_tags(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("tagged", tags=["pydantic-v2", "models"]))
        backend.store(make_entry("untagged"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "tagged" in ids

    def test_search_case_insensitive(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("upper", "PYDANTIC VALIDATION"))
        results = backend.search("pydantic")
        assert any(e.id == "upper" for e in results)

    def test_search_no_match_returns_empty(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("unrelated", "something completely different"))
        results = backend.search("zzz_unlikely_term")
        assert results == []

    def test_search_top_k_limits(self, backend: SQLiteBackend) -> None:
        for i in range(10):
            backend.store(make_entry(f"s{i}", f"python code item {i}"))
        results = backend.search("python", top_k=3)
        assert len(results) <= 3

    def test_search_status_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("active_match", "pydantic thing", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("resolved_match", "pydantic thing", status=MemoryStatus.RESOLVED))

        results = backend.search("pydantic", status=MemoryStatus.ACTIVE)
        ids = {e.id for e in results}
        assert "active_match" in ids
        assert "resolved_match" not in ids

    def test_search_namespace_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("ns1", "pydantic hint", namespace="team"))
        backend.store(make_entry("ns2", "pydantic hint", namespace="global"))

        results = backend.search("pydantic", namespace="team")
        ids = {e.id for e in results}
        assert "ns1" in ids
        assert "ns2" not in ids

    def test_search_min_importance_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("low", "pydantic thing", importance=0.2))
        backend.store(make_entry("high", "pydantic thing", importance=0.9))

        results = backend.search("pydantic", min_importance=0.5)
        ids = {e.id for e in results}
        assert "high" in ids
        assert "low" not in ids

    def test_search_tags_all_of_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("both", tags=["python", "pydantic"]))
        backend.store(make_entry("one", tags=["python"]))

        results = backend.search("python", tags=["python", "pydantic"])
        ids = {e.id for e in results}
        assert "both" in ids
        assert "one" not in ids

    def test_search_matches_entry_id(self, backend: SQLiteBackend) -> None:
        """FIX-055: search LIKE clause includes the id column."""
        backend.store(make_entry("L-b6911941", "some content about patterns"))
        backend.store(make_entry("L-deadbeef", "unrelated content"))

        results = backend.search("L-b6911941")
        ids = [e.id for e in results]
        assert "L-b6911941" in ids
        assert "L-deadbeef" not in ids

    def test_search_partial_id_match(self, backend: SQLiteBackend) -> None:
        """FIX-055: partial ID substring matches via LIKE."""
        backend.store(make_entry("L-b6911941", "test content"))
        results = backend.search("b6911941")
        assert any(e.id == "L-b6911941" for e in results)


# ---------------------------------------------------------------------------
# Update (partial field update)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_importance(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u1", importance=0.3))
        result = backend.update("u1", importance=0.9)
        assert result is not None
        assert result.importance == pytest.approx(0.9)

    def test_update_nonexistent_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.update("no-such-id", importance=0.9)
        assert result is None

    def test_update_no_fields_returns_current(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u2", "original"))
        result = backend.update("u2")
        assert result is not None
        assert result.content == "original"

    def test_update_status(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u3", status=MemoryStatus.ACTIVE))
        result = backend.update("u3", status=MemoryStatus.RESOLVED)
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_tags(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u4", tags=["old"]))
        result = backend.update("u4", tags=["new", "updated"])
        assert result is not None
        assert result.tags == ["new", "updated"]


# ---------------------------------------------------------------------------
# Cross-thread safety (check_same_thread=False)
# ---------------------------------------------------------------------------


class TestCrossThreadSafety:
    """Verify SQLiteBackend can be used from multiple threads."""

    def test_store_and_get_from_different_thread(self, backend: SQLiteBackend) -> None:
        """Store from main thread, get from a worker thread."""
        import threading

        backend.store(make_entry("cross-1", "shared data"))
        result_holder: list[MemoryEntry | None] = [None]

        def worker() -> None:
            result_holder[0] = backend.get("cross-1")

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        assert result_holder[0] is not None
        assert result_holder[0].content == "shared data"

    def test_concurrent_writes_from_threads(self, backend: SQLiteBackend) -> None:
        """Multiple threads can write concurrently without ProgrammingError."""
        import threading

        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(10):
                    backend.store(make_entry(f"t{thread_id}-{i}", f"data-{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert backend.count() == 40


# ===========================================================================
# FR03 (PRD-QUAL-038): SQLite Branch Coverage Tests
# ===========================================================================


class TestUpdateBranchCoverage:
    """FR03: Branch coverage for update() with various field types."""

    def test_update_list_field_tags(self, backend: SQLiteBackend) -> None:
        """update(entry_id, tags=['new']) serializes tags as JSON."""
        backend.store(make_entry("u-tags", tags=["old"]))
        result = backend.update("u-tags", tags=["new", "added"])
        assert result is not None
        assert result.tags == ["new", "added"]

    def test_update_list_field_evidence(self, backend: SQLiteBackend) -> None:
        """update with evidence list serializes correctly."""
        backend.store(make_entry("u-ev"))
        result = backend.update("u-ev", evidence=["log1.txt", "trace.json"])
        assert result is not None
        assert result.evidence == ["log1.txt", "trace.json"]

    def test_update_dict_field_metadata(self, backend: SQLiteBackend) -> None:
        """update(entry_id, metadata={'key': 'val'}) serializes as JSON."""
        backend.store(make_entry("u-meta"))
        result = backend.update("u-meta", metadata={"sprint": "52", "agent": "tm-b"})
        assert result is not None
        assert result.metadata == {"sprint": "52", "agent": "tm-b"}

    def test_update_dict_field_vector_clock(self, backend: SQLiteBackend) -> None:
        """update with vector_clock dict serializes correctly."""
        backend.store(make_entry("u-vc"))
        result = backend.update("u-vc", vector_clock={"node-1": 3, "node-2": 1})
        assert result is not None
        # vector_clock stores as dict[str, int]; serialized through JSON
        assert result.vector_clock["node-1"] == 3
        assert result.vector_clock["node-2"] == 1

    def test_update_datetime_field(self, backend: SQLiteBackend) -> None:
        """update with datetime value converts to isoformat."""
        backend.store(make_entry("u-dt"))
        from datetime import datetime, timezone

        new_time = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        result = backend.update("u-dt", last_accessed_at=new_time)
        assert result is not None
        assert result.last_accessed_at is not None
        assert result.last_accessed_at.year == 2026
        assert result.last_accessed_at.month == 3

    def test_update_memory_status_enum(self, backend: SQLiteBackend) -> None:
        """update with MemoryStatus.RESOLVED extracts .value correctly."""
        backend.store(make_entry("u-status", status=MemoryStatus.ACTIVE))
        result = backend.update("u-status", status=MemoryStatus.RESOLVED)
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_invalid_column_raises(self, backend: SQLiteBackend) -> None:
        """update(entry_id, id='hacked') raises StorageError for immutable field."""
        from trw_memory.exceptions import StorageError

        backend.store(make_entry("u-invalid"))
        with pytest.raises(StorageError, match="Invalid update field"):
            backend.update("u-invalid", id="hacked")

    def test_update_auto_sets_updated_at(self, backend: SQLiteBackend) -> None:
        """update without explicit updated_at auto-sets it to now."""
        entry = make_entry("u-auto")
        backend.store(entry)
        original_updated = entry.updated_at

        result = backend.update("u-auto", importance=0.9)
        assert result is not None
        # updated_at should have been auto-set to a recent time
        assert result.updated_at >= original_updated


class TestSearchLikeEscaping:
    """FR03: LIKE metacharacter escaping in search()."""

    def test_search_like_metachar_percent(self, backend: SQLiteBackend) -> None:
        """search('%admin%') escapes % so it matches literal '%admin%' only."""
        backend.store(make_entry("literal-pct", "100%admin% access"))
        backend.store(make_entry("no-pct", "regular admin access"))
        results = backend.search("%admin%")
        ids = [e.id for e in results]
        assert "literal-pct" in ids
        # 'no-pct' should NOT match because % is escaped (literal)
        assert "no-pct" not in ids

    def test_search_like_metachar_underscore(self, backend: SQLiteBackend) -> None:
        """search('admin_test') escapes _ so it matches literal underscore only."""
        backend.store(make_entry("literal-us", "admin_test configuration"))
        backend.store(make_entry("no-us", "adminXtest configuration"))
        results = backend.search("admin_test")
        ids = [e.id for e in results]
        assert "literal-us" in ids
        # Without escaping, _ would match any single char including X
        assert "no-us" not in ids

    def test_search_like_metachar_backslash(self, backend: SQLiteBackend) -> None:
        """Backslash in search query is escaped to prevent LIKE wildcard interpretation."""
        backend.store(make_entry("bs-entry", "C:\\path\\to\\file config"))
        backend.store(make_entry("no-bs", "Cpath_to_file config"))
        results = backend.search("C:\\path")
        ids = [e.id for e in results]
        assert "bs-entry" in ids


class TestDeleteVectorAbsentRow:
    """FR03: _delete_vector when no vec_index row is a no-op."""

    def test_delete_vector_absent_row(self, backend: SQLiteBackend) -> None:
        """_delete_vector for a nonexistent entry_id is a no-op.

        When sqlite-vec is not installed the vec_index table does not exist,
        so we create it manually to exercise the _delete_vector branch.
        """
        backend.store(make_entry("no-vec", "test content"))

        # If vec_index doesn't exist (no sqlite-vec), create it for this test
        if not backend._vec_available:
            backend._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_index ("
                "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
                "entry_id TEXT UNIQUE NOT NULL)"
            )
            backend._conn.commit()

        # Calling _delete_vector directly should not raise (entry has no vec row)
        backend._delete_vector("no-vec")
        # Entry should still exist in memories table
        preserved = backend.get("no-vec")
        assert preserved is not None
        assert preserved.id == "no-vec"
        assert preserved.content == "test content"


class TestStoredEmbeddings:
    def test_get_stored_embeddings_returns_empty_when_vec_unavailable(self, backend: SQLiteBackend) -> None:
        if backend._vec_available:
            pytest.skip("sqlite-vec available; use round-trip test instead")
        assert backend.get_stored_embeddings(["missing"]) == {}

    def test_get_stored_embeddings_round_trip(self, backend: SQLiteBackend) -> None:
        pytest.importorskip("sqlite_vec")
        entry = make_entry("vec-entry", "vector content")
        backend.store(entry)
        backend.upsert_vector(entry.id, [0.1] * backend._dim)

        embeddings = backend.get_stored_embeddings([entry.id, "missing"])

        assert set(embeddings) == {entry.id}
        assert embeddings[entry.id] == pytest.approx([0.1] * backend._dim)


class TestListEntriesCombinedFilters:
    """FR03: list_entries with combined status + namespace filters."""

    def test_list_entries_combined_filters(self, backend: SQLiteBackend) -> None:
        """list_entries(status=active, namespace='test') applies both filters."""
        backend.store(make_entry("a1", status=MemoryStatus.ACTIVE, namespace="test"))
        backend.store(make_entry("a2", status=MemoryStatus.ACTIVE, namespace="other"))
        backend.store(make_entry("r1", status=MemoryStatus.RESOLVED, namespace="test"))
        backend.store(make_entry("r2", status=MemoryStatus.RESOLVED, namespace="other"))

        results = backend.list_entries(status=MemoryStatus.ACTIVE, namespace="test")
        ids = {e.id for e in results}
        assert ids == {"a1"}


class TestDeleteByNamespace:
    """FR03: delete_by_namespace with populated and empty namespaces."""

    def test_delete_by_namespace_populated(self, backend: SQLiteBackend) -> None:
        """delete_by_namespace('test-ns') with 3 entries returns 3."""
        for i in range(3):
            backend.store(make_entry(f"dns-{i}", namespace="test-ns"))
        backend.store(make_entry("other-ns", namespace="keep"))

        deleted = backend.delete_by_namespace("test-ns")
        assert deleted == 3
        assert backend.count("test-ns") == 0
        assert backend.count("keep") == 1

    def test_delete_by_namespace_empty(self, backend: SQLiteBackend) -> None:
        """delete_by_namespace('empty-ns') with 0 entries returns 0."""
        backend.store(make_entry("x", namespace="different"))
        deleted = backend.delete_by_namespace("empty-ns")
        assert deleted == 0


class TestCountWithNamespaceFilter:
    """FR03: count(namespace='specific') returns filtered count."""

    def test_count_with_namespace_filter(self, backend: SQLiteBackend) -> None:
        """count(namespace='specific') returns only that namespace's count."""
        for i in range(4):
            backend.store(make_entry(f"spec-{i}", namespace="specific"))
        for i in range(2):
            backend.store(make_entry(f"other-{i}", namespace="other"))

        assert backend.count(namespace="specific") == 4
        assert backend.count(namespace="other") == 2
        assert backend.count() == 6
