# ruff: noqa: F401,F811
"""SQLiteBackend search behavior tests."""

from __future__ import annotations

from trw_memory.models.memory import MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_storage_sqlite_support import backend, make_entry


class TestSearch:
    def test_search_keyword_in_content(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("p1", "pydantic validation error with strict mode"))
        backend.store(make_entry("p2", "fastmcp middleware registration pattern"))
        backend.store(make_entry("p3", "structlog reserved keyword event"))

        results = backend.search("pydantic")
        ids = [entry.id for entry in results]
        assert "p1" in ids
        assert "p2" not in ids
        assert "p3" not in ids

    def test_search_keyword_in_detail(self, backend: SQLiteBackend) -> None:
        entry = make_entry("detail_match", "short content", detail="pydantic v2 strict mode")
        backend.store(entry)
        backend.store(make_entry("no_match", "something else"))

        results = backend.search("pydantic")
        ids = [result.id for result in results]
        assert "detail_match" in ids

    def test_search_keyword_in_tags(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("tagged", tags=["pydantic-v2", "models"]))
        backend.store(make_entry("untagged"))

        results = backend.search("pydantic")
        ids = [entry.id for entry in results]
        assert "tagged" in ids

    def test_search_case_insensitive(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("upper", "PYDANTIC VALIDATION"))
        results = backend.search("pydantic")
        assert any(entry.id == "upper" for entry in results)

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
        ids = {entry.id for entry in results}
        assert "active_match" in ids
        assert "resolved_match" not in ids

    def test_search_namespace_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("ns1", "pydantic hint", namespace="team"))
        backend.store(make_entry("ns2", "pydantic hint", namespace="global"))

        results = backend.search("pydantic", namespace="team")
        ids = {entry.id for entry in results}
        assert "ns1" in ids
        assert "ns2" not in ids

    def test_search_min_importance_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("low", "pydantic thing", importance=0.2))
        backend.store(make_entry("high", "pydantic thing", importance=0.9))

        results = backend.search("pydantic", min_importance=0.5)
        ids = {entry.id for entry in results}
        assert "high" in ids
        assert "low" not in ids

    def test_search_tags_all_of_filter(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("both", tags=["python", "pydantic"]))
        backend.store(make_entry("one", tags=["python"]))

        results = backend.search("python", tags=["python", "pydantic"])
        ids = {entry.id for entry in results}
        assert "both" in ids
        assert "one" not in ids

    def test_search_matches_entry_id(self, backend: SQLiteBackend) -> None:
        """FIX-055: search LIKE clause includes the id column."""
        backend.store(make_entry("L-b6911941", "some content about patterns"))
        backend.store(make_entry("L-deadbeef", "unrelated content"))

        results = backend.search("L-b6911941")
        ids = [entry.id for entry in results]
        assert "L-b6911941" in ids
        assert "L-deadbeef" not in ids

    def test_search_partial_id_match(self, backend: SQLiteBackend) -> None:
        """FIX-055: partial ID substring matches via LIKE."""
        backend.store(make_entry("L-b6911941", "test content"))
        results = backend.search("b6911941")
        assert any(entry.id == "L-b6911941" for entry in results)


class TestSearchLikeEscaping:
    """FR03: LIKE metacharacter escaping in search()."""

    def test_search_like_metachar_percent(self, backend: SQLiteBackend) -> None:
        """search('%admin%') escapes % so it matches literal '%admin%' only."""
        backend.store(make_entry("literal-pct", "100%admin% access"))
        backend.store(make_entry("no-pct", "regular admin access"))
        results = backend.search("%admin%")
        ids = [entry.id for entry in results]
        assert "literal-pct" in ids
        assert "no-pct" not in ids

    def test_search_like_metachar_underscore(self, backend: SQLiteBackend) -> None:
        """search('admin_test') escapes _ so it matches literal underscore only."""
        backend.store(make_entry("literal-us", "admin_test configuration"))
        backend.store(make_entry("no-us", "adminXtest configuration"))
        results = backend.search("admin_test")
        ids = [entry.id for entry in results]
        assert "literal-us" in ids
        assert "no-us" not in ids

    def test_search_like_metachar_backslash(self, backend: SQLiteBackend) -> None:
        """Backslash in search query is escaped to prevent LIKE wildcard interpretation."""
        backend.store(make_entry("bs-entry", "C:\\path\\to\\file config"))
        backend.store(make_entry("no-bs", "Cpath_to_file config"))
        results = backend.search("C:\\path")
        ids = [entry.id for entry in results]
        assert "bs-entry" in ids
