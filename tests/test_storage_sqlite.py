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

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
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
        deleted = backend.delete("del1")
        assert deleted is True
        assert backend.get("del1") is None

    def test_delete_returns_true_when_existed(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del2"))
        assert backend.delete("del2") is True

    def test_delete_nonexistent_returns_false(self, backend: SQLiteBackend) -> None:
        result = backend.delete("no-such-entry")
        assert result is False

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
