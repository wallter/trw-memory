"""Tests for YAMLBackend.

Covers:
- store / get round-trip with field verification
- store creates a {id}.yaml file on disk
- get on unknown id returns None
- delete removes file, returns True/False appropriately
- update modifies a specific field via read-modify-write
- count with and without namespace filter
- search keyword matching (content + detail + tags)
- list_entries with limit
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.yaml_backend import YAMLBackend

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
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> YAMLBackend:
    return YAMLBackend(tmp_path / "entries")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestStoreAndGet:
    def test_store_and_get_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry(
            "e1",
            "pydantic validation error",
            detail="Use model_config strict=True",
            tags=["pydantic", "models"],
            importance=0.8,
            namespace="proj",
        )
        backend.store(entry)
        result = backend.get("e1")

        assert result is not None
        assert result.id == "e1"
        assert result.content == "pydantic validation error"
        assert result.detail == "Use model_config strict=True"
        assert result.tags == ["pydantic", "models"]
        assert result.importance == pytest.approx(0.8)
        assert result.namespace == "proj"

    def test_store_creates_yaml_file(self, backend: YAMLBackend, tmp_path: Path) -> None:
        backend.store(make_entry("file_check"))
        expected = tmp_path / "entries" / "file_check.yaml"
        assert expected.exists()

    def test_store_yaml_file_is_human_readable(self, backend: YAMLBackend, tmp_path: Path) -> None:
        backend.store(make_entry("readable", "some important content"))
        yaml_file = tmp_path / "entries" / "readable.yaml"
        raw = yaml_file.read_text(encoding="utf-8")
        assert "some important content" in raw

    def test_get_preserves_status(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("e2", status=MemoryStatus.ARCHIVED))
        result = backend.get("e2")
        assert result is not None
        assert result.status == MemoryStatus.ARCHIVED

    def test_get_preserves_evidence(self, backend: YAMLBackend) -> None:
        entry = make_entry("e3")
        entry = entry.model_copy(update={"evidence": ["ref1", "ref2"]})
        backend.store(entry)
        result = backend.get("e3")
        assert result is not None
        assert result.evidence == ["ref1", "ref2"]

    def test_get_nonexistent_returns_none(self, backend: YAMLBackend) -> None:
        assert backend.get("does-not-exist") is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_file(self, backend: YAMLBackend, tmp_path: Path) -> None:
        backend.store(make_entry("del1"))
        backend.delete("del1")
        assert not (tmp_path / "entries" / "del1.yaml").exists()

    def test_delete_returns_true_when_existed(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("del2"))
        assert backend.delete("del2") is True

    def test_delete_get_returns_none_after_delete(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("del3"))
        backend.delete("del3")
        assert backend.get("del3") is None

    def test_delete_nonexistent_returns_false(self, backend: YAMLBackend) -> None:
        assert backend.delete("not-here") is False

    def test_double_delete_returns_false_second_time(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("del4"))
        backend.delete("del4")
        assert backend.delete("del4") is False


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_importance(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("u1", importance=0.3))
        result = backend.update("u1", importance=0.9)
        assert result is not None
        assert result.importance == pytest.approx(0.9)

    def test_update_status(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("u2", status=MemoryStatus.ACTIVE))
        result = backend.update("u2", status=MemoryStatus.RESOLVED)
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_tags(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("u3", tags=["old"]))
        result = backend.update("u3", tags=["new", "updated"])
        assert result is not None
        assert result.tags == ["new", "updated"]

    def test_update_nonexistent_returns_none(self, backend: YAMLBackend) -> None:
        result = backend.update("no-such-id", importance=0.8)
        assert result is None

    def test_update_persists_to_disk(self, backend: YAMLBackend, tmp_path: Path) -> None:
        backend.store(make_entry("persist", importance=0.2))
        backend.update("persist", importance=0.95)
        # Re-read from disk to verify persistence
        result = backend.get("persist")
        assert result is not None
        assert result.importance == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_empty_store(self, backend: YAMLBackend) -> None:
        assert backend.count() == 0

    def test_count_returns_total(self, backend: YAMLBackend) -> None:
        for i in range(3):
            backend.store(make_entry(f"c{i}"))
        assert backend.count() == 3

    def test_count_with_namespace(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("n1a", namespace="ns1"))
        backend.store(make_entry("n1b", namespace="ns1"))
        backend.store(make_entry("n2a", namespace="ns2"))

        assert backend.count("ns1") == 2
        assert backend.count("ns2") == 1

    def test_count_decrements_after_delete(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("d1"))
        backend.store(make_entry("d2"))
        backend.delete("d1")
        assert backend.count() == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_keyword_in_content(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("p1", "pydantic validation error"))
        backend.store(make_entry("p2", "fastmcp middleware pattern"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "p1" in ids
        assert "p2" not in ids

    def test_search_keyword_in_detail(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("detail_match", detail="pydantic v2 strict mode"))
        backend.store(make_entry("no_match", "unrelated"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "detail_match" in ids

    def test_search_keyword_in_tags(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("tagged", tags=["pydantic-v2"]))
        backend.store(make_entry("untagged", "unrelated"))

        results = backend.search("pydantic")
        ids = [e.id for e in results]
        assert "tagged" in ids

    def test_search_case_insensitive(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("upper", "PYDANTIC MODEL"))
        results = backend.search("pydantic")
        assert any(e.id == "upper" for e in results)

    def test_search_no_match_returns_empty(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("other", "completely different topic"))
        assert backend.search("zzz_unlikely_term") == []

    def test_search_top_k_limits(self, backend: YAMLBackend) -> None:
        for i in range(8):
            backend.store(make_entry(f"s{i}", f"python code {i}"))
        results = backend.search("python", top_k=3)
        assert len(results) <= 3

    def test_search_status_filter(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("active_match", "pydantic thing", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("resolved_match", "pydantic thing", status=MemoryStatus.RESOLVED))

        results = backend.search("pydantic", status=MemoryStatus.ACTIVE)
        ids = {e.id for e in results}
        assert "active_match" in ids
        assert "resolved_match" not in ids

    def test_search_min_importance_filter(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("low", "pydantic thing", importance=0.1))
        backend.store(make_entry("high", "pydantic thing", importance=0.9))

        results = backend.search("pydantic", min_importance=0.5)
        ids = {e.id for e in results}
        assert "high" in ids
        assert "low" not in ids

    def test_search_tags_all_of_filter(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("both", tags=["python", "pydantic"]))
        backend.store(make_entry("one_tag", tags=["python"]))

        results = backend.search("python", tags=["python", "pydantic"])
        ids = {e.id for e in results}
        assert "both" in ids
        assert "one_tag" not in ids

    def test_search_namespace_filter(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("ns1", "pydantic hint", namespace="team"))
        backend.store(make_entry("ns2", "pydantic hint", namespace="global"))

        results = backend.search("pydantic", namespace="team")
        ids = {e.id for e in results}
        assert "ns1" in ids
        assert "ns2" not in ids


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------


class TestListEntries:
    def test_list_entries_returns_all(self, backend: YAMLBackend) -> None:
        for i in range(4):
            backend.store(make_entry(f"le{i}"))
        entries = backend.list_entries()
        assert len(entries) == 4

    def test_list_entries_with_limit(self, backend: YAMLBackend) -> None:
        for i in range(5):
            backend.store(make_entry(f"lim{i}"))
        results = backend.list_entries(limit=2)
        assert len(results) == 2

    def test_list_entries_filtered_by_status(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("active1", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("active2", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("resolved1", status=MemoryStatus.RESOLVED))

        results = backend.list_entries(status=MemoryStatus.ACTIVE)
        ids = {e.id for e in results}
        assert "active1" in ids
        assert "active2" in ids
        assert "resolved1" not in ids

    def test_list_entries_filtered_by_namespace(self, backend: YAMLBackend) -> None:
        backend.store(make_entry("ns1a", namespace="team"))
        backend.store(make_entry("ns2a", namespace="global"))

        results = backend.list_entries(namespace="team")
        assert len(results) == 1
        assert results[0].id == "ns1a"

    def test_list_entries_empty_store_returns_empty(self, backend: YAMLBackend) -> None:
        assert backend.list_entries() == []

    def test_close_is_noop(self, backend: YAMLBackend) -> None:
        # close() on YAMLBackend is a no-op — should not raise
        backend.close()
        backend.close()  # double-close also safe
