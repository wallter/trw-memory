"""Behavior tests for SQLiteBackend.find_active_by_content (PRD-CORE-042).

Embedding-independent exact-content dedup lookup: equality match on content +
detail, scoped to active status + namespace. Read-only.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.memory import MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

from .conftest import make_entry


class TestFindActiveByContent:
    def test_returns_id_on_exact_active_match(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        backend.store(make_entry(entry_id="M-exact", content="hello world", detail="the body"))

        found = backend.find_active_by_content("hello world", "the body")
        assert found == "M-exact"

    def test_returns_none_when_content_differs(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        backend.store(make_entry(entry_id="M-1", content="hello world", detail="the body"))

        assert backend.find_active_by_content("hello world", "different body") is None
        assert backend.find_active_by_content("other content", "the body") is None

    def test_empty_detail_matches_via_coalesce(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        backend.store(make_entry(entry_id="M-nodetail", content="just content", detail=""))

        assert backend.find_active_by_content("just content", "") == "M-nodetail"

    def test_ignores_non_active_entries(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        backend.store(
            make_entry(
                entry_id="M-obsolete",
                content="dupe content",
                detail="dupe detail",
                status=MemoryStatus.OBSOLETE,
            )
        )

        # An obsolete exact match must NOT be returned — only active dupes merge.
        assert backend.find_active_by_content("dupe content", "dupe detail") is None

    def test_namespace_scoped(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        backend.store(
            make_entry(
                entry_id="M-team",
                content="scoped content",
                detail="scoped detail",
                namespace="team",
            )
        )

        # Default namespace lookup must not see the "team"-namespaced entry.
        assert backend.find_active_by_content("scoped content", "scoped detail") is None
        assert backend.find_active_by_content("scoped content", "scoped detail", namespace="team") == "M-team"

    def test_returns_none_when_no_entries(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "db.sqlite")
        assert backend.find_active_by_content("anything", "") is None
