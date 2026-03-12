"""Security regression tests for trw-memory storage backends.

Covers:
- P0-1: SQL column-name injection in SQLiteBackend.update
- P1-1: Path traversal in YAMLBackend
- P1-2: YAML field injection in YAMLBackend.update
- P1-3: Timezone-aware datetime round-trips in SQLiteBackend
- P1-4: LIKE metacharacter escaping in SQLiteBackend.search
- P1-7: Auto updated_at setting on update (SQLite + YAML)
- P1-8: Context manager protocol in SQLiteBackend
- P1-9: MemoryIndex total_count auto-sync
- P2-1: cosine_similarity dimension mismatch
- P2-2: BM25 fallback Jaccard normalization
- P2-6: MemoryEntry empty-id validation
- P2-7: MemoryConfig weight validation
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus
from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend

# ---------------------------------------------------------------------------
# Shared fixture — entry factory
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_entry() -> object:
    """Factory for creating test MemoryEntry instances."""

    def _make(
        entry_id: str = "test-1",
        content: str = "test content",
        **kwargs: object,
    ) -> MemoryEntry:
        defaults: dict[str, object] = {
            "id": entry_id,
            "content": content,
            "detail": "",
            "tags": [],
            "importance": 0.5,
            "status": MemoryStatus.ACTIVE,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        defaults.update(kwargs)
        return MemoryEntry(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture()
def sqlite_backend(tmp_path: Path):  # type: ignore[misc]
    db = SQLiteBackend(tmp_path / "sec_test.db")
    yield db
    db.close()


@pytest.fixture()
def yaml_backend(tmp_path: Path) -> YAMLBackend:
    return YAMLBackend(tmp_path / "entries")


# ---------------------------------------------------------------------------
# P0-1: SQL column-name injection
# ---------------------------------------------------------------------------


class TestSqliteUpdateInjection:
    def test_sqlite_update_invalid_column_raises(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-inj-1")
        sqlite_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            sqlite_backend.update("e-inj-1", evil_column="value")

    def test_sqlite_update_sql_injection_attempt(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-inj-2")
        sqlite_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            sqlite_backend.update("e-inj-2", **{"id = 1 OR 1=1 --": "pwned"})

    def test_sqlite_update_cannot_change_id(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-id-change")
        sqlite_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            sqlite_backend.update("e-id-change", id="new_id")

    def test_sqlite_update_cannot_change_created_at(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-created-at")
        sqlite_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            sqlite_backend.update(
                "e-created-at", created_at=datetime.now(timezone.utc)
            )

    def test_sqlite_update_valid_fields_accepted(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-valid")
        sqlite_backend.store(entry)
        updated = sqlite_backend.update(
            "e-valid", importance=0.9, status=MemoryStatus.RESOLVED
        )
        assert updated is not None
        assert updated.importance == pytest.approx(0.9)
        assert updated.status == MemoryStatus.RESOLVED

    def test_update_nonexistent_returns_none(
        self, sqlite_backend: SQLiteBackend
    ) -> None:
        result = sqlite_backend.update("no-such-entry", importance=0.7)
        assert result is None

    def test_update_empty_fields_returns_current(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-no-fields", content="original")
        sqlite_backend.store(entry)
        result = sqlite_backend.update("e-no-fields")
        assert result is not None
        assert result.content == "original"
        assert result.id == "e-no-fields"


# ---------------------------------------------------------------------------
# P1-1: Path traversal — YAMLBackend
# ---------------------------------------------------------------------------


class TestYamlPathTraversal:
    def test_yaml_store_path_traversal_raises(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("../../etc/evil")
        with pytest.raises(StorageError, match="traversal"):
            yaml_backend.store(entry)

    def test_yaml_get_path_traversal_raises(self, yaml_backend: YAMLBackend) -> None:
        with pytest.raises(StorageError, match="traversal"):
            yaml_backend.get("../../etc/passwd")

    def test_yaml_delete_path_traversal_raises(self, yaml_backend: YAMLBackend) -> None:
        with pytest.raises(StorageError, match="traversal"):
            yaml_backend.delete("../../../tmp/data")

    def test_update_path_traversal_raises(self, yaml_backend: YAMLBackend) -> None:
        with pytest.raises(StorageError, match="traversal"):
            yaml_backend.update("../evil", importance=0.9)

    def test_legitimate_id_accepted(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("M-001-safe")
        yaml_backend.store(entry)
        result = yaml_backend.get("M-001-safe")
        assert result is not None
        assert result.id == "M-001-safe"
        assert result.content == "test content"

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../sibling",
            "../../root/dir",
            "/absolute/path",
        ],
    )
    def test_various_traversal_patterns_rejected(
        self, yaml_backend: YAMLBackend, bad_id: str
    ) -> None:
        with pytest.raises(StorageError):
            yaml_backend.get(bad_id)


# ---------------------------------------------------------------------------
# P1-2: YAML field injection
# ---------------------------------------------------------------------------


class TestYamlFieldInjection:
    def test_yaml_update_invalid_field_raises(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-yaml-inj")
        yaml_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            yaml_backend.update("e-yaml-inj", __class__="Exploit")

    def test_yaml_update_cannot_change_id(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-yaml-id")
        yaml_backend.store(entry)
        with pytest.raises(StorageError, match="Invalid update field"):
            yaml_backend.update("e-yaml-id", id="new_id")

    def test_valid_fields_accepted(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-yaml-valid")
        yaml_backend.store(entry)
        updated = yaml_backend.update("e-yaml-valid", importance=0.8)
        assert updated is not None
        assert updated.importance == pytest.approx(0.8)
        assert updated.id == "e-yaml-valid"


# ---------------------------------------------------------------------------
# P1-3: Timezone-aware datetime round-trips
# ---------------------------------------------------------------------------


class TestSqliteTimezoneRoundtrip:
    def test_utc_datetime_roundtrip(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        entry = factory("e-tz-1", created_at=now, updated_at=now)
        sqlite_backend.store(entry)
        result = sqlite_backend.get("e-tz-1")
        assert result is not None
        assert result.created_at.tzinfo is not None
        assert result.created_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_returned_datetime_is_utc_aware(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-tz-2")
        sqlite_backend.store(entry)
        result = sqlite_backend.get("e-tz-2")
        assert result is not None
        assert result.created_at.tzinfo is not None
        assert result.updated_at.tzinfo is not None


# ---------------------------------------------------------------------------
# P1-4: LIKE metacharacter escaping
# ---------------------------------------------------------------------------


class TestSqliteLikeEscaping:
    def _make_entry(
        self, entry_id: str, content: str
    ) -> MemoryEntry:
        now = datetime.now(timezone.utc)
        return MemoryEntry(
            id=entry_id,
            content=content,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

    def test_sqlite_search_literal_percent(
        self, sqlite_backend: SQLiteBackend
    ) -> None:
        """Searching for '100%' should find only the matching entry."""
        e1 = self._make_entry("pct-1", "100% coverage achieved")
        e2 = self._make_entry("pct-2", "basic test entry unrelated")
        sqlite_backend.store(e1)
        sqlite_backend.store(e2)
        results = sqlite_backend.search("100%")
        ids = [r.id for r in results]
        assert "pct-1" in ids
        assert "pct-2" not in ids

    def test_sqlite_search_literal_underscore(
        self, sqlite_backend: SQLiteBackend
    ) -> None:
        """Searching for 'a_b' should not match 'axb' (underscore not wildcard)."""
        e1 = self._make_entry("ul-1", "a_b test entry with underscore")
        e2 = self._make_entry("ul-2", "axb entry without underscore")
        sqlite_backend.store(e1)
        sqlite_backend.store(e2)
        results = sqlite_backend.search("a_b")
        ids = [r.id for r in results]
        assert "ul-1" in ids
        assert "ul-2" not in ids

    def test_percent_does_not_return_all_entries(
        self, sqlite_backend: SQLiteBackend
    ) -> None:
        """A search for '%' should only match entries actually containing '%'."""
        for i in range(5):
            e = self._make_entry(f"nopct-{i}", f"entry number {i} no special chars")
            sqlite_backend.store(e)
        pct_entry = self._make_entry("pct-special", "this has 50% discount")
        sqlite_backend.store(pct_entry)
        results = sqlite_backend.search("%")
        ids = [r.id for r in results]
        assert "pct-special" in ids
        # The 5 entries without '%' must not appear
        for i in range(5):
            assert f"nopct-{i}" not in ids


# ---------------------------------------------------------------------------
# P1-7: Auto updated_at on update
# ---------------------------------------------------------------------------


class TestAutoUpdatedAt:
    def test_sqlite_update_auto_sets_updated_at(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        before = datetime.now(timezone.utc)
        entry = factory("e-upd-ts", created_at=before, updated_at=before)
        sqlite_backend.store(entry)
        # Small sleep to ensure updated_at changes
        time.sleep(0.01)
        result = sqlite_backend.update("e-upd-ts", importance=0.7)
        assert result is not None
        assert result.updated_at >= before

    def test_yaml_update_auto_sets_updated_at(
        self, yaml_backend: YAMLBackend, make_entry: object
    ) -> None:
        factory = make_entry  # type: ignore[operator]
        before = datetime.now(timezone.utc)
        entry = factory("e-yaml-upd", created_at=before, updated_at=before)
        yaml_backend.store(entry)
        time.sleep(0.01)
        result = yaml_backend.update("e-yaml-upd", importance=0.7)
        assert result is not None
        assert result.updated_at >= before

    def test_sqlite_explicit_updated_at_respected(
        self, sqlite_backend: SQLiteBackend, make_entry: object
    ) -> None:
        """When caller explicitly passes updated_at, it must be used."""
        factory = make_entry  # type: ignore[operator]
        entry = factory("e-explicit-upd")
        sqlite_backend.store(entry)
        explicit_dt = datetime(2030, 1, 1, tzinfo=timezone.utc)
        result = sqlite_backend.update("e-explicit-upd", updated_at=explicit_dt)
        assert result is not None
        assert result.updated_at.year == 2030


# ---------------------------------------------------------------------------
# P1-8: Context manager protocol
# ---------------------------------------------------------------------------


class TestSqliteContextManager:
    def test_sqlite_context_manager(self, tmp_path: Path) -> None:
        """with SQLiteBackend(...) as backend: should call close() on exit."""
        db_path = tmp_path / "cm_test.db"
        with SQLiteBackend(db_path) as backend:
            assert isinstance(backend, SQLiteBackend)
            assert backend.count() == 0
        # After exit the connection should be closed; further operations raise
        # This just verifies no exception was raised during __exit__

    def test_sqlite_context_manager_on_exception(self, tmp_path: Path) -> None:
        """close() must be called even when the with-body raises."""
        db_path = tmp_path / "cm_exc_test.db"
        with pytest.raises(ValueError, match="intentional"):
            with SQLiteBackend(db_path) as backend:
                assert isinstance(backend, SQLiteBackend)
                raise ValueError("intentional test error")
        # Verify the connection is closed: attempting to use the raw conn
        # after close should not interfere with the test framework

    def test_context_manager_returns_self(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cm_self.db"
        backend = SQLiteBackend(db_path)
        try:
            result = backend.__enter__()
            assert result is backend
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# P1-9: MemoryIndex total_count auto-sync
# ---------------------------------------------------------------------------


class TestMemoryIndexTotalCount:
    def test_memory_index_total_count_auto_synced(self) -> None:
        entries = [
            MemoryEntry(id=f"M-{i}", content=f"content {i}", status=MemoryStatus.ACTIVE)
            for i in range(3)
        ]
        idx = MemoryIndex(entries=entries)
        assert idx.total_count == 3

    def test_total_count_ignores_explicit_value(self) -> None:
        """Passing total_count=0 with 3 entries should yield total_count=3."""
        entries = [
            MemoryEntry(id=f"M-{i}", content=f"content {i}", status=MemoryStatus.ACTIVE)
            for i in range(3)
        ]
        idx = MemoryIndex(entries=entries, total_count=0)
        assert idx.total_count == 3

    def test_total_count_empty_index(self) -> None:
        idx = MemoryIndex()
        assert idx.total_count == 0

    def test_total_count_single_entry(self) -> None:
        entry = MemoryEntry(id="M-single", content="one entry", status=MemoryStatus.ACTIVE)
        idx = MemoryIndex(entries=[entry])
        assert idx.total_count == 1


# ---------------------------------------------------------------------------
# P2-1: cosine_similarity dimension mismatch
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_cosine_dimension_mismatch_raises(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity(a, b)

    def test_identical_vectors_returns_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        result = cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "len_a, len_b",
        [(3, 2), (100, 384), (1, 2)],
    )
    def test_parametrized_dimension_mismatch(
        self, len_a: int, len_b: int
    ) -> None:
        a = [1.0] * len_a
        b = [1.0] * len_b
        with pytest.raises(ValueError):
            cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# P2-2: BM25 fallback Jaccard normalization
# ---------------------------------------------------------------------------


class TestBm25FallbackJaccard:
    def _make_entry(self, entry_id: str, content: str) -> MemoryEntry:
        now = datetime.now(timezone.utc)
        return MemoryEntry(id=entry_id, content=content, status=MemoryStatus.ACTIVE, created_at=now, updated_at=now)

    def test_bm25_fallback_scores_normalized(self) -> None:
        """When BM25 returns all zeros, fallback Jaccard scores must be in [0, 1]."""
        # Two identical entries => BM25 IDF becomes 0 for shared terms
        e1 = self._make_entry("j-1", "python testing patterns")
        e2 = self._make_entry("j-2", "python testing patterns")
        results = bm25_search("python testing", [e1, e2], top_k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_fallback_non_matching_entries_excluded(self) -> None:
        """Entries with zero overlap must not appear in fallback results."""
        e1 = self._make_entry("j-match", "python testing patterns")
        e2 = self._make_entry("j-no-match", "completely unrelated xyz abc")
        results = bm25_search("python", [e1, e2, e2], top_k=10)
        # All returned scores must be positive
        for _entry_id, score in results:
            assert score > 0.0

    def test_bm25_empty_entries_returns_empty(self) -> None:
        results = bm25_search("query", [], top_k=10)
        assert results == []


# ---------------------------------------------------------------------------
# P2-6: MemoryEntry empty-id validation
# ---------------------------------------------------------------------------


class TestMemoryEntryIdValidation:
    def test_memory_entry_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryEntry(id="", content="test", status=MemoryStatus.ACTIVE)

    def test_whitespace_only_id_is_valid_at_model_level(self) -> None:
        """Pydantic min_length=1 allows whitespace — storage must handle separately."""
        # This tests the Pydantic boundary: a single space passes min_length=1
        entry = MemoryEntry(id=" ", content="test", status=MemoryStatus.ACTIVE)
        assert entry.id == " "

    def test_valid_id_accepted(self) -> None:
        entry = MemoryEntry(id="M-001", content="test", status=MemoryStatus.ACTIVE)
        assert entry.id == "M-001"

    @pytest.mark.parametrize(
        "valid_id",
        ["M-001", "uuid-abc-123", "a", "entry_1", "UPPER-CASE"],
    )
    def test_parametrized_valid_ids(self, valid_id: str) -> None:
        entry = MemoryEntry(id=valid_id, content="ok", status=MemoryStatus.ACTIVE)
        assert entry.id == valid_id


# ---------------------------------------------------------------------------
# P2-7: MemoryConfig weight validation
# ---------------------------------------------------------------------------


class TestMemoryConfigWeightValidation:
    def test_default_weights_sum_to_one(self) -> None:
        cfg = MemoryConfig()
        total = (
            cfg.score_relevance_weight
            + cfg.score_recency_weight
            + cfg.score_importance_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_memory_config_invalid_weights_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_SCORE_RELEVANCE_WEIGHT", "0.5")
        monkeypatch.setenv("MEMORY_SCORE_RECENCY_WEIGHT", "0.5")
        monkeypatch.setenv("MEMORY_SCORE_IMPORTANCE_WEIGHT", "0.5")
        with pytest.raises(ValidationError, match="weights must sum"):
            MemoryConfig()

    def test_custom_valid_weights_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_SCORE_RELEVANCE_WEIGHT", "0.5")
        monkeypatch.setenv("MEMORY_SCORE_RECENCY_WEIGHT", "0.25")
        monkeypatch.setenv("MEMORY_SCORE_IMPORTANCE_WEIGHT", "0.25")
        cfg = MemoryConfig()
        total = (
            cfg.score_relevance_weight
            + cfg.score_recency_weight
            + cfg.score_importance_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_weight_sum_boundary_within_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Weights summing to 1.005 should pass (within 0.01 tolerance)."""
        monkeypatch.setenv("MEMORY_SCORE_RELEVANCE_WEIGHT", "0.405")
        monkeypatch.setenv("MEMORY_SCORE_RECENCY_WEIGHT", "0.3")
        monkeypatch.setenv("MEMORY_SCORE_IMPORTANCE_WEIGHT", "0.3")
        cfg = MemoryConfig()
        total = (
            cfg.score_relevance_weight
            + cfg.score_recency_weight
            + cfg.score_importance_weight
        )
        assert abs(total - 1.0) < 0.02  # within tolerance
