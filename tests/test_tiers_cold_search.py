# ruff: noqa: F811
"""Tests for lifecycle/tiers.py cold search behavior and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers import TierManager

from ._test_tiers_support import mem_dir, mgr  # noqa: F401


class TestColdTier:
    def test_cold_promote_purges_sidecar_when_warm_rollback_raises(
        self,
        mgr: TierManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "06"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "e7.yaml"
        write_yaml(yaml_file, {"id": "e7", "content": "rollback purge"})

        original_unlink = Path.unlink
        original_warm_remove = mgr._warm_store.warm_remove

        def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
            if path == yaml_file:
                raise OSError("archive delete failed")
            original_unlink(path, missing_ok=missing_ok)

        def _raise_warm_remove(_entry_id: str) -> bool:
            raise OSError("warm cleanup failed")

        monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
        mgr._warm_store.warm_remove = _raise_warm_remove  # type: ignore[assignment]
        try:
            result = mgr.cold_promote("e7")
        finally:
            mgr._warm_store.warm_remove = original_warm_remove  # type: ignore[method-assign]

        assert result is None
        assert yaml_file.exists()
        assert mgr.warm_search(["rollback"], None, top_k=5) == []

    def test_cold_search_finds_matching_entry(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "02"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "entry1.yaml",
            {"id": "e1", "content": "machine learning basics", "tags": ["ml"]},
        )
        write_yaml(
            cold_partition / "entry2.yaml",
            {"id": "e2", "content": "cooking recipes", "tags": ["food"]},
        )

        results = mgr.cold_search(["machine", "learning"])
        ids = [result.get("id") for result in results]
        assert "e1" in ids
        assert "e2" not in ids

    def test_cold_search_empty_tokens_returns_empty(self, mgr: TierManager) -> None:
        assert mgr.cold_search([]) == []

    def test_cold_promote_restores_warm_vector_from_archived_embedding(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        class _FakeWarmBackend:
            def __init__(self) -> None:
                self.vectors: dict[str, list[float]] = {}

            def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
                self.vectors[entry_id] = embedding

        cold_partition = mgr._cold_dir() / "2026" / "04"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "vector-entry.yaml",
            {
                "id": "vector-entry",
                "content": "keyword promotion",
                "_warm_embedding": [1.0, 0.0],
            },
        )

        fake_backend = _FakeWarmBackend()
        mgr._warm_store._get_warm_backend = lambda dim=None: fake_backend  # type: ignore[assignment,return-value]

        result = mgr.cold_promote("vector-entry")

        assert result is not None
        assert fake_backend.vectors["vector-entry"] == [1.0, 0.0]

    def test_cold_search_no_cold_dir_returns_empty(self, mgr: TierManager) -> None:
        assert mgr.cold_search(["anything"]) == []

    def test_cold_search_by_tag(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "tagged.yaml",
            {"id": "tagged-e", "content": "something", "tags": ["special-tag"]},
        )
        results = mgr.cold_search(["special-tag"])
        ids = [result.get("id") for result in results]
        assert "tagged-e" in ids

    def test_cold_search_matches_detail_text(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2026" / "01"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(
            cold_partition / "detail-hit.yaml",
            {"id": "detail-hit", "content": "opaque title", "detail": "semantic migration note", "tags": []},
        )

        results = mgr.cold_search(["migration"])
        assert [result.get("id") for result in results] == ["detail-hit"]

    def test_path_traversal_guard(self, mgr: TierManager, tmp_path: Path) -> None:
        from trw_memory.storage.persistence import write_yaml

        outside = tmp_path.parent / "evil.yaml"
        write_yaml(outside, {"id": "evil", "content": "escape"})
        with pytest.raises(Exception):
            mgr.cold_archive("evil", outside)


class TestColdPromoteEdgeCases:
    """FR04: cold_promote() edge cases."""

    def test_cold_promote_found(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "12"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "promote-found.yaml"
        write_yaml(yaml_file, {"id": "promote-found", "content": "found me", "tags": ["test"]})

        result = mgr.cold_promote("promote-found")
        assert result is not None
        assert result["id"] == "promote-found"
        assert result["content"] == "found me"
        assert not yaml_file.exists()

    def test_cold_promote_not_found(self, mgr: TierManager) -> None:
        assert mgr.cold_promote("nonexistent-entry-xyz") is None

    def test_cold_promote_read_yaml_failure(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "11"
        cold_partition.mkdir(parents=True, exist_ok=True)
        (cold_partition / "corrupt.yaml").write_bytes(b"\xff\xfe not valid yaml!")
        valid_file = cold_partition / "valid.yaml"
        write_yaml(valid_file, {"id": "valid-entry", "content": "valid data"})

        result = mgr.cold_promote("valid-entry")
        assert result is not None
        assert result["id"] == "valid-entry"

    def test_cold_promote_updates_last_accessed(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        cold_partition = mgr._cold_dir() / "2025" / "10"
        cold_partition.mkdir(parents=True, exist_ok=True)
        yaml_file = cold_partition / "promote-accessed.yaml"
        old_time = "2024-01-01T00:00:00+00:00"
        write_yaml(
            yaml_file,
            {"id": "promote-accessed", "content": "old", "last_accessed_at": old_time},
        )

        result = mgr.cold_promote("promote-accessed")
        assert result is not None
        assert result["last_accessed_at"] != old_time
