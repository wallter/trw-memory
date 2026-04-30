from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import _merge_tier_entries, memory_recall_impl


class TestMemoryRecallImpl:
    def test_recall_surfaces_semantic_warm_tier_hit(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            manager.warm_add(
                "M-semantic",
                MemoryEntry(
                    id="M-semantic",
                    content="opaque title",
                    detail="vector-only match",
                    namespace="project:default",
                    importance=0.9,
                    q_value=0.95,
                    q_observations=5,
                ).model_dump(mode="json"),
                [1.0, 0.0],
            )

            fake_embedder = MagicMock()
            fake_embedder.embed.return_value = [1.0, 0.0]

            with patch("trw_memory.tools.recall.get_local_embedder", return_value=fake_embedder):
                result = memory_recall_impl(
                    "semantic query",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert [memory["id"] for memory in memories] == ["M-semantic"]

    def test_recall_refreshes_hot_recency_for_ttl(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path), hot_ttl_days=7, hot_max_entries=5)
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            stale_time = datetime.now(timezone.utc) - timedelta(days=30)
            entry = MemoryEntry(
                id="M-tool-hot-ttl",
                content="tool ttl refresh lesson",
                namespace="project:default",
                last_accessed_at=stale_time,
            )
            backend.store(entry)
            manager.warm_add("M-tool-hot-ttl", entry.model_dump(mode="json"), None)

            result = memory_recall_impl(
                "tool ttl refresh",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-hot-ttl" for memory in memories)
            hot_entry = manager.hot_get("M-tool-hot-ttl")
            assert hot_entry is not None
            sweep_result = manager.sweep(config=cfg)
            assert sweep_result.demoted == 0

    def test_merge_tier_entries_reranks_by_composite_score(self) -> None:
        cfg = MemoryConfig()
        merged = _merge_tier_entries(
            [
                {
                    "id": "M-local",
                    "content": "deploy lesson",
                    "detail": "stale local result",
                    "importance": 0.1,
                    "q_value": 0.1,
                    "q_observations": 5,
                    "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                    "score": 0.9,
                    "namespace": "project:default",
                }
            ],
            [
                {
                    "id": "M-tier",
                    "content": "deploy lesson",
                    "detail": "fresh tier result",
                    "importance": 0.9,
                    "q_value": 0.95,
                    "q_observations": 5,
                    "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                    "score": 0.2,
                    "namespace": "project:default",
                }
            ],
            ["deploy"],
            cfg,
            None,
        )
        assert [str(entry["id"]) for entry in merged] == ["M-tier", "M-local"]
