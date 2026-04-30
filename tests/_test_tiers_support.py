from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus


@pytest.fixture
def cfg() -> MemoryConfig:
    return _cfg()


def _cfg() -> MemoryConfig:
    return MemoryConfig(
        hot_max_entries=3,
        hot_ttl_days=7,
        cold_threshold_days=90,
        retention_days=365,
        decay_half_life_days=14.0,
        score_relevance_weight=0.4,
        score_recency_weight=0.3,
        score_importance_weight=0.3,
    )


@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def mgr(tmp_path: Path) -> TierManager:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return TierManager(base_dir=tmp_path, config=_cfg())


def _make_entry(
    entry_id: str = "test-id",
    importance: float = 0.5,
    status: str = "active",
    days_old: int = 0,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    last_accessed_at = now - timedelta(days=days_old)
    return MemoryEntry(
        id=entry_id,
        content=f"content for {entry_id}",
        detail="some detail",
        tags=["tag1"],
        importance=importance,
        status=MemoryStatus(status),
        last_accessed_at=last_accessed_at,
    )
