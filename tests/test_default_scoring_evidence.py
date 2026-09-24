"""CORE268: historical rewards are records, not default ranking authority."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.lifecycle import utility_based_prune_candidates
from trw_memory.lifecycle.scoring import (
    entry_utility,
)
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig


@pytest.mark.parametrize("importance_key", ["importance", "impact"])
@pytest.mark.parametrize("observations", [0, 1, 3, 100])
def test_default_utility_ignores_historical_rewards(importance_key: str, observations: int) -> None:
    today = datetime.now(timezone.utc).date()
    entry: dict[str, object] = {
        "id": "fixture",
        importance_key: 0.6,
        "last_accessed_at": today.isoformat(),
        "q_value": 0.0,
        "q_observations": observations,
        "outcome_history": ["2026-01-01:-1:tests_failed"],
    }
    other = {**entry, "q_value": 1.0, "q_observations": 1000, "outcome_history": []}
    before = deepcopy(entry)
    assert entry_utility(entry, today=today) == entry_utility(other, today=today)
    assert entry == before


def test_tier_importance_uses_declared_importance_not_reward_history() -> None:
    config = MemoryConfig()
    entry: dict[str, object] = {"content": "fixture", "importance": 0.6, "q_value": 0.0, "q_observations": 100}
    other = {**entry, "q_value": 1.0, "outcome_history": ["2026-01-01:1:delivered"]}
    assert compute_importance_score(entry, ["fixture"], config=config) == compute_importance_score(
        other, ["fixture"], config=config
    )


@pytest.mark.parametrize("protection", ["normal", "protected", "permanent"])
def test_prune_nominations_ignore_q_without_changing_protection(protection: str) -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    entries: list[dict[str, object]] = [
        {
            "id": "low",
            "importance": 0.1,
            "created_at": old,
            "last_accessed_at": old,
            "protection_tier": protection,
            "status": "active",
            "q_value": 0.0,
            "q_observations": 20,
        }
    ]
    other = [{**entry, "q_value": 1.0, "outcome_history": ["2026-01-01:1:delivered"]} for entry in entries]
    actual = utility_based_prune_candidates(entries)
    assert actual == utility_based_prune_candidates(other)
    if protection in {"protected", "permanent"}:
        assert actual == []
    else:
        assert [candidate["id"] for candidate in actual] == ["low"]
