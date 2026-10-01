"""UF-MCP-07: the forced tier-distribution function and its two caps are gone from trw-memory."""

from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.lifecycle import scoring
from trw_memory.models import _config_sources
from trw_memory.models.config import MemoryConfig


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config_sources, "_warned_retired_settings", set())
    for name in ("IMPACT_TIER_CRITICAL_CAP", "IMPACT_TIER_HIGH_CAP", "MEMORY_IMPACT_TIER_CRITICAL_CAP"):
        monkeypatch.delenv(name, raising=False)


def test_the_function_and_the_caps_are_gone() -> None:
    assert not hasattr(scoring, "enforce_tier_distribution")
    assert "impact_tier_critical_cap" not in MemoryConfig.model_fields
    assert "impact_tier_high_cap" not in MemoryConfig.model_fields


@pytest.mark.parametrize("name", ["impact_tier_critical_cap", "impact_tier_high_cap"])
def test_a_config_that_still_sets_a_cap_warns_once_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    monkeypatch.chdir(tmp_path)

    with capture_logs() as logs:
        cfg = MemoryConfig(**{name: 0.5})
        MemoryConfig(**{name: 0.5})

    warned = [e for e in logs if e["event"] == "retired_setting_ignored"]
    assert len(warned) == 1, warned
    assert str(warned[0]["setting"]).lower() == name
    assert not hasattr(cfg, name)
