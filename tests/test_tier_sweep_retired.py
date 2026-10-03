"""UF-MEM-01: the trw-memory tier sweep and its five sweep-only knobs are gone.

``TierManager.sweep`` had no production caller. Wired into the daemon it would have archived canonical rows
into a cold YAML tier that no daemon-served read consults and then purged them with no trash path, so it was
deleted rather than wired. A config that still sets a removed knob is warned about once and otherwise ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models import _config_sources
from trw_memory.models.config import MemoryConfig

_KNOBS = {
    "hot_ttl_days": "3",
    "cold_threshold_days": "45",
    "retention_days": "180",
    "warm_archive_max_score": "0.3",
    "cold_purge_max_score": "0.2",
}


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config_sources, "_warned_retired_settings", set())
    for name in _KNOBS:
        monkeypatch.delenv(f"MEMORY_{name.upper()}", raising=False)


def _retired(logs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in logs if e["event"] == "retired_setting_ignored"]


def test_the_sweep_and_its_knobs_no_longer_exist() -> None:
    assert not hasattr(TierManager, "sweep")
    for name in _KNOBS:
        assert name not in MemoryConfig.model_fields, name
    assert "hot_max_entries" in MemoryConfig.model_fields, "the hot LRU's size stays"


def test_memory_status_no_longer_echoes_a_retention_that_nothing_enforces() -> None:
    import inspect

    from trw_memory.tools import status

    assert "retention_days" not in inspect.getsource(status)


@pytest.mark.parametrize("name", sorted(_KNOBS))
@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_a_removed_knob_warns_once_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, source: str
) -> None:
    monkeypatch.chdir(tmp_path)
    kwargs: dict[str, object] = {}
    if source == "constructor":
        kwargs[name] = _KNOBS[name]
    else:
        monkeypatch.setenv(f"MEMORY_{name.upper()}", _KNOBS[name])

    with capture_logs() as logs:
        cfg = MemoryConfig(**kwargs)
        MemoryConfig(**kwargs)  # a second construction does not repeat it

    warned = _retired(logs)
    assert len(warned) == 1, warned
    assert str(warned[0]["setting"]).lower().removeprefix("memory_") == name
    assert not hasattr(cfg, name)


def test_the_memory_prefixed_yaml_spellings_are_retired_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """trw-mcp's own YAML-store sweep (their only other reader) was removed (UF-PRD-23), so these are leftovers now."""
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text(
        "memory_hot_ttl_days: 3\nmemory_cold_threshold_days: 45\nmemory_retention_days: 180\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    with capture_logs() as logs:
        MemoryConfig()

    assert sorted(str(e["setting"]).lower() for e in _retired(logs)) == [
        "memory_cold_threshold_days",
        "memory_hot_ttl_days",
        "memory_retention_days",
    ]
