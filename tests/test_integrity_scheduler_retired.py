"""UF-MEM-06: the periodic integrity scheduler is gone, and so is the knob that never reached it.

``memory_integrity_check_interval_minutes`` was never passed to the backend by the daemon-era factory, and the
``integrity_warning`` flag the scheduler set had no reader anywhere (no doctor row, no status field), so wiring it
would have produced a log line nobody reads. A config that still sets the knob is warned about once and ignored.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.models import _config_sources
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend

_KNOB = "memory_integrity_check_interval_minutes"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config_sources, "_warned_retired_settings", set())
    monkeypatch.delenv(_KNOB.upper(), raising=False)


def test_the_scheduler_module_and_its_backend_parameter_are_gone() -> None:
    assert importlib.util.find_spec("trw_memory.storage._integrity_scheduler") is None
    assert "integrity_check_interval_minutes" not in inspect.signature(SQLiteBackend.__init__).parameters
    assert _KNOB not in MemoryConfig.model_fields


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_a_config_that_still_sets_the_knob_warns_once_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    monkeypatch.chdir(tmp_path)
    kwargs: dict[str, object] = {}
    if source == "constructor":
        kwargs[_KNOB] = 5
    else:
        monkeypatch.setenv(_KNOB.upper(), "5")

    with capture_logs() as logs:
        cfg = MemoryConfig(**kwargs)
        MemoryConfig(**kwargs)

    warned = [e for e in logs if e["event"] == "retired_setting_ignored"]
    assert len(warned) == 1, warned
    assert str(warned[0]["setting"]).lower() == _KNOB
    assert not hasattr(cfg, _KNOB)
