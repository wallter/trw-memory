"""PRD-CORE-195 FR01 — HyPE config gate defaults, bounds, env aliases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.config import MemoryConfig


def test_hype_defaults_are_disabled() -> None:
    cfg = MemoryConfig()
    # Backward-compat: HyPE is OFF by default and the knobs match the PRD.
    assert cfg.hype_enabled is False
    assert cfg.hype_questions_per_entry == 3
    assert cfg.hype_min_question_chars == 8


def test_questions_per_entry_bounds() -> None:
    assert MemoryConfig(hype_questions_per_entry=1).hype_questions_per_entry == 1
    assert MemoryConfig(hype_questions_per_entry=10).hype_questions_per_entry == 10
    with pytest.raises(ValidationError):
        MemoryConfig(hype_questions_per_entry=0)
    with pytest.raises(ValidationError):
        MemoryConfig(hype_questions_per_entry=11)


def test_min_question_chars_lower_bound() -> None:
    assert MemoryConfig(hype_min_question_chars=1).hype_min_question_chars == 1
    with pytest.raises(ValidationError):
        MemoryConfig(hype_min_question_chars=0)


def test_env_alias_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_HYPE_ENABLED", "true")
    monkeypatch.setenv("MEMORY_HYPE_QUESTIONS_PER_ENTRY", "5")
    cfg = MemoryConfig()
    assert cfg.hype_enabled is True
    assert cfg.hype_questions_per_entry == 5
