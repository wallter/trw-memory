"""The shim hands mem0's extraction the session date. Run: pytest trw-memory/benchmarks/locomo/test_server_dates.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mem0")
sys.path.insert(0, str(Path(__file__).parent))
import mem0.memory.main as mem0_main
from mem0_dates import mem0_extraction_date


def test_extraction_prompt_carries_the_session_date_only_inside_the_context() -> None:
    before = mem0_main.generate_additive_extraction_prompt(new_messages="hi")
    with mem0_extraction_date("2023-05-08"):
        inside = mem0_main.generate_additive_extraction_prompt(new_messages="hi")
    after = mem0_main.generate_additive_extraction_prompt(new_messages="hi")
    assert "## Observation Date\n2023-05-08" in inside and "## Current Date\n2023-05-08" in inside
    assert "2023-05-08" not in before and after == before


def test_restores_the_builder_when_add_raises() -> None:
    original = mem0_main.generate_additive_extraction_prompt
    with pytest.raises(RuntimeError), mem0_extraction_date("2023-01-01"):
        raise RuntimeError("add failed")
    assert mem0_main.generate_additive_extraction_prompt is original


def test_add_resolves_the_builder_through_the_module_attribute() -> None:
    """The swap only works while mem0's add() looks the builder up on its module at call time."""
    import inspect

    src = inspect.getsource(mem0_main.Memory._add_to_vector_store)
    assert "generate_additive_extraction_prompt(" in src
    assert "from mem0.configs.prompts import generate_additive_extraction_prompt" not in src
