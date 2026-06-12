"""PRD-QUAL-110-FR06: trw-memory README carries the disclosure surfaces.

The public trw-memory README must contain a "Telemetry & network behavior"
section, an env-var inventory (TRW_OFFLINE / HF_HUB_OFFLINE / MEMORY_*), a
security-defaults table, and an enterprise hardening recipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _README.read_text(encoding="utf-8")


def test_has_telemetry_network_section(readme_text: str) -> None:
    assert "## Telemetry & network behavior" in readme_text


def test_has_env_var_inventory(readme_text: str) -> None:
    for var in ("TRW_OFFLINE", "HF_HUB_OFFLINE", "MEMORY_*"):
        assert var in readme_text, f"env var {var} missing"


def test_has_security_defaults_and_recipe(readme_text: str) -> None:
    lowered = readme_text.lower()
    assert "security defaults" in lowered
    assert "0600" in readme_text
    assert "hardening recipe" in lowered
    assert "TRW_OFFLINE=1" in readme_text
