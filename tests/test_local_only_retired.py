"""F5 (2026-09-24, trw-memory 4.0.0 security fix): 'local_only' fails closed.

Before this fix, a leftover ``local_only`` (or ``MEMORY_LOCAL_ONLY``) logged
one ``retired_setting_ignored`` warning and was otherwise ignored. 4.0.0
dropped ``local_only``'s old forced override (it used to set
``sync_enabled: false`` regardless of what a config otherwise asked for), so
a config that ALSO set ``sync_enabled``/``learning_sharing_enabled`` started
syncing after the upgrade -- a privacy flip an operator relying on
``local_only`` would never notice from a warning alone. This suite proves the
replacement behavior: any source that still sets ``local_only`` refuses to
construct ``MemoryConfig`` instead of silently changing behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig

_SPELLINGS = ["local_only", "memory_local_only"]
_VALUES = ["true", "false"]


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("key", _SPELLINGS)
def test_constructor_local_only_refuses(key: str, value: str) -> None:
    with pytest.raises(ConfigError, match="local_only"):
        MemoryConfig(**{key: value == "true"})


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("key", _SPELLINGS)
def test_environment_local_only_refuses(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key.upper(), value)
    with pytest.raises(ConfigError, match="local_only"):
        MemoryConfig()


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("key", _SPELLINGS)
def test_dotenv_local_only_refuses(tmp_path: Path, key: str, value: str) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(f"{key.upper()}={value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="local_only"):
        MemoryConfig(_env_file=env_path)


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("key", _SPELLINGS)
def test_yaml_local_only_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text(f"{key}: {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="local_only"):
        MemoryConfig()


def test_refusal_names_the_key_and_gives_the_remedy() -> None:
    with pytest.raises(ConfigError) as exc_info:
        MemoryConfig(local_only=True)
    message = str(exc_info.value)
    assert "local_only" in message
    assert "4.0.0" in message
    assert "sync_enabled: false" in message


def test_absence_of_local_only_still_constructs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative case: no local_only anywhere -- construction is unaffected."""
    monkeypatch.chdir(tmp_path)
    cfg = MemoryConfig()
    assert cfg.storage_backend == "sqlite"
    assert "local_only" not in MemoryConfig.model_fields


def test_local_only_false_paired_with_sync_enabled_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a 'neutral-looking' local_only=false must refuse -- unconditional, not value-gated."""
    monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    with pytest.raises(ConfigError, match="local_only"):
        MemoryConfig()
