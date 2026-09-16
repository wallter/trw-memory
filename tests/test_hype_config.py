"""PRD-CORE-272 FR01: explicit retired inputs never disappear silently."""

from unittest.mock import Mock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig

DEFAULTS = {"hype_enabled": False, "hype_questions_per_entry": 3, "hype_min_question_chars": 8}
ALIASES = [(prefix + key, value) for key, value in DEFAULTS.items() for prefix in ("", "memory_")]


@pytest.mark.parametrize("key,value", ALIASES)
@pytest.mark.parametrize("source", ["constructor", "environment", "yaml"])
def test_neutral_legacy_input_warns_and_is_not_emitted(tmp_path, monkeypatch, key, value, source):
    monkeypatch.chdir(tmp_path)
    kwargs = {}
    if source == "constructor":
        kwargs[key] = value
    elif source == "environment":
        monkeypatch.setenv(key.upper(), str(value).lower())
    else:
        (tmp_path / ".trw").mkdir()
        (tmp_path / ".trw/config.yaml").write_text(f"{key}: {str(value).lower()}\n")
    with pytest.warns(UserWarning, match="HyPE is retired"):
        cfg = MemoryConfig(**kwargs)
    assert not any("hype" in k for k in cfg.model_dump())
    assert not any("hype" in k for k in MemoryConfig.model_fields)
    assert "hype" not in str(MemoryConfig.model_json_schema()).lower()


@pytest.mark.parametrize("key,_default", ALIASES)
@pytest.mark.parametrize("value", [True, None, "false", 0, 1, [], {}])
def test_nondefault_or_invalid_constructor_rejected(key, _default, value):
    with pytest.raises(ConfigError, match="retired"):
        MemoryConfig(**{key: value})


@pytest.mark.parametrize("source", ["environment", "yaml"])
def test_lower_precedence_activation_cannot_be_hidden(tmp_path, monkeypatch, source):
    monkeypatch.chdir(tmp_path)
    if source == "environment":
        monkeypatch.setenv("MeMoRy_HyPe_EnAbLeD", "true")
    else:
        (tmp_path / ".trw").mkdir()
        (tmp_path / ".trw/config.yaml").write_text("memory_hype_enabled: true\n")
    with pytest.warns(UserWarning), pytest.raises(ConfigError, match="retired"):
        MemoryConfig(hype_enabled=False)


@pytest.mark.parametrize("value", ["true", "1", "no", "off", "", "False "])
def test_environment_activation_and_noncanonical_spellings_reject(monkeypatch, value):
    monkeypatch.setenv("MEMORY_HYPE_ENABLED", value)
    with pytest.raises(ConfigError, match="retired"):
        MemoryConfig()


def test_conflicting_alias_not_hidden():
    with pytest.warns(UserWarning), pytest.raises(ConfigError, match="retired"):
        MemoryConfig(hype_enabled=False, memory_hype_enabled=True)


def test_generator_activation_rejected_before_lifecycle(monkeypatch):
    from trw_memory import _client_lifecycle

    init = Mock(side_effect=AssertionError("lifecycle must not run"))
    generator = Mock()
    monkeypatch.setattr(_client_lifecycle, "init_client", init)
    with pytest.raises(TypeError, match="retired"):
        MemoryClient("default", question_generator=generator)
    init.assert_not_called()
    generator.generate.assert_not_called()


def test_config_activation_precedes_backend_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_HYPE_ENABLED", "true")
    path = tmp_path / "never-created" / "memory.db"
    with pytest.raises(ConfigError, match="retired"):
        MemoryClient("default", mode="local", db_path=path)
    assert not path.parent.exists()


def test_unrelated_nonstring_yaml_keys_still_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw/config.yaml").write_text("1: unrelated\n")
    assert MemoryConfig().storage_backend


@pytest.mark.parametrize("key,default", ALIASES)
@pytest.mark.parametrize("source", ["environment", "yaml"])
def test_each_source_rejects_nondefault_settings(tmp_path, monkeypatch, key, default, source):
    monkeypatch.chdir(tmp_path)
    if source == "environment":
        monkeypatch.setenv(key.upper(), "true" if default is False else "9")
    else:
        (tmp_path / ".trw").mkdir()
        (tmp_path / ".trw/config.yaml").write_text(f"{key}: {'true' if default is False else '9'}\n")
    with pytest.raises(ConfigError, match="retired"):
        MemoryConfig()


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_ignore_empty_does_not_hide_retired_input(tmp_path, monkeypatch, source):
    kwargs = {"_env_ignore_empty": True}
    if source == "environment":
        monkeypatch.setenv("MEMORY_HYPE_ENABLED", "")
    else:
        path = tmp_path / ".env"
        path.write_text("MEMORY_HYPE_ENABLED=\n")
        kwargs["_env_file"] = path
    with pytest.raises(ConfigError, match="retired"):
        MemoryConfig(**kwargs)


def test_dotenv_precedence_cannot_hide_activation(tmp_path):
    enabled, neutral = tmp_path / "enabled.env", tmp_path / "neutral.env"
    enabled.write_text("MEMORY_HYPE_ENABLED=true\n")
    neutral.write_text("MEMORY_HYPE_ENABLED=false\n")
    with pytest.raises(ConfigError, match="retired"):
        MemoryConfig(_env_file=(enabled, neutral))
