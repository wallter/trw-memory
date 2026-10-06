"""The machine-level jev store (``~/.trw/jev.env``) and the one key/endpoint/model resolver."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from trw_memory.decisions import _machine_store
from trw_memory.decisions._env import judge_from_env
from trw_memory.decisions._machine_store import (
    MACHINE_STORE_LABEL,
    read_machine_store,
    resolve_jev_settings,
    write_machine_store,
)

_KEY = "sk-or-machine-test-0000000000001234"
_PROJECT_KEY = "sk-or-project-test-000000000005678"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Undo the conftest guard that hides the operator's real store: this module tests the store itself.
    monkeypatch.setattr(_machine_store, "machine_store_path", lambda: home / ".trw" / "jev.env")
    return home


def _store(home: Path, body: str, mode: int = 0o600) -> Path:
    (home / ".trw").mkdir(exist_ok=True)
    path = home / ".trw" / "jev.env"
    path.write_text(body, encoding="utf-8")
    path.chmod(mode)
    return path


def test_store_is_the_last_fallback_for_the_key(home: Path, tmp_path: Path) -> None:
    _store(home, f"OPENROUTER_API_KEY={_KEY}\n")
    dotenv = tmp_path / ".env"

    assert resolve_jev_settings({}, dotenv).key_source == MACHINE_STORE_LABEL
    dotenv.write_text(f"OPENROUTER_API_KEY={_PROJECT_KEY}\n", encoding="utf-8")
    from_project = resolve_jev_settings({}, dotenv)
    assert (from_project.api_key, from_project.key_source) == (_PROJECT_KEY, "project .env")
    from_env = resolve_jev_settings({"OPENROUTER_API_KEY": "sk-env"}, dotenv)
    assert (from_env.api_key, from_env.key_source) == ("sk-env", "environment")


def test_endpoint_and_model_come_from_env_then_store_never_the_project_dotenv(home: Path, tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("TRW_JEV_BASE_URL=https://evil.example/\nTRW_JEV_MODEL=evil\n", encoding="utf-8")
    assert resolve_jev_settings({}, dotenv).base_url is None  # the repo-controlled file cannot point the key

    _store(home, "TRW_JEV_BASE_URL=https://openrouter.ai/api/alpha/decisions\nTRW_JEV_MODEL=~typesafe/jev-latest\n")
    settings = resolve_jev_settings({"TRW_JEV_MODEL": "env-model"}, dotenv)
    assert (settings.base_url_source, settings.model, settings.model_source) == (
        MACHINE_STORE_LABEL,
        "env-model",
        "environment",
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660])
def test_a_store_readable_by_others_is_refused_and_named(home: Path, mode: int) -> None:
    _store(home, f"OPENROUTER_API_KEY={_KEY}\n", mode=mode)

    read = read_machine_store()
    settings = resolve_jev_settings({}, None)

    assert read.values == {} and read.present and "looser than 0600" in read.problem
    assert settings.api_key is None and "chmod 600" in settings.store_problem


def test_a_symlinked_store_is_refused(home: Path, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.env"
    real.write_text(f"OPENROUTER_API_KEY={_KEY}\n", encoding="utf-8")
    real.chmod(0o600)
    (home / ".trw").mkdir()
    (home / ".trw" / "jev.env").symlink_to(real)

    read = read_machine_store()

    assert read.values == {} and read.problem == "is a symlink"


def test_absent_store_is_not_a_problem(home: Path) -> None:
    read = read_machine_store()
    assert (dict(read.values), read.present, read.problem) == ({}, False, "")


def test_writer_publishes_0600_merges_and_rejects_foreign_keys(home: Path) -> None:
    path = write_machine_store({"OPENROUTER_API_KEY": _KEY})
    write_machine_store({"TRW_JEV_MODEL": "~typesafe/jev-latest"})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert dict(read_machine_store().values) == {"OPENROUTER_API_KEY": _KEY, "TRW_JEV_MODEL": "~typesafe/jev-latest"}
    with pytest.raises(ValueError, match="not a machine-store key"):
        write_machine_store({"TRW_JEV_ENABLED": "true"})
    with pytest.raises(ValueError, match="newline"):
        write_machine_store({"OPENROUTER_API_KEY": "a\nTRW_JEV_BASE_URL=https://evil.example"})


def test_settings_repr_never_carries_the_key(home: Path) -> None:
    _store(home, f"OPENROUTER_API_KEY={_KEY}\n")
    assert _KEY not in repr(resolve_jev_settings({}, None))
    assert _KEY not in repr(read_machine_store())


def test_machine_switch_plus_machine_key_builds_the_live_judge(home: Path) -> None:
    (home / ".trw").mkdir()
    (home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    _store(home, f"OPENROUTER_API_KEY={_KEY}\n")

    assert type(judge_from_env({})).__name__ == "JevHttpJudge"


def test_store_endpoint_off_the_allowlist_abstains(home: Path) -> None:
    (home / ".trw").mkdir()
    (home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    _store(home, f"OPENROUTER_API_KEY={_KEY}\nTRW_JEV_BASE_URL=https://evil.example/decisions\n")

    judge = judge_from_env({})

    assert type(judge).__name__ == "NullJudge"


def test_enabled_without_any_key_names_the_machine_store(home: Path) -> None:
    judge = judge_from_env({"TRW_JEV_ENABLED": "1"})
    assert type(judge).__name__ == "NullJudge" and MACHINE_STORE_LABEL in judge.detail
