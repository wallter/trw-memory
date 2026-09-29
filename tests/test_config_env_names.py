"""Every MemoryConfig setting is read from a ``MEMORY_``-prefixed env var, never a bare one (ENV-DOUBLE-PREFIX).

pydantic-settings reads a field that has a ``validation_alias`` from each alias name VERBATIM, without
``env_prefix``. Aliases of the form ``AliasChoices("x", "memory_x")`` therefore made the bare ``X`` a live env
var: a generic ``DAEMON_PORT`` moved the memory daemon, and ``EMBEDDING_TRUST_REMOTE_CODE`` could switch on
remote code execution. A field with no alias reads ``env_prefix + name``, so one already named ``memory_x``
needed ``MEMORY_MEMORY_X``. The census below derives every env name each field answers to and refuses both
shapes; the planted cases prove it can fail.
"""

from __future__ import annotations

import pytest
from pydantic import AliasChoices, AliasPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from trw_memory.exceptions import EncryptionAtRestUnsupportedError
from trw_memory.models.config import MemoryConfig


def env_names(settings: type[BaseSettings]) -> dict[str, list[str]]:
    """``field -> the env var names pydantic-settings reads it from`` (upper-cased; case_sensitive is off)."""
    prefix = str(settings.model_config.get("env_prefix", "")).upper()
    by_name = bool(settings.model_config.get("populate_by_name"))
    names: dict[str, list[str]] = {}
    for field, info in settings.model_fields.items():
        alias = info.validation_alias
        if alias is None:
            names[field] = [prefix + field.upper()]
            continue
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        names[field] = [_env_name(choice) for choice in choices]
        if by_name:
            names[field].append(prefix + field.upper())
    return names


def _env_name(choice: object) -> str:
    """The env var one alias choice reads: a str verbatim, an AliasPath its first element. Anything else is
    reported, never skipped: a census that ignores an alias shape would pass a bare name it never looked at."""
    if isinstance(choice, str):
        return choice.upper()
    if isinstance(choice, AliasPath) and choice.path and isinstance(choice.path[0], str):
        return choice.path[0].upper()
    return f"<uncheckable alias {choice!r}>"


def census(settings: type[BaseSettings]) -> list[str]:
    """Every env name without the prefix, and every field readable ONLY under a doubled prefix."""
    prefix = str(settings.model_config.get("env_prefix", "")).upper()
    problems: list[str] = []
    for field, names in sorted(env_names(settings).items()):
        problems += [f"{field}: unprefixed {name}" for name in names if not name.startswith(prefix)]
        if names and all(name.startswith(prefix * 2) for name in names):
            problems.append(f"{field}: only {names}")
    return problems


def test_every_memory_setting_is_read_from_its_memory_prefixed_name() -> None:
    assert census(MemoryConfig) == []


_SECURITY_KNOBS = {  # field -> (bare env value that must be ignored, its default)
    "rbac_enabled": ("true", False),
    "embedding_trust_remote_code": ("true", False),
    "namespace_roles": ('{"shared": "admin"}', {}),
    "encryption_enabled": ("true", False),
}


@pytest.mark.parametrize("field", sorted(_SECURITY_KNOBS))
def test_a_security_knob_ignores_its_bare_env_name(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value, default = _SECURITY_KNOBS[field]
    monkeypatch.setenv(field.upper(), value)

    assert getattr(MemoryConfig(), field) == default


@pytest.mark.parametrize("field", sorted(set(_SECURITY_KNOBS) - {"encryption_enabled"}))
def test_a_security_knob_reads_its_memory_prefixed_name(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value, default = _SECURITY_KNOBS[field]
    monkeypatch.setenv(f"MEMORY_{field.upper()}", value)

    assert getattr(MemoryConfig(), field) != default


def test_encryption_enabled_is_read_from_its_memory_name_and_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")

    with pytest.raises(EncryptionAtRestUnsupportedError):
        MemoryConfig()


@pytest.mark.parametrize(
    ("env", "field", "expected"),
    [
        ("MEMORY_CORRUPT_BACKUP_KEEP", "memory_corrupt_backup_keep", 9),
        ("MEMORY_RECOVERY_INLINE_MAX_BYTES", "memory_recovery_inline_max_bytes", 12345),
        ("MEMORY_DAEMON_PORT", "memory_daemon_port", 7001),
        ("MEMORY_MEMORY_CORRUPT_BACKUP_KEEP", "memory_corrupt_backup_keep", 9),  # by-name spelling still read
    ],
)
def test_memory_fields_read_the_single_prefix_name(
    env: str, field: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, str(expected))

    assert getattr(MemoryConfig(), field) == expected


def test_a_field_name_kwarg_still_works_beside_its_memory_alias() -> None:
    assert MemoryConfig(recall_fusion_mode="rrf").recall_fusion_mode == "rrf"


# --- the census itself, on planted settings ----------------------------------------------------


class _Planted(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_", populate_by_name=True)

    bare_alias: bool = Field(default=False, validation_alias=AliasChoices("bare_alias", "memory_bare_alias"))
    bare_path: bool = Field(default=False, validation_alias=AliasChoices(AliasPath("BARE", 0), "memory_bare_path"))
    memory_doubled: int = 0
    memory_good: int = Field(default=0, validation_alias=AliasChoices("memory_good"))
    plain: int = 0


def test_the_census_reports_a_bare_alias_and_a_doubled_only_name() -> None:
    assert census(_Planted) == [
        "bare_alias: unprefixed BARE_ALIAS",
        "bare_path: unprefixed BARE",
        "memory_doubled: only ['MEMORY_MEMORY_DOUBLED']",
    ]
