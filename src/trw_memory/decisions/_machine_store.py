"""The machine-level jev store and the ONE resolver for the jev key, endpoint and model.

**Why.** An operator with TRW installed in many projects on one computer should configure the
OpenRouter key once, not copy it into every project's ``.env``. The store is
``~/.trw/jev.env``: the same ``KEY=value`` lines a project ``.env`` holds, beside the
``~/.trw/config.yaml`` machine switch (``assess_enabled``) that already turns the backend on for
every project. It follows ``Path.home()`` exactly as that switch does (not ``TRW_USER_DIR``, which
gives each canary env its own memory dir but must not hide the operator's key from it).

**Precedence (first non-blank wins), resolved by :func:`resolve_jev_settings` only:**

* ``OPENROUTER_API_KEY``: process env > project ``.env`` > machine store.
* ``TRW_JEV_BASE_URL`` / ``TRW_JEV_MODEL``: process env > machine store. Never the project
  ``.env``: that file is repo-controlled, so a cloned repo could otherwise redirect where the key is
  sent (release-verify R1). The machine store is the operator's own file, outside every repo, so it
  may carry them; the base-URL host allowlist in :mod:`trw_memory.decisions._env` still applies.

**Owner-only or refused.** Reading and writing go through :mod:`trw_memory.machine_secrets`, the
helper every machine secret shares: a regular file (not a symlink) owned by the current user with no
group or other bits, else refused, logged with the fix, and reported by
:attr:`JevSettings.store_problem` for ``trw-mcp doctor`` and ``trw-mcp assess status``; writes are
atomic and 0600 from creation.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trw_memory.decisions._dotenv import _parse_dotenv_lines, parse_dotenv_subset
from trw_memory.machine_secrets import machine_secret_path, read_private_file, write_private_file

__all__ = [
    "MACHINE_STORE_KEYS",
    "MACHINE_STORE_LABEL",
    "JevSettings",
    "StoreRead",
    "machine_store_path",
    "read_machine_store",
    "resolve_jev_settings",
    "write_machine_store",
]

#: The only keys the store is read for or written with. Enablement is not among them: the machine
#: switch is ``assess_enabled`` in ``~/.trw/config.yaml`` (:mod:`trw_memory.decisions._enablement`).
MACHINE_STORE_KEYS = frozenset({"OPENROUTER_API_KEY", "TRW_JEV_BASE_URL", "TRW_JEV_MODEL"})

#: How the store is named in sources, messages and the doctor row (never with the home path expanded).
MACHINE_STORE_LABEL = "~/.trw/jev.env"

_KEY = "OPENROUTER_API_KEY"
_FILE_NAME = "jev.env"


def machine_store_path() -> Path:
    """``~/.trw/jev.env``, resolved at call time so a test (or a changed ``HOME``) is honoured."""
    return machine_secret_path(_FILE_NAME)


@dataclass(frozen=True)
class StoreRead:
    """What the store yielded: its allowlisted values, or why it was refused (``problem``)."""

    values: Mapping[str, str] = field(repr=False)
    present: bool
    problem: str = ""


def _parse(text: str) -> dict[str, str]:
    return _parse_dotenv_lines(text, allowed_keys=MACHINE_STORE_KEYS)


def read_machine_store(path: Path | None = None) -> StoreRead:
    """Read the store's allowlisted keys through the shared owner-only reader (:mod:`trw_memory.machine_secrets`)."""
    target = machine_store_path() if path is None else path
    read = read_private_file(target, _parse, shown=MACHINE_STORE_LABEL)
    return StoreRead(read.value or {}, present=read.present, problem=read.problem)


@dataclass(frozen=True)
class JevSettings:
    """The resolved key, endpoint and model, each with the layer that supplied it (``""`` = none).

    ``api_key`` is excluded from ``repr`` so a logged or printed settings object never carries it.
    """

    api_key: str | None = field(repr=False)
    key_source: str
    base_url: str | None
    base_url_source: str
    model: str | None
    model_source: str
    store_problem: str = ""


def resolve_jev_settings(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
    *,
    store_path: Path | None = None,
) -> JevSettings:
    """THE resolver every jev reader uses — see the module docstring for the precedence.

    The machine store is read at most once, and only when the process env (and, for the key, the
    project ``.env``) left something unset. Never raises.
    """
    process_env = os.environ if env is None else env
    store: StoreRead | None = None

    def from_store(name: str) -> str | None:
        nonlocal store
        if store is None:
            store = read_machine_store(store_path)
        return store.values.get(name) or None

    key, key_source = process_env.get(_KEY) or None, "environment"
    if key is None and dotenv_path is not None:
        key, key_source = (
            parse_dotenv_subset(dotenv_path, allowed_keys=frozenset({_KEY})).get(_KEY) or None,
            "project .env",
        )
    if key is None:
        key, key_source = from_store(_KEY), MACHINE_STORE_LABEL

    picked: dict[str, tuple[str | None, str]] = {}
    for name in ("TRW_JEV_BASE_URL", "TRW_JEV_MODEL"):
        value = process_env.get(name) or None
        picked[name] = (value, "environment") if value else (from_store(name), MACHINE_STORE_LABEL)

    def source(value: str | None, layer: str) -> str:
        return layer if value else ""

    base_url, base_layer = picked["TRW_JEV_BASE_URL"]
    model, model_layer = picked["TRW_JEV_MODEL"]
    final_store: StoreRead | None = store  # mypy: narrowed copy of the closure variable
    return JevSettings(
        api_key=key,
        key_source=source(key, key_source),
        base_url=base_url,
        base_url_source=source(base_url, base_layer),
        model=model,
        model_source=source(model, model_layer),
        store_problem=final_store.problem if final_store is not None else "",
    )


def write_machine_store(values: Mapping[str, str], *, home: Path | None = None) -> Path:
    """Merge ``values`` into the store and publish it atomically at exactly 0600; return its path.

    Only :data:`MACHINE_STORE_KEYS` are accepted (anything else raises ``ValueError``, as does a
    value carrying a newline). Keys already in the store and not in ``values`` are kept. A symlinked
    ``~/.trw`` or store is refused by :func:`trw_memory.safe_fs.write_beneath`.
    """
    unknown = set(values) - MACHINE_STORE_KEYS
    if unknown:
        raise ValueError(f"not a machine-store key: {sorted(unknown)}")
    if any("\n" in v or "\r" in v for v in values.values()):
        raise ValueError("a machine-store value may not contain a newline")
    target = (Path.home() if home is None else home) / ".trw" / _FILE_NAME
    existing = read_machine_store(target)
    merged = {**existing.values, **{k: v.strip() for k, v in values.items() if v.strip()}}
    body = "# TRW machine-level jev settings (trw-mcp assess configure). Owner-only: keep it 0600.\n"
    body += "".join(f"{name}={merged[name]}\n" for name in sorted(merged))
    write_private_file(target, body)
    return target
