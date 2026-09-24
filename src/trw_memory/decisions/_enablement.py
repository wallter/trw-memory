"""Is the trw-jev backend enabled, and which layer decided it.

The ONE resolver both packages read (PRD-CORE-288 follow-up, 2026-09-23 operator decision):
``trw_mcp.tools._assess_enablement.backend_enablement`` calls straight into this module rather
than re-implementing the cascade, and ``judge_from_env``/``toolkit_from_env`` in this package use
it too — so the MCP tool's answer and the in-process/daemon judge's answer can never disagree.
Owned here (not in trw-mcp) because trw-memory must not import trw-mcp, and the judge that
actually makes the network call already lives in this package.

**Precedence (first explicit wins, evaluated lazily in this order — a higher layer's explicit
value short-circuits before any lower layer's file is even read):**

1. ``TRW_JEV_ENABLED`` in the process env — the operator's shell, or what the client forwarded.
2. Project scope: ``assess_enabled`` in the project's ``.trw/config.yaml``, or ``TRW_JEV_ENABLED``
   in the project ``.env`` (checked in that order within the tier).
3. User scope: ``assess_enabled`` in ``~/.trw/config.yaml`` (the operator's machine switch,
   outside every repo).
4. Default: off.

**2026-09-23 operator decision relaxed the prior rule.** A project used to be able to switch the
backend OFF but never ON; a project may now enable it too (the key stays per-repo,
``OPENROUTER_API_KEY`` in the env or the project ``.env``, and the base-URL host allowlist in
:mod:`trw_memory.decisions._env` is unchanged — this resolver only ever decides on/off, never
where the key is sent). An explicit value at a higher-precedence layer wins outright, whether
true or false: an explicit env ``false`` beats a project ``true``, and an explicit project
``false`` beats a user-scope ``true``, matching ordinary cascading config precedence.

**Structural fail-closed rule (2026-09-23 review, round 3).** Both the YAML layers (project and
user ``.trw/config.yaml``) and the project ``.env`` layer are read through the ONE shared
:func:`_read_layer_text` — the single place in this module allowed to decide "genuinely absent"
vs. "exists (or existence unprovable) but unreadable" — so the same fail-closed rule cannot drift
between the two file kinds. :func:`_read_layer_text` returns :data:`_LayerRead.ABSENT` ONLY when
the OS *confirms* the path does not exist at all: an ``lstat`` raising ``FileNotFoundError``, or
any other ``OSError`` whose ``errno`` is ``ENOENT``/``ENOTDIR`` (a path component isn't a
directory, so the path cannot exist either). Every other outcome is
:data:`_LayerRead.REFUSED` and maps to explicit ``False`` at that layer, never to ABSENT: a
``PermissionError``/``EACCES`` (or any other ``OSError``) from the existence check itself, or the
hardened reader (:func:`_read_small_regular_file`) refusing to read what IS there — a symlink, a
FIFO/device/directory, an oversized file, non-UTF-8 content. "Existence could not be disproven"
is deliberately treated the same as "exists and is broken": only a confirmed-absent file may let a
lower, more permissive layer stand in for it.

On top of a confirmed-readable layer, :func:`_explicit` is the one further tri-state classifier:
``None`` (ABSENT — the layer has no opinion, cascade to the next one), or an explicit
``True``/``False``. Anything that is not a clean bool and not the ABSENT sentinel maps to
``False`` — a YAML ``null``, an int like ``1``, a list, a blank string ``""`` — because ABSENT is
the one state a lower layer is allowed to shine through, and a present-but-broken value must never
mimic it. The env/dotenv sources are the one legitimate exception: a blank value there counts as
ABSENT, because a client that forwards ``${env:X}`` for an unset ``X`` passes an empty string,
unlike a present, non-empty, unrecognized YAML value someone deliberately wrote.
"""

from __future__ import annotations

import enum
import errno
import os
from collections.abc import Mapping
from pathlib import Path

from trw_memory.decisions._dotenv import _parse_dotenv_lines, _read_small_regular_file

__all__ = ["resolve_backend_enablement"]

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Sentinel distinguishing "key/layer genuinely absent" (unset, cascade) from "present but not a
#: value we understand" (fail closed to explicit False). Every call site must pass this as the
#: ``.get(key, _MISSING)`` default — never a bare ``.get(key)``, whose implicit ``None`` default
#: would otherwise be indistinguishable from a present YAML ``null``.
_MISSING = object()

#: The confirmed-absent / errno set from an ``lstat`` that means "this path cannot exist" rather
#: than "we could not determine whether it exists" -- see :func:`_read_layer_text`.
_CONFIRMED_ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR})


class _LayerRead(enum.Enum):
    """The non-text outcomes of :func:`_read_layer_text` — see the module docstring."""

    ABSENT = enum.auto()
    REFUSED = enum.auto()


def _read_layer_text(path: Path) -> str | _LayerRead:
    """``path``'s text, or the ABSENT/REFUSED classification — see the module docstring.

    Every layer reader in this module (YAML and dotenv alike) goes through this ONE function so
    the fail-closed rule cannot drift between file kinds: only a confirmed ``FileNotFoundError``/
    ``ENOENT``/``ENOTDIR`` is ABSENT; every other outcome, including an existence check that
    itself failed for an unrelated reason (``PermissionError``, ...), is REFUSED.
    """
    try:
        os.lstat(path)
    except FileNotFoundError:
        return _LayerRead.ABSENT
    except OSError as exc:
        if exc.errno in _CONFIRMED_ABSENT_ERRNOS:
            return _LayerRead.ABSENT
        return _LayerRead.REFUSED
    text = _read_small_regular_file(path)
    if text is None:
        return _LayerRead.REFUSED  # lstat confirmed something is there; the hardened reader refused it
    return text


def _explicit(value: object, *, blank_is_absent: bool) -> bool | None:
    """The ONE tri-state classifier every layer's VALUE goes through: ``None`` (ABSENT) or a bool.

    ``blank_is_absent`` is the one place layers legitimately differ: env/dotenv values pass
    ``True`` (an unresolved ``${env:X}`` template is not a deliberate configuration choice), YAML
    values pass ``False`` (a present, empty ``assess_enabled: ""`` IS a deliberate write and means
    off, not "say nothing"). Every other input path is identical and strict: only ``_MISSING`` and
    a real ``bool`` are handled specially; a recognized truthy/falsy string decides; anything else
    present — ``None`` (YAML null), an int, a float, a list, a dict, an unrecognized string — is
    explicit ``False``.
    """
    if value is _MISSING:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None if blank_is_absent else False
        return stripped.lower() in _TRUTHY
    return False  # None (YAML null) or any other type we don't recognize as a bool


def _read_yaml_bool(path: Path, key: str) -> bool | None:
    """The value of ``key`` in the YAML mapping at ``path``, via :func:`_read_layer_text`/:func:`_explicit`."""
    result = _read_layer_text(path)
    if isinstance(result, _LayerRead):
        return None if result is _LayerRead.ABSENT else False

    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError

        data = YAML(typ="safe").load(result)
    # trw-fail-silent-allow: a project/user config file that exists but fails to parse must fail closed (explicit off) rather than raise into a tool call or an MCP request; the malformed layer is treated the same as an explicit "off" everywhere else in this module
    except (YAMLError, ValueError):
        return False  # the file exists but we can't trust it: fail closed, not unset
    if not isinstance(data, dict):
        return False  # exists and parsed, but not a mapping: fail closed, not unset
    return _explicit(data.get(key, _MISSING), blank_is_absent=False)


def _dotenv_names_key_without_a_clean_assignment(text: str, key: str) -> bool:
    """True if some non-comment, non-blank line's leading token IS ``key``, but the line is not a
    clean ``key=value`` assignment ``_parse_dotenv_lines`` would extract.

    Deliberately does NOT require the absence of an ``=`` anywhere in the line (round-5 review
    fix): ``TRW_JEV_ENABLED true=1`` names the key in its leading token but the ``=`` belongs to a
    garbled tail, not a clean assignment, so an earlier version of this check (which bailed out on
    any ``=`` in the line) missed it. The leading token is isolated by splitting on the FIRST
    ``=`` and then on whitespace, so a key that merely shares ``key`` as a PREFIX --
    ``TRW_JEV_ENABLED_OTHER=true`` -- is never matched: its leading token is the whole distinct
    name, not ``key``. Mirrors ``_parse_dotenv_lines``'s own ``export`` handling and comment/blank
    skipping so "named" here means exactly what that function would have looked at.
    """
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        head = line.split("=", 1)[0].split()
        token = head[0] if head else ""
        if token == key:
            return True
    return False


def _read_dotenv_bool(path: Path, key: str) -> bool | None:
    """``key`` from the dotenv at ``path``, via the SAME :func:`_read_layer_text` classification
    the YAML layers use — so a project ``.env`` refused by the hardened reader fails closed
    exactly like a refused ``.trw/config.yaml``, rather than silently cascading via
    :func:`~trw_memory.decisions._dotenv.parse_dotenv_subset`'s "any refusal is empty" contract
    (which is the right behavior for that function's OTHER caller, the credential lookup, but
    wrong for an enablement decision).

    **Round-4/5 review fixes.** A line that clearly NAMES ``key`` — with a clean ``key=value``
    assignment or not — is decisive and can never be treated as "this file never mentioned it":
    ``key=value`` with a value that isn't a clean bool (including a BLANK value — unlike the
    process-env layer, a human-written dotenv line is a deliberate write, not an unresolved
    ``${env:X}`` template) is explicit ``False``, and so is any line naming ``key`` without a
    clean assignment — no ``=`` at all (``KEY true``, a bare ``KEY``), or an ``=`` that belongs to
    a garbled tail rather than a clean value (``KEY true=1``). Only a file that never mentions
    ``key`` anywhere is genuinely absent and cascades.
    """
    result = _read_layer_text(path)
    if isinstance(result, _LayerRead):
        return None if result is _LayerRead.ABSENT else False

    values = _parse_dotenv_lines(result, allowed_keys=frozenset({key}))
    if key in values:
        return _explicit(values[key], blank_is_absent=False)
    if _dotenv_names_key_without_a_clean_assignment(result, key):
        return False
    return None


def resolve_backend_enablement(project_root: Path | None, env: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """``(enabled, source)``. ``source`` names the deciding layer, or ``""`` when none did.

    ``project_root`` may be ``None`` for a caller with no project context (e.g. the bare CLI or a
    daemon started outside a project) — project-scope layers are then skipped and only the
    process env and user scope are consulted. Layers are evaluated lazily in precedence order: the
    first explicit answer wins and no lower-precedence file is read once a higher layer decides
    (an explicit process env setting reads no file at all).
    """
    process_env = os.environ if env is None else env

    env_value = _explicit(process_env.get("TRW_JEV_ENABLED", _MISSING), blank_is_absent=True)
    if env_value is not None:
        return env_value, "TRW_JEV_ENABLED"

    if project_root is not None:
        project_yaml = _read_yaml_bool(project_root / ".trw" / "config.yaml", "assess_enabled")
        if project_yaml is not None:
            return project_yaml, "project .trw/config.yaml"

        project_dotenv = _read_dotenv_bool(project_root / ".env", "TRW_JEV_ENABLED")
        if project_dotenv is not None:
            return project_dotenv, "project .env"

    user_yaml = _read_yaml_bool(Path.home() / ".trw" / "config.yaml", "assess_enabled")
    if user_yaml is not None:
        return user_yaml, "~/.trw/config.yaml"

    return False, ""
