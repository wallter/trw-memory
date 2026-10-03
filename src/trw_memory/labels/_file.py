"""Reading and validating the user's labels.yaml (PRD-SEC-023 FR01).

The file is untrusted input: it is size-bounded, must be a regular file the user alone can write, and is validated against a closed
vocabulary. Every failure raises :class:`InvalidLabelsFile` carrying a reason from a fixed list, never a fragment of the file.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from trw_memory.labels._levels import Level, tag_key

FILE_NAME = "labels.yaml"
MAX_BYTES = 64 * 1024
MAX_RULES = 200
_MAX_TAGS = 64
_MAX_TEXT = 256
_TOP_KEYS = frozenset({"version", "auto_surface_max", "agent_max", "rules"})
_RULE_KEYS = frozenset({"namespace", "tags_any", "level"})
_RULE_LEVELS = {"personal": Level.PERSONAL, "sensitive": Level.SENSITIVE}
_SURFACE_LEVELS = {"team": Level.TEAM, "personal": Level.PERSONAL}


class InvalidLabelsFile(Exception):
    """The file cannot be trusted; ``reason`` is one of a closed set and never quotes the file."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Rule:
    """One raise-only rule: it applies when the namespace glob (if any) AND any of the tags (if any) match."""

    namespace: str | None
    tags: frozenset[str]
    level: Level


@dataclass(frozen=True)
class FileData:
    auto_surface_max: Level
    agent_max: Level
    rules: tuple[Rule, ...]


def read_text(path: Path, st: os.stat_result) -> str:
    """The file's text, only if it is a regular, owner-writable-only file within the size bound.

    *st* (the ``lstat`` the caller keyed its cache on) is a first check; the descriptor actually opened is then judged again with ``fstat`` and
    must be the same file, so a file swapped between the check and the open is never trusted on the strength of the old file's mode.
    """
    _check(st)
    try:
        # O_NONBLOCK: a FIFO swapped in after the check would otherwise block this open until a writer appears; the descriptor is judged next.
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise InvalidLabelsFile("unreadable") from exc
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino):
            raise InvalidLabelsFile("changed_while_reading")
        _check(opened)
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InvalidLabelsFile("oversize")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidLabelsFile("unparseable") from exc


def _check(st: os.stat_result) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise InvalidLabelsFile("not_regular_file")
    if st.st_mode & 0o022:
        raise InvalidLabelsFile("writable_by_others")
    if st.st_size > MAX_BYTES:
        raise InvalidLabelsFile("oversize")


def parse(text: str) -> FileData:
    """Validate *text* and return the policy it describes."""
    try:
        document: Any = YAML(typ="safe").load(text)
    except (YAMLError, ValueError, RecursionError) as exc:
        raise InvalidLabelsFile("unparseable") from exc
    if document is None:
        raise InvalidLabelsFile("empty")
    if not isinstance(document, dict) or not set(document) <= _TOP_KEYS:
        raise InvalidLabelsFile("unknown_key")
    version = document.get("version")
    if version != 1 or isinstance(version, bool):
        raise InvalidLabelsFile("bad_version")
    auto = _surface_level(document, "auto_surface_max", Level.TEAM)
    agent = _surface_level(document, "agent_max", Level.PERSONAL)
    if auto > agent:
        raise InvalidLabelsFile("auto_above_agent")
    raw_rules = document.get("rules", [])
    if not isinstance(raw_rules, list):
        raise InvalidLabelsFile("bad_rule")
    if len(raw_rules) > MAX_RULES:
        raise InvalidLabelsFile("too_many_rules")
    return FileData(auto, agent, tuple(_rule(item) for item in raw_rules))


def _surface_level(document: dict[str, Any], key: str, default: Level) -> Level:
    value = document.get(key, default.name.lower())
    if not isinstance(value, str) or value not in _SURFACE_LEVELS:
        raise InvalidLabelsFile("bad_surface_level")
    return _SURFACE_LEVELS[value]


def _rule(item: Any) -> Rule:
    if not isinstance(item, dict) or not set(item) <= _RULE_KEYS:
        raise InvalidLabelsFile("bad_rule")
    level = item.get("level")
    if not isinstance(level, str) or level not in _RULE_LEVELS:
        raise InvalidLabelsFile("bad_rule_level")
    namespace, tags = item.get("namespace"), item.get("tags_any")
    if namespace is None and tags is None:
        raise InvalidLabelsFile("bad_rule")
    if namespace is not None and (not isinstance(namespace, str) or not namespace or len(namespace) > _MAX_TEXT):
        raise InvalidLabelsFile("bad_rule")
    clean: frozenset[str] = frozenset()
    if tags is not None:
        if not isinstance(tags, list) or not tags or len(tags) > _MAX_TAGS:
            raise InvalidLabelsFile("bad_rule")
        if not all(isinstance(t, str) and t and len(t) <= _MAX_TEXT for t in tags):
            raise InvalidLabelsFile("bad_rule")
        # A tag that is only spacing or invisible characters keys as "" and still matches a blank row tag, as before.
        # Never a reason to load strict: strict drops EVERY rule, which would lower the label of every rule-matched row.
        clean = frozenset(tag_key(t) for t in tags)
    return Rule(namespace, clean, _RULE_LEVELS[level])
