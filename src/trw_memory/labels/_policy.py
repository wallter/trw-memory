"""The label policy: load once per file identity, label a row, admit rows to a surface or sink (PRD-SEC-023 FR01-FR03).

Rules compile once into a tag -> level table and a per-namespace table (cached, bounded), so labelling a row costs a few dict lookups
whatever the rule count; the p95 budget for 500 rows and 50 rules is 2 ms.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from trw_memory.labels import _file
from trw_memory.labels._levels import STAMP_VALUES, Level, Sink, Surface, tag_key

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

Source = Literal["default", "file", "strict"]
_NS_CACHE_MAX = 4096
_USER_PREFIX = "user:"
_LOCAL_USER = "user:local"
_STAMP_KEY = "trw_label"


def _higher_stamp(existing: str | None, mark: Level) -> str:
    """The stamp value for *mark*, unless *existing* is higher (an unknown existing value is ``sensitive``)."""
    if existing is None:
        return mark.name.lower()
    return max(STAMP_VALUES.get(existing, Level.SENSITIVE), mark).name.lower()


@dataclass(frozen=True)
class Admission:
    """The rows a surface or sink may have, how many were withheld, and the highest label among the admitted ones."""

    admitted: list[MemoryEntry]
    withheld: int
    top: Level = Level.TEAM


class LabelPolicy:
    """An immutable policy. Obtain the current one with :meth:`current`; it re-reads labels.yaml only when the file's identity changes."""

    def __init__(self, source: Source, data: _file.FileData | None = None) -> None:
        self.source: Source = source
        auto, agent = (data.auto_surface_max, data.agent_max) if data else (Level.TEAM, Level.PERSONAL)
        self._clearance: dict[Surface | Sink, Level] = {
            Surface.AUTO: auto,
            Surface.AGENT: agent,
            Sink.PLATFORM: Level.TEAM,
            Sink.PROJECT_FILES: Level.TEAM,
        }
        self._strict = source == "strict"
        self._tag_levels: dict[str, Level] = {}
        namespaced: list[_file.Rule] = []
        for rule in data.rules if data else ():
            if rule.namespace is not None:
                namespaced.append(rule)
                continue
            for tag in rule.tags:
                self._tag_levels[tag] = max(self._tag_levels.get(tag, Level.PUBLIC), rule.level)
        self._namespaced = tuple(namespaced)
        self._has_rules = bool(self._tag_levels or self._namespaced)
        self._ns_cache: dict[str, tuple[Level, dict[str, Level]]] = {}

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def current(cls) -> LabelPolicy:
        """The policy now in force: default (no file), file, or strict (a file that cannot be trusted). One ``lstat`` when nothing changed."""
        path = _base_dir() / _file.FILE_NAME
        try:
            st = _stat_file(path)
            identity: tuple[int, ...] | None = None if st is None else _identity(st)
        except OSError:
            st, identity = None, (-1,)
        cached = _CACHE.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
        with _LOCK:
            cached = _CACHE.get(path)
            if cached is not None and cached[0] == identity:
                return cached[1]
            policy = _build(path, st, identity)
            _CACHE[path] = (identity, policy)
            return policy

    # ── labelling ────────────────────────────────────────────────────────────

    def label_of(self, entry: MemoryEntry) -> Level:
        """The row's label: the max of the team floor, its namespace, every matching rule and its stamp. Never raises: an error is ``sensitive``."""
        try:
            return self._label(entry)
        except Exception:  # trw-fail-silent-allow: a row that cannot be labelled is the most sensitive one (fail closed); the log names no content
            logger.warning("labels_row_evaluation_failed", level=Level.SENSITIVE.name.lower())
            return Level.SENSITIVE

    def _label(self, entry: MemoryEntry) -> Level:
        namespace = entry.namespace
        level = Level.TEAM
        if namespace.startswith(_USER_PREFIX) and (self._strict or namespace != _LOCAL_USER):
            level = Level.PERSONAL
        if self._has_rules:
            floor, by_tag = self._for_namespace(namespace)
            level = max(level, floor)
            for tag in entry.tags:
                key = tag_key(tag)
                level = max(level, self._tag_levels.get(key, Level.PUBLIC), by_tag.get(key, Level.PUBLIC))
        stamp = entry.metadata.get(_STAMP_KEY)
        if stamp is not None:
            level = max(level, STAMP_VALUES.get(stamp, Level.SENSITIVE))
        return level

    def _for_namespace(self, namespace: str) -> tuple[Level, dict[str, Level]]:
        """``(level every row of this namespace gets, level per tag in this namespace)`` from the namespace-bound rules; cached."""
        cached = self._ns_cache.get(namespace)
        if cached is not None:
            return cached
        floor = Level.PUBLIC
        by_tag: dict[str, Level] = {}
        for rule in self._namespaced:
            if rule.namespace is None or not fnmatchcase(namespace, rule.namespace):
                continue
            if not rule.tags:
                floor = max(floor, rule.level)
                continue
            for tag in rule.tags:
                by_tag[tag] = max(by_tag.get(tag, Level.PUBLIC), rule.level)
        if len(self._ns_cache) >= _NS_CACHE_MAX:
            self._ns_cache.clear()
        self._ns_cache[namespace] = (floor, by_tag)
        return floor, by_tag

    # ── stamps ───────────────────────────────────────────────────────────────

    def highest(self, entries: Sequence[MemoryEntry]) -> Level:
        """The highest label among *entries* (``team`` for none): what a recall that returned them raises the session mark to."""
        top = Level.TEAM
        for entry in entries:
            top = max(top, self.label_of(entry))
        return top

    def stamped(self, metadata: Mapping[str, str], mark: Level, entry_level: Level) -> dict[str, str]:
        """*metadata* for a row about to be written: stamped with *mark* when the mark exceeds the row's own label, else unchanged.

        Raise-only: an existing higher stamp is kept, and a session that never rose adds no key (existing tests assert ``metadata`` exactly).
        """
        out = dict(metadata)
        if mark > entry_level:
            out[_STAMP_KEY] = _higher_stamp(out.get(_STAMP_KEY), mark)
        return out

    def joined(self, base: Mapping[str, str], other: Mapping[str, str]) -> dict[str, str]:
        """*base* with the maximum stamp of *base* and *other*: what a merge or a correction does so a stamp can only rise.

        Neither side stamped means no key is added. An unknown stamp counts as ``sensitive``.
        """
        out = dict(base)
        stamps = [value for value in (base.get(_STAMP_KEY), other.get(_STAMP_KEY)) if value is not None]
        if stamps:
            out[_STAMP_KEY] = max(
                (STAMP_VALUES.get(value, Level.SENSITIVE) for value in stamps), default=Level.TEAM
            ).name.lower()
        return out

    # ── admission ────────────────────────────────────────────────────────────

    def admit(self, entries: Sequence[MemoryEntry], to: Surface | Sink) -> Admission:
        """The rows of *entries* whose label is at or below *to*'s clearance (equal levels flow), in order."""
        clearance = self._clearance[to]
        admitted: list[MemoryEntry] = []
        top = Level.TEAM
        for entry in entries:
            level = self.label_of(entry)
            if level <= clearance:
                admitted.append(entry)
                top = max(top, level)
        return Admission(admitted, len(entries) - len(admitted), top)


# ── the file-identity cache ──────────────────────────────────────────────────

_CACHE: dict[Path, tuple[tuple[int, ...] | None, LabelPolicy]] = {}
_LOCK = threading.Lock()
_BASE: tuple[tuple[str | None, str | None, str | None], Path] | None = None
_WARNED: set[tuple[str, str]] = set()


def _base_dir() -> Path:
    """The USER base directory (``$TRW_USER_DIR``, else ``$XDG_DATA_HOME/trw``, else ``~/.trw``); recomputed only when those variables change."""
    global _BASE
    env = (os.environ.get("TRW_USER_DIR"), os.environ.get("XDG_DATA_HOME"), os.environ.get("HOME"))
    cached = _BASE
    if cached is not None and cached[0] == env:
        return cached[1]
    from trw_memory.user_paths import user_memory_dir_path

    base = user_memory_dir_path().parent
    _BASE = (env, base)
    return base


def _stat_file(path: Path) -> os.stat_result | None:
    """``lstat`` of the policy file (a symlink is judged, not followed); ``None`` when there is no file."""
    try:
        return os.lstat(path)
    except FileNotFoundError:  # trw-fail-silent-allow: no labels.yaml is the documented default policy, not an error
        return None


def _identity(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, st.st_mode)


def _parse_file(path: Path, st: os.stat_result) -> _file.FileData:
    return _file.parse(_file.read_text(path, st))


def _build(path: Path, st: os.stat_result | None, identity: tuple[int, ...] | None) -> LabelPolicy:
    if identity is None:
        return LabelPolicy("default")
    if st is None:  # the file is there but could not even be stat'ed
        return _strict(path, "unreadable")
    try:
        return LabelPolicy("file", _parse_file(path, st))
    except _file.InvalidLabelsFile as exc:
        return _strict(path, exc.reason)
    except (
        Exception
    ):  # trw-fail-silent-allow: any unexpected failure reading the policy file is a strict policy, never an open one
        return _strict(path, "unexpected_error")


def _strict(path: Path, reason: str) -> LabelPolicy:
    """The fail-closed policy; warns once per (file, reason) naming the path and the error class, never the contents."""
    key = (str(path), reason)
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning("labels_policy_strict", path=str(path), error_class="InvalidLabelsFile", reason=reason)
    return LabelPolicy("strict")
