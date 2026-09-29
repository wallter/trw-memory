"""Ordering trw-memory versions for the daemon's upgrade decisions (DAEMON-AUTO-RESTART-ON-UPGRADE).

A daemon is replaced only when it is provably OLDER, so every reading that cannot
order two versions answers "not older": a version is its leading numeric release
components (``"5.0.1rc1"`` -> ``(5, 0, 1)``), compared as integers (``"10" > "9"``),
and one with no leading number (``"unknown"``, ``""``) orders against nothing.
"""

from __future__ import annotations

import re

__all__ = ["is_older", "major", "majors_differ", "version_key"]

_RELEASE = re.compile(r"\d+(?:\.\d+)*")


def version_key(version: str) -> tuple[int, ...] | None:
    """The numeric release components of *version*, or ``None`` when it has none."""
    found = _RELEASE.match(version.strip())
    return tuple(int(part) for part in found.group(0).split(".")) if found else None


def major(version: str) -> int | None:
    """The leading integer of *version*, or ``None`` when it has none (``"unknown"``)."""
    key = version_key(version)
    return key[0] if key else None


def majors_differ(version: str, other: str) -> bool:
    """Whether both versions have a major and the two differ; an unreadable one differs from nothing."""
    mine, theirs = major(version), major(other)
    return mine is not None and theirs is not None and mine != theirs


def is_older(version: str, than: str) -> bool:
    """Whether *version* is provably older than *than*; ``False`` when either cannot be read.

    Missing trailing components are zeros, so ``"9.9"`` equals ``"9.9.0"`` rather than preceding it.
    """
    mine, theirs = version_key(version), version_key(than)
    if mine is None or theirs is None:
        return False
    width = max(len(mine), len(theirs))
    return mine + (0,) * (width - len(mine)) < theirs + (0,) * (width - len(theirs))
