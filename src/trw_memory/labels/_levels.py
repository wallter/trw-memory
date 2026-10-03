"""The label levels and the places a row can go (PRD-SEC-023)."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Level(IntEnum):
    """A confidentiality level; a higher number is more sensitive. ``PUBLIC`` is reserved: no phase-0 rule or sink uses it."""

    PUBLIC = 0
    TEAM = 1
    PERSONAL = 2
    SENSITIVE = 3


class Surface(StrEnum):
    """Where a recalled row is shown: ``auto`` is everything the agent did not ask for; ``agent`` is ``trw_recall``."""

    AUTO = "auto"
    AGENT = "agent"


class Sink(StrEnum):
    """Where a row can leave the machine or land in a tracked file; both clear ``team`` only in phase 0."""

    PLATFORM = "platform"
    PROJECT_FILES = "project_files"


#: The only stamp values; any other value in ``metadata['trw_label']`` reads as ``SENSITIVE`` (fail closed).
STAMP_VALUES: dict[str, Level] = {level.name.lower(): level for level in Level}
