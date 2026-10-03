"""The session high-water mark (PRD-SEC-023 FR04).

A session starts at ``team``. Every row a recall returns raises it; nothing lowers it, and a new session (a new server process) is the only
reset. A row written while the mark is above the row's own label carries the mark as its stamp.
"""

from __future__ import annotations

import threading

from trw_memory.labels._levels import Level


class SessionMark:
    """Thread-safe and monotone: ``level`` is the maximum of everything ever passed to :meth:`raise_to`."""

    def __init__(self) -> None:
        self._level = Level.TEAM
        self._lock = threading.Lock()

    @property
    def level(self) -> Level:
        return self._level

    def raise_to(self, level: Level) -> Level:
        """Raise the mark to *level* if that is higher; return the mark now."""
        if level <= self._level:
            return self._level
        with self._lock:
            if level > self._level:
                self._level = level
            return self._level

    def reported(self) -> str | None:
        """The level name a response should report (``session_label``), or ``None`` while the mark is still ``team``."""
        level = self._level
        return level.name.lower() if level > Level.TEAM else None
