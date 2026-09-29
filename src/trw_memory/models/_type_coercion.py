"""Lenient ``MemoryType`` coercion for raw sync ingress (PRD-CORE-334 FR05).

A row synced from a newer client can carry a type this build does not know. Every raw ingress (sync push/pull,
team pull, the org-shared decode) turns it into ``PATTERN`` and keeps the raw string for ``metadata["type_raw"]``,
so the row is never dropped and the pull never crashes. Split out of ``memory.py`` (350-eLOC gate).
"""

from __future__ import annotations

from trw_memory.models.memory import MemoryType

__all__ = ["coerce_memory_type_lenient"]


def coerce_memory_type_lenient(raw: object) -> tuple[MemoryType, str | None]:
    """PRD-CORE-334 FR05, every raw sync ingress: an unknown type is PATTERN plus the raw value for ``type_raw``."""
    value = raw.value if isinstance(raw, MemoryType) else str(raw or "pattern")
    try:
        return MemoryType(value), None
    except ValueError:
        return MemoryType.PATTERN, value
