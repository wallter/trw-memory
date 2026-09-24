"""Shared helpers for the daemon discovery test family.

``read_discovery`` was a public PROBE wrapper on
``trw_memory.daemon._discovery.read_discovery_result`` -- read-only diagnostics
that collapse ``DiscoveryAbsent``/``DiscoveryInvalid`` into ``None``. It had no
production caller (verified 2026-09 PRD-CORE-293 slice 5) so it was removed
from the public surface; this local mirror keeps the pre-existing test bodies
that exercise discovery-file behaviour through that same two-valued shape.
"""

from __future__ import annotations

from trw_memory.daemon import DaemonInfo, DaemonPaths
from trw_memory.daemon._discovery import read_discovery_result


def read_discovery(paths: DaemonPaths) -> DaemonInfo | None:
    """Return the recorded daemon, or ``None`` when no record can be trusted."""
    result = read_discovery_result(paths)
    return result if isinstance(result, DaemonInfo) else None
