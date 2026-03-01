"""Namespace-to-path mapping logic.

Extracted from ``trw_memory.namespace`` for package organisation.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.namespaces.validation import validate_namespace


def namespace_to_path(ns: str) -> Path:
    """Map a namespace to a relative directory path.

    Examples:
        >>> namespace_to_path("project:repo-a")
        PosixPath('project/repo-a')
        >>> namespace_to_path("global")
        PosixPath('global')
        >>> namespace_to_path("team:research")
        PosixPath('team/research')

    Args:
        ns: A validated namespace string.

    Returns:
        Relative :class:`Path` suitable for constructing storage directories.

    Raises:
        ConfigError: If *ns* is not a valid namespace.
    """
    validate_namespace(ns)
    # "project:repo-a" -> "project/repo-a", "global" -> "global"
    return Path(ns.replace(":", "/"))
