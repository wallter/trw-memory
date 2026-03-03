"""Adapter factory — auto-detect installed frameworks.

Provides :func:`get_adapter` which lazily imports the correct adapter class
based on the requested framework name, and :func:`list_available` which
reports which framework extras are currently installed.

No framework is imported at module load time.

Usage::

    from trw_memory.integrations.factory import get_adapter, list_available

    cls = get_adapter("langchain")     # -> TRWChatMessageHistory
    available = list_available()       # -> ["langchain", "vscode"]
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

# Mapping of framework name -> (spec_check_module, adapter_module, adapter_class)
_REGISTRY: dict[str, tuple[str | None, str, str]] = {
    "langchain": (
        "langchain_core",
        "trw_memory.integrations.langchain",
        "TRWChatMessageHistory",
    ),
    "llamaindex": (
        "llama_index.core",
        "trw_memory.integrations.llamaindex",
        "TRWChatStore",
    ),
    "crewai": (
        "crewai",
        "trw_memory.integrations.crewai",
        "TRWCrewStorage",
    ),
    "vscode": (
        None,  # No external dependency required
        "trw_memory.integrations.vscode",
        "LocalMemoryAdapter",
    ),
}

_INSTALL_HINTS: dict[str, str] = {
    "langchain": 'pip install "trw-memory[langchain]"',
    "llamaindex": 'pip install "trw-memory[llamaindex]"',
    "crewai": 'pip install "trw-memory[crewai]"',
}


def get_adapter(framework: str) -> type[Any]:
    """Return the adapter class for the given framework.

    Args:
        framework: One of ``"langchain"``, ``"llamaindex"``, ``"crewai"``,
            or ``"vscode"``.

    Returns:
        The adapter class (not an instance).

    Raises:
        ValueError: If *framework* is not a recognised name.
        ImportError: If the required optional dependency is not installed.
    """
    if framework not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown framework {framework!r}. Valid options: {valid}"
        )

    spec_module, adapter_module, class_name = _REGISTRY[framework]

    # Check if the external dependency is installed
    if spec_module is not None and importlib.util.find_spec(spec_module) is None:
        hint = _INSTALL_HINTS.get(framework, f"pip install the {framework} package")
        raise ImportError(
            f"{framework} is not installed. Install it with: {hint}"
        )

    # Lazy-import the adapter module
    mod = importlib.import_module(adapter_module)
    cls: type[Any] = getattr(mod, class_name)
    return cls


def list_available() -> list[str]:
    """Return framework names whose dependencies are currently installed.

    The ``"vscode"`` adapter is always available (no external dependency).

    Returns:
        Sorted list of available framework names.
    """
    available: list[str] = []
    for name, (spec_module, _adapter_module, _class_name) in _REGISTRY.items():
        if spec_module is None:
            # No external dependency — always available
            available.append(name)
        else:
            try:
                if importlib.util.find_spec(spec_module) is not None:
                    available.append(name)
            except (ModuleNotFoundError, ValueError):
                pass
    return sorted(available)
