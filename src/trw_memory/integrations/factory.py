"""Adapter factory — resolve an integration adapter by name.

Provides :func:`get_adapter` which lazily imports the correct adapter class
based on the requested framework name, and :func:`list_available` which
reports which adapters are currently usable.

No adapter module is imported at module load time.

Usage::

    from trw_memory.integrations.factory import get_adapter, list_available

    cls = get_adapter("vscode")        # -> LocalMemoryAdapter
    available = list_available()       # -> ["vscode"]
"""

from __future__ import annotations

import importlib
import importlib.util

# Mapping of framework name -> (spec_check_module, adapter_module, adapter_class).
#
# A ``spec_check_module`` of ``None`` means the adapter has no external
# dependency, so it is always available. The probe is kept for an adapter that
# does need one; the LangChain, LlamaIndex and CrewAI entries that used it were
# removed along with their modules (see CHANGELOG.md [Unreleased] Removed).
_REGISTRY: dict[str, tuple[str | None, str, str]] = {
    "vscode": (
        None,  # No external dependency required
        "trw_memory.integrations.vscode",
        "LocalMemoryAdapter",
    ),
}


def get_adapter(framework: str) -> type[object]:
    """Return the adapter class for the given framework.

    Args:
        framework: A registered adapter name — currently ``"vscode"``.

    Returns:
        The adapter class (not an instance).

    Raises:
        ValueError: If *framework* is not a recognised name.
        ImportError: If the required optional dependency is not installed.
    """
    if framework not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown framework {framework!r}. Valid options: {valid}")

    spec_module, adapter_module, class_name = _REGISTRY[framework]

    # Check if the external dependency is installed
    if spec_module is not None and importlib.util.find_spec(spec_module) is None:
        raise ImportError(f"{framework} is not installed. Install it with: pip install the {framework} package")

    # Lazy-import the adapter module
    mod = importlib.import_module(adapter_module)
    cls: type[object] = getattr(mod, class_name)
    return cls


def list_available() -> list[str]:
    """Return adapter names whose dependencies are currently installed.

    The ``"vscode"`` adapter is always available (no external dependency).

    Returns:
        Sorted list of available adapter names.
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
