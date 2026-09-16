"""Integration adapters for trw-memory.

Provides the VSCode interface contract plus the adapter factory. Adapter
imports are lazy — ``import trw_memory`` never pulls an adapter module in.

The LangChain, LlamaIndex and CrewAI adapters were removed as unused surface;
see CHANGELOG.md [Unreleased] Removed.

Usage::

    from trw_memory.integrations import get_adapter, list_available

    # Factory auto-detect
    adapter_cls = get_adapter("vscode")

    # Direct import
    from trw_memory.integrations.vscode import LocalMemoryAdapter
"""

from __future__ import annotations

from trw_memory.integrations.factory import get_adapter, list_available

__all__ = ["get_adapter", "list_available"]
