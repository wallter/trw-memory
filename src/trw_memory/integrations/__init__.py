"""Framework integration adapters for trw-memory.

Provides drop-in adapters for LangChain, LlamaIndex, CrewAI, and a VSCode
interface contract.  All framework imports are lazy — ``import trw_memory``
never pulls in any framework dependency.

Usage::

    from trw_memory.integrations import get_adapter, list_available

    # Factory auto-detect
    adapter_cls = get_adapter("langchain")

    # Direct import (requires extras)
    from trw_memory.integrations.langchain import TRWChatMessageHistory
"""

from __future__ import annotations

from trw_memory.integrations.factory import get_adapter, list_available

__all__ = ["get_adapter", "list_available"]
