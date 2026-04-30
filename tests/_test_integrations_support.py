"""Shared helpers for split integration adapter tests."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import sys
import types
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from trw_memory.models.memory import MemoryEntry


@pytest.fixture()
def tmp_backend(tmp_path: Any) -> Generator[Any, None, None]:
    """Create a temporary SQLite backend for testing."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "test" / "memory.db"
    backend = SQLiteBackend(db_path=db_path, dim=384)
    yield backend
    backend.close()


@pytest.fixture()
def sample_entry() -> MemoryEntry:
    """Create a sample MemoryEntry for testing."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id="M-test0001",
        content="test content",
        detail="",
        tags=["test"],
        importance=0.5,
        namespace="default",
        metadata={},
        created_at=now,
        updated_at=now,
        source="agent",
    )


def _make_langchain_mocks() -> dict[str, types.ModuleType]:
    """Create mock modules for langchain_core."""
    lc_core = types.ModuleType("langchain_core")
    lc_core.__spec__ = importlib.machinery.ModuleSpec("langchain_core", None)

    lc_history = types.ModuleType("langchain_core.chat_history")

    class BaseChatMessageHistory:
        """Mock BaseChatMessageHistory ABC."""

    lc_history.BaseChatMessageHistory = BaseChatMessageHistory  # type: ignore[attr-defined]
    lc_core.chat_history = lc_history  # type: ignore[attr-defined]

    lc_messages = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content: str = "") -> None:
            self.content = content
            self.type = "human"

    class HumanMessage(BaseMessage):
        def __init__(self, content: str = "") -> None:
            super().__init__(content)
            self.type = "human"

    class AIMessage(BaseMessage):
        def __init__(self, content: str = "") -> None:
            super().__init__(content)
            self.type = "ai"

    lc_messages.BaseMessage = BaseMessage  # type: ignore[attr-defined]
    lc_messages.HumanMessage = HumanMessage  # type: ignore[attr-defined]
    lc_messages.AIMessage = AIMessage  # type: ignore[attr-defined]
    lc_core.messages = lc_messages  # type: ignore[attr-defined]

    return {
        "langchain_core": lc_core,
        "langchain_core.chat_history": lc_history,
        "langchain_core.messages": lc_messages,
    }


def _make_llamaindex_mocks() -> dict[str, types.ModuleType]:
    """Create mock modules for llama_index.core."""
    import enum

    li_core = types.ModuleType("llama_index")
    li_core.__spec__ = importlib.machinery.ModuleSpec("llama_index", None)
    li_core_mod = types.ModuleType("llama_index.core")
    li_core_mod.__spec__ = importlib.machinery.ModuleSpec("llama_index.core", None)

    li_llms = types.ModuleType("llama_index.core.llms")

    class MessageRole(str, enum.Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        TOOL = "tool"
        CHATBOT = "chatbot"
        MODEL = "model"
        FUNCTION = "function"

    class ChatMessage:
        def __init__(self, role: Any = MessageRole.USER, content: str = "") -> None:
            self.role = role
            self.content = content

    li_llms.ChatMessage = ChatMessage  # type: ignore[attr-defined]
    li_llms.MessageRole = MessageRole  # type: ignore[attr-defined]

    li_storage = types.ModuleType("llama_index.core.storage")
    li_chat_store = types.ModuleType("llama_index.core.storage.chat_store")
    li_chat_store_base = types.ModuleType("llama_index.core.storage.chat_store.base")

    class BaseChatStore:
        """Mock BaseChatStore ABC."""

        model_config: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            pass

        @classmethod
        def class_name(cls) -> str:
            return cls.__name__

    li_chat_store_base.BaseChatStore = BaseChatStore  # type: ignore[attr-defined]
    li_chat_store.base = li_chat_store_base  # type: ignore[attr-defined]
    li_storage.chat_store = li_chat_store  # type: ignore[attr-defined]

    li_core_mod.llms = li_llms  # type: ignore[attr-defined]
    li_core_mod.storage = li_storage  # type: ignore[attr-defined]
    li_core.core = li_core_mod  # type: ignore[attr-defined]

    return {
        "llama_index": li_core,
        "llama_index.core": li_core_mod,
        "llama_index.core.llms": li_llms,
        "llama_index.core.storage": li_storage,
        "llama_index.core.storage.chat_store": li_chat_store,
        "llama_index.core.storage.chat_store.base": li_chat_store_base,
    }


def _make_crewai_mocks() -> dict[str, types.ModuleType]:
    """Create mock modules for crewai."""
    crewai_mod = types.ModuleType("crewai")
    crewai_mod.__spec__ = importlib.machinery.ModuleSpec("crewai", None)
    return {"crewai": crewai_mod}


def _purge_modules(prefix: str) -> None:
    for key in list(sys.modules.keys()):
        if key.startswith(prefix):
            del sys.modules[key]


def _import_adapter(
    prefix: str,
    module_name: str,
    mocks: dict[str, types.ModuleType],
) -> types.ModuleType:
    """Import an adapter module with mocked framework deps."""
    _purge_modules(prefix)
    with patch.dict(sys.modules, mocks):
        return importlib.import_module(module_name)


def _import_langchain_adapter(
    mocks: dict[str, types.ModuleType],
) -> types.ModuleType:
    return _import_adapter(
        "trw_memory.integrations.langchain",
        "trw_memory.integrations.langchain",
        mocks,
    )


def _import_llamaindex_adapter(
    mocks: dict[str, types.ModuleType],
) -> types.ModuleType:
    return _import_adapter(
        "trw_memory.integrations.llamaindex",
        "trw_memory.integrations.llamaindex",
        mocks,
    )


def _import_crewai_adapter(
    mocks: dict[str, types.ModuleType],
) -> types.ModuleType:
    orig_version = importlib.metadata.version

    def _patched_version(name: str) -> str:
        if name == "crewai":
            return "0.74.0"
        return orig_version(name)

    with patch("importlib.metadata.version", side_effect=_patched_version):
        return _import_adapter(
            "trw_memory.integrations.crewai",
            "trw_memory.integrations.crewai",
            mocks,
        )
