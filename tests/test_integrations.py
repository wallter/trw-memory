"""Tests for trw-memory integration adapters.

All framework dependencies (langchain-core, llama-index-core, crewai) are
mocked so tests run in a base install without any extras.

Test count target: >= 40 test functions.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.machinery
import sys
import types
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.integrations._backend import create_backend, make_entry
from trw_memory.models.memory import MemoryEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock framework modules
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared import helper (DRY)
# ---------------------------------------------------------------------------


def _import_adapter(
    prefix: str,
    module_name: str,
    mocks: dict[str, types.ModuleType],
) -> types.ModuleType:
    """Import an adapter module with mocked framework deps.

    Clears any cached import matching *prefix*, then imports *module_name*
    under the mocked ``sys.modules``.
    """
    for key in list(sys.modules.keys()):
        if key.startswith(prefix):
            del sys.modules[key]
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


# ===================================================================
# BACKEND HELPER TESTS
# ===================================================================


class TestBackendHelper:
    """Tests for _backend.py helpers."""

    def test_make_entry_generates_id(self) -> None:
        entry = make_entry(content="test", namespace="ns")
        assert entry.id.startswith("M-")
        assert len(entry.id) == 18  # M- + 16 hex chars

    def test_make_entry_sets_timestamps(self) -> None:
        entry = make_entry(content="test")
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert entry.created_at.tzinfo == timezone.utc

    def test_make_entry_sets_tags(self) -> None:
        entry = make_entry(content="test", tags=["a", "b"])
        assert entry.tags == ["a", "b"]

    def test_create_backend_returns_storage_backend(self, tmp_path: Any) -> None:
        from trw_memory.storage.interface import StorageBackend

        backend = create_backend("test", storage_path=str(tmp_path))
        try:
            assert isinstance(backend, StorageBackend)
        finally:
            backend.close()

    def test_resolve_backend_with_provided_backend(self) -> None:
        """resolve_backend returns provided backend without ownership."""
        from trw_memory.integrations._backend import resolve_backend

        mock_backend = MagicMock()
        backend, owns = resolve_backend("ns", None, mock_backend)
        assert backend is mock_backend
        assert not owns

    def test_resolve_backend_creates_new_backend(self, tmp_path: Any) -> None:
        """resolve_backend creates and owns backend when none provided."""
        from trw_memory.integrations._backend import resolve_backend
        from trw_memory.storage.interface import StorageBackend

        backend, owns = resolve_backend("test", str(tmp_path), None)
        try:
            assert isinstance(backend, StorageBackend)
            assert owns
        finally:
            backend.close()


# ===================================================================
# LANGCHAIN ADAPTER TESTS (12 tests)
# ===================================================================


class TestLangChainAdapter:
    """Tests for TRWChatMessageHistory."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        self.mocks = _make_langchain_mocks()
        self.mod = _import_langchain_adapter(self.mocks)
        self.HumanMessage = self.mocks["langchain_core.messages"].HumanMessage
        self.AIMessage = self.mocks["langchain_core.messages"].AIMessage

    def test_is_subclass_of_base(self) -> None:
        """UT-LC-01: TRWChatMessageHistory is a subclass of BaseChatMessageHistory."""
        BaseCls = self.mocks["langchain_core.chat_history"].BaseChatMessageHistory
        assert issubclass(self.mod.TRWChatMessageHistory, BaseCls)

    def test_session_id_stored(self, tmp_backend: Any) -> None:
        """UT-LC-10: Constructor sets session_id."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        assert history.session_id == "s1"

    def test_messages_empty_initially(self, tmp_backend: Any) -> None:
        """UT-LC-02: Empty history returns empty list."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        assert history.messages == []

    def test_add_messages_stores_entries(self, tmp_backend: Any) -> None:
        """UT-LC-04: add_messages stores entries retrievable via messages."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        history.add_messages([self.HumanMessage("hello"), self.AIMessage("hi")])

        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "hi"

    def test_messages_returns_chronological_order(self, tmp_backend: Any) -> None:
        """UT-LC-03: Messages are returned in insertion order."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        for i in range(5):
            history.add_messages([self.HumanMessage(f"msg-{i}")])

        msgs = history.messages
        assert [m.content for m in msgs] == [f"msg-{i}" for i in range(5)]

    def test_clear_removes_all_messages(self, tmp_backend: Any) -> None:
        """UT-LC-05: clear() removes all messages for the session."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        history.add_messages([self.HumanMessage("hello")])
        assert len(history.messages) == 1

        history.clear()
        assert history.messages == []

    def test_messages_respect_max_results(self, tmp_backend: Any) -> None:
        """messages returns only the most recent max_results items."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", max_results=2, backend=tmp_backend)
        for i in range(4):
            history.add_messages([self.HumanMessage(f"msg-{i}")])

        msgs = history.messages
        assert [m.content for m in msgs] == ["msg-2", "msg-3"]

    def test_session_isolation(self, tmp_backend: Any) -> None:
        """UT-LC-06: Messages from different sessions don't interfere."""
        h1 = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        h2 = self.mod.TRWChatMessageHistory(session_id="s2", backend=tmp_backend)

        h1.add_messages([self.HumanMessage("from s1")])
        h2.add_messages([self.HumanMessage("from s2")])

        assert len(h1.messages) == 1
        assert h1.messages[0].content == "from s1"
        assert len(h2.messages) == 1
        assert h2.messages[0].content == "from s2"

    def test_message_role_preserved(self, tmp_backend: Any) -> None:
        """UT-LC-07: Message roles (human/ai) are preserved."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        history.add_messages([self.HumanMessage("q"), self.AIMessage("a")])

        msgs = history.messages
        assert msgs[0].type == "human"
        assert msgs[1].type == "ai"

    def test_import_error_without_langchain(self) -> None:
        """UT-LC-09: ImportError raised when langchain-core not installed."""
        for key in list(sys.modules.keys()):
            if key.startswith("trw_memory.integrations.langchain"):
                del sys.modules[key]

        saved = {}
        for key in list(sys.modules.keys()):
            if "langchain" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.langchain")
        finally:
            sys.modules.update(saved)
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.langchain"):
                    del sys.modules[key]

    def test_close_non_owned_backend(self) -> None:
        """close() on non-owned backend does not call backend.close()."""
        mock_backend = MagicMock()
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=mock_backend)
        history.close()
        mock_backend.close.assert_not_called()

    def test_close_owned_backend(self, tmp_path: Any) -> None:
        """close() on owned backend releases resources."""
        mock_backend = MagicMock()
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=mock_backend)
        # Force ownership
        history._owns_backend = True
        history.close()
        mock_backend.close.assert_called_once()

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        with history as h:
            assert h is history
        # Should not raise — close is safe on non-owned backend


# ===================================================================
# LLAMAINDEX ADAPTER TESTS (11 tests)
# ===================================================================


class TestLlamaIndexAdapter:
    """Tests for TRWChatStore."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        self.mocks = _make_llamaindex_mocks()
        self.mod = _import_llamaindex_adapter(self.mocks)
        self.ChatMessage = self.mocks["llama_index.core.llms"].ChatMessage
        self.MessageRole = self.mocks["llama_index.core.llms"].MessageRole

    def test_is_subclass_of_base(self) -> None:
        """UT-LI-01: TRWChatStore extends BaseChatStore."""
        BaseCls = self.mocks["llama_index.core.storage.chat_store.base"].BaseChatStore
        assert issubclass(self.mod.TRWChatStore, BaseCls)

    def test_class_name(self) -> None:
        """UT-LI-02: class_name() returns 'TRWChatStore'."""
        assert self.mod.TRWChatStore.class_name() == "TRWChatStore"

    def test_add_and_get_messages(self, tmp_backend: Any) -> None:
        """UT-LI-03: add_message + get_messages round-trip."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("s1", self.ChatMessage(role=self.MessageRole.USER, content="hello"))
        store.add_message("s1", self.ChatMessage(role=self.MessageRole.ASSISTANT, content="hi"))

        msgs = store.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "hi"

    def test_delete_messages(self, tmp_backend: Any) -> None:
        """UT-LI-05: delete_messages removes all messages for key."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("s1", self.ChatMessage(content="hello"))
        assert len(store.get_messages("s1")) == 1

        result = store.delete_messages("s1")
        assert result is not None
        assert len(result) == 1  # the message that was deleted
        assert store.get_messages("s1") == []

    def test_delete_messages_returns_none_for_empty_key(self, tmp_backend: Any) -> None:
        """delete_messages returns None when key has no messages."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        result = store.delete_messages("nonexistent")
        assert result is None

    def test_get_keys(self, tmp_backend: Any) -> None:
        """UT-LI-06: get_keys returns all conversation keys."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("session-a", self.ChatMessage(content="a"))
        store.add_message("session-b", self.ChatMessage(content="b"))
        store.add_message("session-a", self.ChatMessage(content="a2"))

        keys = store.get_keys()
        assert keys == ["session-a", "session-b"]

    def test_get_messages_respects_message_limit(self, tmp_backend: Any) -> None:
        """get_messages returns only the most recent message_limit items."""
        store = self.mod.TRWChatStore(namespace="test", message_limit=2, backend=tmp_backend)
        for i in range(4):
            store.add_message("s1", self.ChatMessage(content=f"msg-{i}"))

        msgs = store.get_messages("s1")
        assert [m.content for m in msgs] == ["msg-2", "msg-3"]

    def test_set_messages_replaces(self, tmp_backend: Any) -> None:
        """UT-LI-04: set_messages replaces existing messages."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("s1", self.ChatMessage(content="old"))
        store.set_messages("s1", [self.ChatMessage(content="new")])

        msgs = store.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0].content == "new"

    def test_delete_last_message(self, tmp_backend: Any) -> None:
        """UT-LI-07: delete_last_message removes most recent."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("s1", self.ChatMessage(content="first"))
        store.add_message("s1", self.ChatMessage(content="second"))

        removed = store.delete_last_message("s1")
        assert removed is not None
        assert removed.content == "second"
        assert len(store.get_messages("s1")) == 1

    def test_delete_last_message_empty_returns_none(self, tmp_backend: Any) -> None:
        """delete_last_message on empty key returns None."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        assert store.delete_last_message("empty") is None

    def test_delete_message_invalid_index(self, tmp_backend: Any) -> None:
        """delete_message with out-of-bounds index returns None."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message("s1", self.ChatMessage(content="only"))
        assert store.delete_message("s1", 5) is None
        assert store.delete_message("s1", -1) is None

    def test_delete_message_with_message_limit_preserves_older_history(self, tmp_backend: Any) -> None:
        """Deleting from the visible window must not drop older hidden messages."""
        store = self.mod.TRWChatStore(namespace="test", message_limit=2, backend=tmp_backend)
        for i in range(4):
            store.add_message("s1", self.ChatMessage(content=f"msg-{i}"))

        removed = store.delete_message("s1", 0)
        assert removed is not None
        assert removed.content == "msg-2"

        all_messages = self.mod.TRWChatStore(namespace="test", message_limit=10, backend=tmp_backend).get_messages("s1")
        assert [message.content for message in all_messages] == ["msg-0", "msg-1", "msg-3"]

    def test_import_error_without_llamaindex(self) -> None:
        """UT-LI-08: ImportError raised when llama-index-core not installed."""
        for key in list(sys.modules.keys()):
            if key.startswith("trw_memory.integrations.llamaindex"):
                del sys.modules[key]

        saved = {}
        for key in list(sys.modules.keys()):
            if "llama_index" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.llamaindex")
        finally:
            sys.modules.update(saved)
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.llamaindex"):
                    del sys.modules[key]

    def test_get_keys_scans_full_namespace_not_default_limit(self, tmp_backend: Any) -> None:
        """get_keys must not stop at the adapter default list window."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        for key in ["session-a", "session-b", "session-c"]:
            store.add_message(key, self.ChatMessage(content=key))

        with patch("trw_memory.integrations._backend.DEFAULT_LIST_LIMIT", 1):
            assert store.get_keys() == ["session-a", "session-b", "session-c"]

    def test_delete_messages_scans_full_namespace_not_default_limit(self, tmp_backend: Any) -> None:
        """delete_messages must remove all key entries even with a tiny list window."""
        store = self.mod.TRWChatStore(namespace="test", backend=tmp_backend)
        for i in range(3):
            store.add_message("s1", self.ChatMessage(content=f"msg-{i}"))

        with patch("trw_memory.integrations._backend.DEFAULT_LIST_LIMIT", 1):
            deleted = store.delete_messages("s1")

        assert deleted is not None
        assert [message.content for message in deleted] == ["msg-0", "msg-1", "msg-2"]
        assert store.get_messages("s1") == []


# ===================================================================
# CREWAI ADAPTER TESTS (10 tests)
# ===================================================================


class TestCrewAIAdapter:
    """Tests for TRWCrewStorage."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        self.mocks = _make_crewai_mocks()
        self.mod = _import_crewai_adapter(self.mocks)

    def test_instantiation(self, tmp_backend: Any) -> None:
        """UT-CA-01: TRWCrewStorage instantiates correctly."""
        storage = self.mod.TRWCrewStorage(namespace="test", backend=tmp_backend)
        assert storage.namespace == "test"

    def test_save_stores_entry(self, tmp_backend: Any) -> None:
        """UT-CA-02: save() stores content in backend."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("found a bug in module X")

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert len(entries) == 1
        assert entries[0].content == "found a bug in module X"

    def test_save_with_agent_tag(self, tmp_backend: Any) -> None:
        """UT-CA-03: save() with agent adds agent tag."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("finding", agent="researcher")

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert "agent:researcher" in entries[0].tags

    def test_search_returns_results(self, tmp_backend: Any) -> None:
        """UT-CA-04: search() returns matching entries."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("bug in authentication module")

        results = storage.search("authentication")
        assert len(results) >= 1
        assert "context" in results[0]

    def test_search_score_threshold(self, tmp_backend: Any) -> None:
        """search() with score_threshold filters results."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("low importance entry")

        # All entries have importance 0.5, so threshold 0.6 should filter all
        results = storage.search("entry", score_threshold=0.6)
        assert len(results) == 0

        # Threshold 0.5 should include
        results = storage.search("entry", score_threshold=0.5)
        assert len(results) >= 1

    def test_reset_clears_all(self, tmp_backend: Any) -> None:
        """UT-CA-05: reset() clears all entries."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("entry 1")
        storage.save("entry 2")
        assert tmp_backend.count(namespace="default") == 2

        storage.reset()
        assert tmp_backend.count(namespace="default") == 0

    def test_reset_uses_bulk_namespace_delete(self, tmp_backend: Any) -> None:
        """reset() should clear the namespace through the backend bulk path."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        with patch.object(tmp_backend, "delete_by_namespace", wraps=tmp_backend.delete_by_namespace) as delete_namespace:
            storage.save("entry 1")
            storage.save("entry 2")
            storage.reset()

        delete_namespace.assert_called_once_with("default")
        assert tmp_backend.count(namespace="default") == 0

    def test_save_with_metadata(self, tmp_backend: Any) -> None:
        """UT-CA-06: save() passes metadata to backend."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("finding", metadata={"priority": "high"})

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert entries[0].metadata.get("priority") == "high"

    def test_search_respects_limit(self, tmp_backend: Any) -> None:
        """UT-CA-07: search() respects limit parameter."""
        storage = self.mod.TRWCrewStorage(namespace="default", search_limit=5, backend=tmp_backend)
        for i in range(10):
            storage.save(f"entry about topic {i}")

        results = storage.search("topic", limit=3)
        assert len(results) <= 3

    def test_search_applies_metadata_filter(self, tmp_backend: Any) -> None:
        """search() applies exact-match metadata filters."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("auth finding", metadata={"team": "auth"})
        storage.save("billing finding", metadata={"team": "billing"})

        results = storage.search("finding", filter={"team": "auth"})
        assert [result["context"] for result in results] == ["auth finding"]

    def test_import_error_without_crewai(self) -> None:
        """UT-CA-08: ImportError raised when crewai not installed."""
        for key in list(sys.modules.keys()):
            if key.startswith("trw_memory.integrations.crewai"):
                del sys.modules[key]

        saved = {}
        for key in list(sys.modules.keys()):
            if key == "crewai" or key.startswith("crewai."):
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.crewai")
        finally:
            sys.modules.update(saved)
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.crewai"):
                    del sys.modules[key]

    def test_import_error_with_too_old_crewai_version(self) -> None:
        """CrewAI adapter rejects versions older than the documented floor."""
        for key in list(sys.modules.keys()):
            if key.startswith("trw_memory.integrations.crewai"):
                del sys.modules[key]

        mocks = _make_crewai_mocks()
        with patch.dict(sys.modules, mocks):
            with patch("importlib.metadata.version", return_value="0.73.9"):
                with pytest.raises(ImportError, match="crewai>=0.74.0"):
                    importlib.import_module("trw_memory.integrations.crewai")

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        with storage as s:
            assert s is storage


# ===================================================================
# VSCODE INTERFACE TESTS (9 tests)
# ===================================================================


class TestVSCodeInterface:
    """Tests for VSCodeMemoryInterface and LocalMemoryAdapter."""

    def test_protocol_importable_without_extras(self) -> None:
        """UT-VS-01: VSCodeMemoryInterface imports in base install."""
        from trw_memory.integrations.vscode import VSCodeMemoryInterface

        # Verify it's a usable protocol class, not just a non-None import
        assert hasattr(VSCodeMemoryInterface, "__protocol_attrs__") or callable(VSCodeMemoryInterface)

    def test_protocol_has_all_methods(self) -> None:
        """UT-VS-02: VSCodeMemoryInterface declares all 4 methods."""
        from trw_memory.integrations.vscode import VSCodeMemoryInterface

        methods = [m for m in dir(VSCodeMemoryInterface) if not m.startswith("_")]
        assert "get_relevant" in methods
        assert "store_selection" in methods
        assert "search" in methods
        assert "get_status" in methods

    def test_local_adapter_satisfies_protocol(self) -> None:
        """UT-VS-03: LocalMemoryAdapter satisfies VSCodeMemoryInterface."""
        from trw_memory.integrations.vscode import (
            LocalMemoryAdapter,
            VSCodeMemoryInterface,
        )

        assert isinstance(LocalMemoryAdapter.__new__(LocalMemoryAdapter), VSCodeMemoryInterface)

    def test_get_relevant(self, tmp_backend: Any) -> None:
        """UT-VS-04: get_relevant returns memories relevant to file path."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        adapter.store_selection("use pytest fixtures", "/src/test.py", ["testing"])

        results = adapter.get_relevant("/src/test.py", limit=5)
        assert isinstance(results, list)

    def test_store_selection(self, tmp_backend: Any) -> None:
        """UT-VS-05: store_selection stores content with file tag."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        result = adapter.store_selection("code snippet", "/file.py", ["python"])

        assert "memory_id" in result
        assert result["status"] == "stored"

        entries = tmp_backend.list_entries(namespace="test", limit=100)
        assert len(entries) == 1
        assert "file:/file.py" in entries[0].tags

    def test_get_status(self, tmp_backend: Any) -> None:
        """UT-VS-06: get_status returns health metrics."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        status = adapter.get_status()

        assert "entry_count" in status
        assert "namespace" in status
        assert status["namespace"] == "test"
        assert status["entry_count"] == 0

    def test_search_uses_instance_namespace_by_default(self, tmp_backend: Any) -> None:
        """search() defaults to adapter's namespace, not 'default'."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="my-ns", backend=tmp_backend)
        adapter.store_selection("content", "/f.py", [])

        # search without explicit namespace should use "my-ns"
        results = adapter.search("content")
        assert isinstance(results, list)

    def test_search_with_explicit_namespace(self, tmp_backend: Any) -> None:
        """search() with explicit namespace overrides default."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="my-ns", backend=tmp_backend)
        # Searching with a different namespace should work
        results = adapter.search("query", namespace="other-ns")
        assert isinstance(results, list)

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        with adapter as a:
            assert a is adapter


# ===================================================================
# FACTORY TESTS (8 tests)
# ===================================================================


class TestFactory:
    """Tests for get_adapter and list_available."""

    def test_get_adapter_langchain_with_dep(self) -> None:
        """UT-FA-01: get_adapter('langchain') returns adapter when installed."""
        mocks = _make_langchain_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.langchain"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "langchain_core":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                from trw_memory.integrations.factory import get_adapter

                cls = get_adapter("langchain")
                assert cls.__name__ == "TRWChatMessageHistory"

    def test_get_adapter_langchain_without_dep(self) -> None:
        """UT-FA-02: get_adapter('langchain') raises ImportError when missing."""
        orig_find_spec = importlib.util.find_spec

        def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
            if name == "langchain_core":
                return None
            return orig_find_spec(name, *a, **kw)

        with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
            from trw_memory.integrations.factory import get_adapter

            with pytest.raises(ImportError, match="pip install"):
                get_adapter("langchain")

    def test_get_adapter_llamaindex(self) -> None:
        """UT-FA-03: get_adapter('llamaindex') returns TRWChatStore."""
        mocks = _make_llamaindex_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.llamaindex"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "llama_index.core":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                from trw_memory.integrations.factory import get_adapter

                cls = get_adapter("llamaindex")
                assert cls.__name__ == "TRWChatStore"

    def test_get_adapter_crewai(self) -> None:
        """UT-FA-04: get_adapter('crewai') returns TRWCrewStorage."""
        mocks = _make_crewai_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.crewai"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "crewai":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                with patch("importlib.metadata.version", return_value="0.74.0"):
                    from trw_memory.integrations.factory import get_adapter

                    cls = get_adapter("crewai")
                    assert cls.__name__ == "TRWCrewStorage"

    def test_get_adapter_vscode_no_extras(self) -> None:
        """UT-FA-05: get_adapter('vscode') returns LocalMemoryAdapter without extras."""
        from trw_memory.integrations.factory import get_adapter

        cls = get_adapter("vscode")
        assert cls.__name__ == "LocalMemoryAdapter"

    def test_get_adapter_unknown_raises_valueerror(self) -> None:
        """UT-FA-06: get_adapter('unknown') raises ValueError."""
        from trw_memory.integrations.factory import get_adapter

        with pytest.raises(ValueError, match="Unknown framework"):
            get_adapter("unknown_framework")

    def test_list_available_includes_vscode(self) -> None:
        """UT-FA-07: list_available always includes 'vscode'."""
        from trw_memory.integrations.factory import list_available

        available = list_available()
        assert "vscode" in available

    def test_factory_import_no_framework_modules(self) -> None:
        """UT-FA-08: importing factory doesn't import framework modules."""
        before = set(sys.modules.keys())
        importlib.import_module("trw_memory.integrations.factory")
        after = set(sys.modules.keys())

        new_modules = after - before
        framework_modules = [m for m in new_modules if any(f in m for f in ["langchain", "llama_index", "crewai"])]
        assert framework_modules == [], f"Framework modules loaded: {framework_modules}"


# ===================================================================
# INTEGRATION TESTS (round-trip)
# ===================================================================


class TestIntegration:
    """Integration tests with real storage backend."""

    def test_langchain_round_trip(self, tmp_backend: Any) -> None:
        """IT-01: LangChain save + load round-trip."""
        mocks = _make_langchain_mocks()
        mod = _import_langchain_adapter(mocks)
        HumanMessage = mocks["langchain_core.messages"].HumanMessage
        AIMessage = mocks["langchain_core.messages"].AIMessage

        history = mod.TRWChatMessageHistory(session_id="round-trip", backend=tmp_backend)
        history.add_messages(
            [
                HumanMessage("What is TRW?"),
                AIMessage("TRW is an operational framework."),
            ]
        )

        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "What is TRW?"
        assert msgs[1].content == "TRW is an operational framework."

    def test_llamaindex_round_trip(self, tmp_backend: Any) -> None:
        """IT-02: LlamaIndex add_message + get_messages round-trip."""
        mocks = _make_llamaindex_mocks()
        mod = _import_llamaindex_adapter(mocks)
        ChatMessage = mocks["llama_index.core.llms"].ChatMessage
        MessageRole = mocks["llama_index.core.llms"].MessageRole

        store = mod.TRWChatStore(namespace="test", backend=tmp_backend)
        store.add_message(
            "conv-1",
            ChatMessage(role=MessageRole.USER, content="hello"),
        )
        store.add_message(
            "conv-1",
            ChatMessage(role=MessageRole.ASSISTANT, content="world"),
        )

        msgs = store.get_messages("conv-1")
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "world"

    def test_crewai_save_search_round_trip(self, tmp_backend: Any) -> None:
        """IT-03: CrewAI save + search round-trip."""
        mocks = _make_crewai_mocks()
        mod = _import_crewai_adapter(mocks)
        storage = mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("Important finding about authentication")
        results = storage.search("authentication")
        assert len(results) >= 1

    def test_vscode_store_and_status(self, tmp_backend: Any) -> None:
        """IT-04: VSCode store + status round-trip."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        adapter.store_selection("use fixtures", "/test.py", ["testing"])

        status = adapter.get_status()
        assert status["entry_count"] == 1
