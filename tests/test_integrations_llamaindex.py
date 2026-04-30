"""Tests for the LlamaIndex integration adapter."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

from ._test_integrations_support import (
    _import_llamaindex_adapter,
    _make_llamaindex_mocks,
    _purge_modules,
    tmp_backend,
)


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
        base_cls = self.mocks["llama_index.core.storage.chat_store.base"].BaseChatStore
        assert issubclass(self.mod.TRWChatStore, base_cls)

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
        assert len(result) == 1
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
        _purge_modules("trw_memory.integrations.llamaindex")

        saved = {}
        for key in list(sys.modules.keys()):
            if "llama_index" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.llamaindex")
        finally:
            sys.modules.update(saved)
            _purge_modules("trw_memory.integrations.llamaindex")

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
