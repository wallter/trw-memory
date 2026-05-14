# ruff: noqa: F401,F811
"""Tests for the LangChain integration adapter."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ._test_integrations_support import (
    _import_langchain_adapter,
    _make_langchain_mocks,
    _purge_modules,
    sample_entry,
    tmp_backend,
)


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
        base_cls = self.mocks["langchain_core.chat_history"].BaseChatMessageHistory
        assert issubclass(self.mod.TRWChatMessageHistory, base_cls)

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
        _purge_modules("trw_memory.integrations.langchain")

        saved = {}
        for key in list(sys.modules.keys()):
            if "langchain" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.langchain")
        finally:
            sys.modules.update(saved)
            _purge_modules("trw_memory.integrations.langchain")

    def test_close_non_owned_backend(self) -> None:
        """close() on non-owned backend does not call backend.close()."""
        mock_backend = MagicMock()
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=mock_backend)
        history.close()
        mock_backend.close.assert_not_called()

    def test_close_owned_backend(self) -> None:
        """close() on owned backend releases resources."""
        mock_backend = MagicMock()
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=mock_backend)
        history._owns_backend = True
        history.close()
        mock_backend.close.assert_called_once()

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        history = self.mod.TRWChatMessageHistory(session_id="s1", backend=tmp_backend)
        with history as h:
            assert h is history
