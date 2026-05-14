# ruff: noqa: F401,F811
"""Round-trip integration tests for adapter families."""

from __future__ import annotations

from typing import Any

from ._test_integrations_support import (
    _import_crewai_adapter,
    _import_langchain_adapter,
    _import_llamaindex_adapter,
    _make_crewai_mocks,
    _make_langchain_mocks,
    _make_llamaindex_mocks,
    tmp_backend,
)


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
        store.add_message("conv-1", ChatMessage(role=MessageRole.USER, content="hello"))
        store.add_message("conv-1", ChatMessage(role=MessageRole.ASSISTANT, content="world"))

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

    def test_langchain_owned_sqlite_backend_persists_across_reopen(
        self,
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """IT-05: LangChain adapter-owned SQLite backend survives reopen."""
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        storage_path = str(tmp_path / "langchain-store")

        mocks = _make_langchain_mocks()
        mod = _import_langchain_adapter(mocks)
        HumanMessage = mocks["langchain_core.messages"].HumanMessage
        AIMessage = mocks["langchain_core.messages"].AIMessage

        with mod.TRWChatMessageHistory(
            session_id="persist",
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        ) as history:
            history.add_messages([HumanMessage("hello"), AIMessage("world")])

        reopened = mod.TRWChatMessageHistory(
            session_id="persist",
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        )
        try:
            assert [message.content for message in reopened.messages] == ["hello", "world"]
        finally:
            reopened.close()

    def test_llamaindex_owned_yaml_backend_persists_across_reopen(
        self,
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """IT-06: LlamaIndex adapter-owned YAML backend survives reopen."""
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        storage_path = str(tmp_path / "llamaindex-store")

        mocks = _make_llamaindex_mocks()
        mod = _import_llamaindex_adapter(mocks)
        ChatMessage = mocks["llama_index.core.llms"].ChatMessage
        MessageRole = mocks["llama_index.core.llms"].MessageRole

        with mod.TRWChatStore(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        ) as store:
            store.add_message("conv-1", ChatMessage(role=MessageRole.USER, content="hello"))
            store.add_message("conv-1", ChatMessage(role=MessageRole.ASSISTANT, content="world"))

        reopened = mod.TRWChatStore(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        )
        try:
            messages = reopened.get_messages("conv-1")
            assert [message.content for message in messages] == ["hello", "world"]
            assert reopened.get_keys() == ["conv-1"]
        finally:
            reopened.close()

    def test_crewai_owned_yaml_backend_persists_filter_and_reset_across_reopen(
        self,
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """IT-07: CrewAI adapter-owned YAML backend persists and resets cleanly."""
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
        storage_path = str(tmp_path / "crewai-store")

        mocks = _make_crewai_mocks()
        mod = _import_crewai_adapter(mocks)

        with mod.TRWCrewStorage(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        ) as storage:
            storage.save("auth finding", metadata={"team": "auth"})
            storage.save("billing finding", metadata={"team": "billing"})

        reopened = mod.TRWCrewStorage(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        )
        try:
            filtered = reopened.search("finding", filter={"team": "auth"})
            assert [result["context"] for result in filtered] == ["auth finding"]

            reopened.reset()
            assert reopened.search("finding") == []
        finally:
            reopened.close()
