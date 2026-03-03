"""LangChain integration — ``BaseChatMessageHistory`` adapter.

Stores chat messages as :class:`~trw_memory.models.memory.MemoryEntry` objects
in the trw-memory storage backend, enabling persistent conversation history
across sessions.

Usage::

    from trw_memory.integrations.langchain import TRWChatMessageHistory

    history = TRWChatMessageHistory(session_id="session-1")
    history.add_messages([HumanMessage("hello"), AIMessage("hi")])
    print(history.messages)  # [HumanMessage("hello"), AIMessage("hi")]

Requires ``langchain-core >= 0.3.0``::

    pip install "trw-memory[langchain]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, Sequence

try:
    from langchain_core.chat_history import BaseChatMessageHistory  # type: ignore[import-not-found]
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "langchain-core is required for the LangChain adapter. "
        'Install it with: pip install "trw-memory[langchain]"'
    ) from exc

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

_TAG_PREFIX = "lc:session:"


class TRWChatMessageHistory(BaseChatMessageHistory):  # type: ignore[misc]
    """Persistent chat message history backed by trw-memory.

    Each message is stored as a :class:`MemoryEntry` with tags encoding
    the session key and message role.  Messages are retrieved in
    chronological order (by ``created_at``).

    Args:
        session_id: Unique conversation identifier used to scope messages.
        namespace: trw-memory namespace for storage isolation.
        storage_path: Override for the storage directory.
        backend: Pre-existing backend (for testing).  If ``None``, one is
            created from *namespace* and *storage_path*.
    """

    def __init__(
        self,
        session_id: str,
        *,
        namespace: str = "default",
        storage_path: str | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        from trw_memory.integrations._backend import resolve_backend

        self._session_id = session_id
        self._namespace = namespace
        self._session_tag = f"{_TAG_PREFIX}{session_id}"
        self._backend, self._owns_backend = resolve_backend(
            namespace, storage_path, backend,
        )

    @property
    def session_id(self) -> str:
        """The session identifier for this history."""
        return self._session_id

    # -- BaseChatMessageHistory interface ------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages for this session in chronological order."""
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT, ROLE_TAG_PREFIX

        entries = self._backend.list_entries(
            namespace=self._namespace,
            limit=DEFAULT_LIST_LIMIT,
        )
        session_entries = [
            e for e in entries if self._session_tag in e.tags
        ]
        session_entries.sort(key=lambda e: e.created_at)

        result: list[BaseMessage] = []
        for entry in session_entries:
            role = "human"
            for tag in entry.tags:
                if tag.startswith(ROLE_TAG_PREFIX):
                    role = tag[len(ROLE_TAG_PREFIX):]
                    break
            if role == "ai":
                result.append(AIMessage(content=entry.content))
            else:
                result.append(HumanMessage(content=entry.content))
        return result

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Store messages in trw-memory."""
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX, make_entry

        for msg in messages:
            role = getattr(msg, "type", "human")
            entry = make_entry(
                content=str(msg.content),
                namespace=self._namespace,
                tags=[self._session_tag, f"{ROLE_TAG_PREFIX}{role}"],
                importance=0.5,
                source="agent",
            )
            self._backend.store(entry)

    def clear(self) -> None:
        """Remove all messages for this session."""
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT

        entries = self._backend.list_entries(
            namespace=self._namespace,
            limit=DEFAULT_LIST_LIMIT,
        )
        for entry in entries:
            if self._session_tag in entry.tags:
                self._backend.delete(entry.id)

    # -- Resource management ------------------------------------------------

    def close(self) -> None:
        """Release backend resources if this instance owns them."""
        if self._owns_backend:
            self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
