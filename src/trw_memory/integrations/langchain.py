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

from collections.abc import Sequence
from typing import TYPE_CHECKING

try:
    from langchain_core.chat_history import BaseChatMessageHistory  # type: ignore[import-not-found]
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        'langchain-core is required for the LangChain adapter. Install it with: pip install "trw-memory[langchain]"'
    ) from exc

from trw_memory.integrations._mixin import BackendOwnerMixin

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

_TAG_PREFIX = "lc:session:"


class TRWChatMessageHistory(BackendOwnerMixin, BaseChatMessageHistory):  # type: ignore[misc]
    """Persistent chat message history backed by trw-memory.

    Each message is stored as a :class:`MemoryEntry` with tags encoding
    the session key and message role.  Messages are retrieved in
    chronological order (by ``created_at``).

    Args:
        session_id: Unique conversation identifier used to scope messages.
        namespace: trw-memory namespace for storage isolation.
        max_results: Maximum number of most-recent messages returned from
            ``messages``.
        storage_path: Override for the storage directory.
        backend: Pre-existing backend (for testing).  If ``None``, one is
            created from *namespace* and *storage_path*.
    """

    def __init__(
        self,
        session_id: str,
        *,
        namespace: str = "default",
        max_results: int = 20,
        storage_path: str | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        from trw_memory.integrations._backend import config_for_storage_path, resolve_backend

        self._session_id = session_id
        self._namespace = namespace
        self._max_results = max_results
        self._session_tag = f"{_TAG_PREFIX}{session_id}"
        self._config = config_for_storage_path(storage_path)
        self._backend, self._owns_backend = resolve_backend(
            namespace,
            storage_path,
            backend,
        )

    @property
    def session_id(self) -> str:
        """The session identifier for this history."""
        return self._session_id

    # -- BaseChatMessageHistory interface ------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages for this session in chronological order."""
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX

        entries = self._list_namespace_entries()
        session_entries = [e for e in entries if self._session_tag in e.tags]
        session_entries.sort(key=lambda e: e.created_at)
        if self._max_results > 0:
            session_entries = session_entries[-self._max_results :]

        result: list[BaseMessage] = []
        for entry in session_entries:
            role = "human"
            for tag in entry.tags:
                if tag.startswith(ROLE_TAG_PREFIX):
                    role = tag[len(ROLE_TAG_PREFIX) :]
                    break
            if role == "ai":
                result.append(AIMessage(content=entry.content))
            else:
                result.append(HumanMessage(content=entry.content))
        return result

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Store messages in trw-memory.

        Every message goes through :func:`guarded_store_or_raise`, so a jailbroken
        model echoing an injection payload back into the transcript is rejected
        here rather than replayed verbatim on every later ``messages`` read. The
        rejection RAISES: a chat history that silently dropped a turn would
        leave the caller unable to tell a censored transcript from a complete
        one, which is the more dangerous failure for a conversational contract.
        Messages earlier in the batch stay stored, matching the existing
        partial-failure behaviour of a mid-batch backend error.

        A QUARANTINE decision raises too, via the ``_or_raise`` seam. Until
        2026-07-30 this method called ``guarded_store`` and discarded its result,
        so a held turn was dropped while ``add_messages`` returned normally —
        precisely the silent-drop this docstring already said was unacceptable.
        The gate reports a quarantine in its return value rather than by raising,
        and ``None`` is not a channel that can carry it.
        """
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX, make_entry
        from trw_memory.security.write_gate import guarded_store_or_raise

        for msg in messages:
            role = getattr(msg, "type", "human")
            entry = make_entry(
                content=str(msg.content),
                namespace=self._namespace,
                tags=[self._session_tag, f"{ROLE_TAG_PREFIX}{role}"],
                importance=0.5,
                source="agent",
            )
            guarded_store_or_raise(self._backend, entry, config=self._config)

    def clear(self) -> None:
        """Remove all messages for this session."""
        entries = self._list_namespace_entries()
        for entry in entries:
            if self._session_tag in entry.tags:
                self._backend.delete(entry.id)

    def _list_namespace_entries(self) -> list[MemoryEntry]:
        """Return a full namespace snapshot for adapter-local filtering.

        The adapter tags multiple conversations into one namespace, so a fixed
        list window can hide older entries and make session-scoped reads or
        clears incomplete once the namespace grows.
        """
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT

        namespace_count = self._backend.count(namespace=self._namespace)
        return self._backend.list_entries(
            namespace=self._namespace,
            limit=max(DEFAULT_LIST_LIMIT, namespace_count),
        )

    # Resource management inherited from BackendOwnerMixin.
