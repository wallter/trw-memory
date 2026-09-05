"""LlamaIndex integration — ``BaseChatStore`` adapter.

Stores chat messages as :class:`~trw_memory.models.memory.MemoryEntry` objects,
providing a persistent chat store usable with LlamaIndex's
``ChatMemoryBuffer``.

Usage::

    from trw_memory.integrations.llamaindex import TRWChatStore
    from llama_index.core.memory import ChatMemoryBuffer

    store = TRWChatStore(namespace="project:my-app")
    memory = ChatMemoryBuffer.from_defaults(chat_store=store, chat_store_key="user-1")

Requires ``llama-index-core >= 0.11.0``::

    pip install "trw-memory[llamaindex]"
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

try:
    from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore[import-not-found]
    from llama_index.core.storage.chat_store.base import BaseChatStore  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        'llama-index-core is required for the LlamaIndex adapter. Install it with: pip install "trw-memory[llamaindex]"'
    ) from exc

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

_TAG_PREFIX = "li:key:"


class TRWChatStore(BaseChatStore):  # type: ignore[misc]
    """Persistent chat store backed by trw-memory.

    Each message is stored as a :class:`MemoryEntry` tagged with the
    conversation key and message role.

    Args:
        namespace: trw-memory namespace for storage isolation.
        message_limit: Maximum number of most-recent messages returned from
            ``get_messages``.
        storage_path: Override for the storage directory.
        backend: Pre-existing backend (for testing).
    """

    namespace: str = "default"
    _storage_path: str | None = None
    _backend: StorageBackend | None = None
    _owns_backend: bool = False

    model_config: ClassVar[dict[str, object]] = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        namespace: str = "default",
        *,
        message_limit: int = 100,
        storage_path: str | None = None,
        backend: StorageBackend | None = None,
        **kwargs: object,
    ) -> None:
        from trw_memory.integrations._backend import resolve_backend

        super().__init__(**kwargs)
        self.namespace = namespace
        self._message_limit = message_limit
        self._storage_path = storage_path
        self._backend, self._owns_backend = resolve_backend(
            namespace,
            storage_path,
            backend,
        )

    @property
    def _backend_or_raise(self) -> StorageBackend:
        """Return the backend, raising if not initialised."""
        if self._backend is None:
            raise RuntimeError("TRWChatStore backend not initialised")
        return self._backend

    @classmethod
    def class_name(cls) -> str:
        """Return class name for LlamaIndex serialization."""
        return "TRWChatStore"

    # -- BaseChatStore abstract methods --------------------------------------

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Replace all messages for *key* with *messages*."""
        self.delete_messages(key)
        for msg in messages:
            self.add_message(key, msg)

    def get_messages(self, key: str) -> list[ChatMessage]:
        """Retrieve all messages for *key* in chronological order."""
        matched = self._get_key_entries(key)
        if self._message_limit > 0:
            matched = matched[-self._message_limit :]

        return [ChatMessage(role=self._message_role(entry), content=entry.content) for entry in matched]

    def add_message(
        self,
        key: str,
        message: ChatMessage,
        idx: int | None = None,
    ) -> None:
        """Append a message to *key*'s collection.

        Routed through :func:`guarded_store_or_raise` so an injection payload
        echoed by a jailbroken model is rejected at write time instead of replayed
        on every later ``get_messages``. The rejection RAISES for the same reason
        as the LangChain adapter: silently dropping a turn would make a censored
        transcript indistinguishable from a complete one. A QUARANTINE raises too;
        until 2026-07-30 this discarded the gate's result, so a held message was
        dropped while ``add_message`` returned normally.
        """
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX, config_for_storage_path, make_entry
        from trw_memory.security.write_gate import guarded_store_or_raise

        role_value = message.role.value if isinstance(message.role, MessageRole) else str(message.role)
        entry = make_entry(
            content=str(message.content),
            namespace=self.namespace,
            tags=[
                f"{_TAG_PREFIX}{key}",
                f"{ROLE_TAG_PREFIX}{role_value}",
            ],
            importance=0.5,
            source="agent",
        )
        guarded_store_or_raise(self._backend_or_raise, entry, config=config_for_storage_path(self._storage_path))

    def delete_messages(self, key: str) -> list[ChatMessage] | None:
        """Remove all messages for *key*.  Returns the deleted messages."""
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX

        entries = self._get_key_entries(key)
        deleted: list[ChatMessage] = []
        for entry in entries:
            role = MessageRole.USER
            for tag in entry.tags:
                if tag.startswith(ROLE_TAG_PREFIX):
                    with contextlib.suppress(ValueError):
                        role = MessageRole(tag[len(ROLE_TAG_PREFIX) :])
                    break
            deleted.append(ChatMessage(role=role, content=entry.content))
            self._backend_or_raise.delete(entry.id, namespace=entry.namespace)
        return deleted or None

    def delete_message(self, key: str, idx: int) -> ChatMessage | None:
        """Remove the message at *idx* from *key*'s collection."""
        entries = self._get_key_entries(key)
        if self._message_limit > 0:
            visible_entries = entries[-self._message_limit :]
        else:
            visible_entries = entries
        if idx < 0 or idx >= len(visible_entries):
            return None
        target = visible_entries[idx]
        removed = ChatMessage(role=self._message_role(target), content=target.content)
        # Delete the selected entry in place so bounded reads do not rewrite and
        # silently truncate older history outside the visible message window.
        self._backend_or_raise.delete(target.id, namespace=target.namespace)
        return removed

    def delete_last_message(self, key: str) -> ChatMessage | None:
        """Remove the most recent message from *key*."""
        messages = self.get_messages(key)
        if not messages:
            return None
        return self.delete_message(key, len(messages) - 1)

    def get_keys(self) -> list[str]:
        """Return all existing conversation keys."""
        entries = self._list_namespace_entries()
        keys: set[str] = set()
        for entry in entries:
            for tag in entry.tags:
                if tag.startswith(_TAG_PREFIX):
                    keys.add(tag[len(_TAG_PREFIX) :])
        return sorted(keys)

    def _get_key_entries(self, key: str) -> list[MemoryEntry]:
        """Return all stored entries for *key* in chronological order."""
        key_tag = f"{_TAG_PREFIX}{key}"
        entries = self._list_namespace_entries()
        matched = [entry for entry in entries if key_tag in entry.tags]
        matched.sort(key=lambda entry: entry.created_at)
        return matched

    def _list_namespace_entries(self) -> list[MemoryEntry]:
        """Return a full namespace snapshot for adapter-local filtering."""
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT

        namespace_count = self._backend_or_raise.count(namespace=self.namespace)
        return self._backend_or_raise.list_entries(
            namespace=self.namespace,
            limit=max(DEFAULT_LIST_LIMIT, namespace_count),
        )

    def _message_role(self, entry: MemoryEntry) -> MessageRole:
        """Recover a stored LlamaIndex role tag, defaulting safely to USER."""
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX

        role = MessageRole.USER
        for tag in entry.tags:
            if tag.startswith(ROLE_TAG_PREFIX):
                with contextlib.suppress(ValueError):
                    role = MessageRole(tag[len(ROLE_TAG_PREFIX) :])
                break
        return role

    # -- Resource management ------------------------------------------------

    def close(self) -> None:
        """Release backend resources if this instance owns them."""
        if self._owns_backend:
            self._backend_or_raise.close()

    def __enter__(self) -> TRWChatStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
