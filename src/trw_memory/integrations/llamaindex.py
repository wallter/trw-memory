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

from typing import TYPE_CHECKING, Any

try:
    from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore[import-not-found]
    from llama_index.core.storage.chat_store.base import BaseChatStore  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "llama-index-core is required for the LlamaIndex adapter. "
        'Install it with: pip install "trw-memory[llamaindex]"'
    ) from exc

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

_TAG_PREFIX = "li:key:"


class TRWChatStore(BaseChatStore):  # type: ignore[misc]
    """Persistent chat store backed by trw-memory.

    Each message is stored as a :class:`MemoryEntry` tagged with the
    conversation key and message role.

    Args:
        namespace: trw-memory namespace for storage isolation.
        storage_path: Override for the storage directory.
        backend: Pre-existing backend (for testing).
    """

    namespace: str = "default"
    _storage_path: str | None = None
    _backend: Any = None
    _owns_backend: bool = False

    model_config: dict[str, Any] = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        namespace: str = "default",
        *,
        storage_path: str | None = None,
        backend: StorageBackend | None = None,
        **kwargs: Any,
    ) -> None:
        from trw_memory.integrations._backend import resolve_backend

        super().__init__(**kwargs)
        self.namespace = namespace
        self._storage_path = storage_path
        self._backend, self._owns_backend = resolve_backend(
            namespace, storage_path, backend,
        )

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
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT, ROLE_TAG_PREFIX

        key_tag = f"{_TAG_PREFIX}{key}"
        entries = self._backend.list_entries(
            namespace=self.namespace,
            limit=DEFAULT_LIST_LIMIT,
        )
        matched = [e for e in entries if key_tag in e.tags]
        matched.sort(key=lambda e: e.created_at)

        result: list[ChatMessage] = []
        for entry in matched:
            role = MessageRole.USER
            for tag in entry.tags:
                if tag.startswith(ROLE_TAG_PREFIX):
                    role_str = tag[len(ROLE_TAG_PREFIX):]
                    try:
                        role = MessageRole(role_str)
                    except ValueError:
                        role = MessageRole.USER
                    break
            result.append(ChatMessage(role=role, content=entry.content))
        return result

    def add_message(
        self,
        key: str,
        message: ChatMessage,
        idx: int | None = None,
    ) -> None:
        """Append a message to *key*'s collection."""
        from trw_memory.integrations._backend import ROLE_TAG_PREFIX, make_entry

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
        self._backend.store(entry)

    def delete_messages(self, key: str) -> list[ChatMessage] | None:
        """Remove all messages for *key*.  Returns the deleted messages."""
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT, ROLE_TAG_PREFIX

        key_tag = f"{_TAG_PREFIX}{key}"
        entries = self._backend.list_entries(
            namespace=self.namespace,
            limit=DEFAULT_LIST_LIMIT,
        )
        deleted: list[ChatMessage] = []
        for entry in entries:
            if key_tag in entry.tags:
                role = MessageRole.USER
                for tag in entry.tags:
                    if tag.startswith(ROLE_TAG_PREFIX):
                        try:
                            role = MessageRole(tag[len(ROLE_TAG_PREFIX):])
                        except ValueError:
                            pass
                        break
                deleted.append(ChatMessage(role=role, content=entry.content))
                self._backend.delete(entry.id)
        return deleted or None

    def delete_message(self, key: str, idx: int) -> ChatMessage | None:
        """Remove the message at *idx* from *key*'s collection."""
        messages = self.get_messages(key)
        if idx < 0 or idx >= len(messages):
            return None
        removed = messages[idx]
        remaining = messages[:idx] + messages[idx + 1:]
        self.set_messages(key, remaining)
        return removed

    def delete_last_message(self, key: str) -> ChatMessage | None:
        """Remove the most recent message from *key*."""
        messages = self.get_messages(key)
        if not messages:
            return None
        return self.delete_message(key, len(messages) - 1)

    def get_keys(self) -> list[str]:
        """Return all existing conversation keys."""
        from trw_memory.integrations._backend import DEFAULT_LIST_LIMIT

        entries = self._backend.list_entries(
            namespace=self.namespace,
            limit=DEFAULT_LIST_LIMIT,
        )
        keys: set[str] = set()
        for entry in entries:
            for tag in entry.tags:
                if tag.startswith(_TAG_PREFIX):
                    keys.add(tag[len(_TAG_PREFIX):])
        return sorted(keys)

    # -- Resource management ------------------------------------------------

    def close(self) -> None:
        """Release backend resources if this instance owns them."""
        if self._owns_backend:
            self._backend.close()

    def __enter__(self) -> TRWChatStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
