"""Shared helpers for the ``test_keys_*`` test family."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Literal

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.keys import clear_key_cache

_KEY_LENGTH = 32


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


def _make_config(
    key_source: Literal["keyring", "env", "file"] = "env",
    key_file_path: str = "~/.trw-memory/master.key",
    auto_generate_key: bool = True,
    *,
    encryption_enabled: bool = True,
) -> MemoryConfig:
    return MemoryConfig(
        encryption_enabled=encryption_enabled,
        key_source=key_source,
        key_file_path=key_file_path,
        auto_generate_key=auto_generate_key,
    )


def _make_entry(entry_id: str = "k-test-1", content: str = "content") -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="detail text",
        namespace="default",
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
