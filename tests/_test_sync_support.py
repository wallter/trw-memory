"""Shared helpers for split sync tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry


def make_sync_entry(
    entry_id: str = "M-test",
    content: str = "test content",
    detail: str = "test detail",
    importance: float = 0.8,
    tags: list[str] | None = None,
    vector_clock: dict[str, int] | None = None,
    merged_from: list[str] | None = None,
    outcome_history: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry for sync tests."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        importance=importance,
        tags=tags or [],
        vector_clock=vector_clock or {},
        merged_from=merged_from or [],
        outcome_history=outcome_history or [],
        metadata=metadata or {},
    )


def make_sync_config(
    sync_enabled: bool = True,
    platform_url: str = "https://api.example.com",
    platform_api_key: str = "test-key-123",
    sync_min_importance: float = 0.7,
    local_only: bool = False,
) -> MemoryConfig:
    """Create a MemoryConfig for sync tests."""
    return MemoryConfig(
        sync_enabled=sync_enabled,
        platform_url=platform_url,
        platform_api_key=platform_api_key,
        sync_min_importance=sync_min_importance,
        local_only=local_only,
    )


def mock_httpx_client(
    mock_client_cls: MagicMock,
    *,
    status_code: int = 200,
    json_data: Any = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Wire up an httpx.Client context-manager mock."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    if side_effect is not None:
        mock_client.post.side_effect = side_effect
        mock_client.patch.side_effect = side_effect
    else:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        if json_data is not None:
            mock_resp.json.return_value = json_data
        mock_client.post.return_value = mock_resp
        mock_client.patch.return_value = mock_resp

    mock_client_cls.return_value = mock_client
    return mock_client
