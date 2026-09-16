"""Shared fixtures for the integration adapter tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture()
def tmp_backend(tmp_path: Any) -> Generator[Any, None, None]:
    """Create a temporary SQLite backend for testing."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "test" / "memory.db"
    backend = SQLiteBackend(db_path=db_path, dim=384)
    yield backend
    backend.close()
