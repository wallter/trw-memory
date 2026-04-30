"""Shared helpers for the ``test_yaml_field_parity*`` test family."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import Assertion, MemoryEntry
from trw_memory.storage.persistence import write_yaml
from trw_memory.storage.yaml_backend import YAMLBackend


@pytest.fixture()
def backend(tmp_path: Path) -> Iterator[YAMLBackend]:
    db = YAMLBackend(tmp_path / "entries")
    yield db


def make_entry(
    entry_id: str,
    content: str,
    *,
    vector_clock: dict[str, int] | None = None,
    remote_id: str | None = None,
    published_to_platform: bool = False,
    pending_delete: bool = False,
    cross_validated: bool = False,
    outcome_history: list[str] | None = None,
    assertions: list[Assertion] | None = None,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=now,
        updated_at=now,
        vector_clock=vector_clock or {},
        remote_id=remote_id,
        published_to_platform=published_to_platform,
        pending_delete=pending_delete,
        cross_validated=cross_validated,
        outcome_history=outcome_history or [],
        assertions=assertions or [],
    )


def write_entry_yaml(backend: YAMLBackend, entry_id: str, data: dict[str, object]) -> None:
    write_yaml(backend._dir / f"{entry_id}.yaml", data)
