"""Regression: canary state must be keyed per (quarantine, backend) pair.

A concurrent multi-backend audit uncovered ``CanaryTamperError`` raised
on every recall when a sweep iterated over multiple configs whose
quarantine DBs were the same but whose memory backends differed
(e.g. ``project_a/memory.db``, ``project_b/memory.db``, …). Root cause:
``CANARY_STATE`` keyed only on quarantine path made config-2's
``initialize_canaries`` short-circuit on "already seeded", but config-2's
backend never received the canaries, so the next ``probe_canaries``
raised.

These tests pin the per-backend isolation the fix introduces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import CanaryTamperError
from trw_memory.models.config import MemoryConfig
from trw_memory.security._runtime_canary import CANARY_STATE
from trw_memory.security.runtime import (
    initialize_canaries,
    probe_canaries,
    should_halt_recalls,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture(autouse=True)
def _reset_canary_state() -> None:
    CANARY_STATE.clear()
    yield
    CANARY_STATE.clear()


def test_initialize_canaries_seeds_each_backend_under_shared_quarantine(
    tmp_path: Path,
) -> None:
    """Two backends sharing a quarantine path must each get canaries seeded."""
    config = MemoryConfig(
        storage_path=str(tmp_path / "storage"),
        canary_probe_interval=1,
    )
    backend_a_path = tmp_path / "backend-a" / "memory.db"
    backend_b_path = tmp_path / "backend-b" / "memory.db"

    with (
        SQLiteBackend(backend_a_path, dim=config.embedding_dim) as backend_a,
        SQLiteBackend(backend_b_path, dim=config.embedding_dim) as backend_b,
    ):
        initialize_canaries(config, backend=backend_a)
        initialize_canaries(config, backend=backend_b)
        assert backend_a.get("canary-001") is not None, "backend A missing canary"
        assert backend_b.get("canary-001") is not None, "backend B canary not seeded — state-key collision regressed"


def test_probe_canaries_per_backend_does_not_cross_pollute(tmp_path: Path) -> None:
    """A canary tamper on backend A must not raise on backend B's recall path."""
    config = MemoryConfig(
        storage_path=str(tmp_path / "storage"),
        canary_probe_interval=1,
    )
    backend_a_path = tmp_path / "backend-a" / "memory.db"
    backend_b_path = tmp_path / "backend-b" / "memory.db"

    with (
        SQLiteBackend(backend_a_path, dim=config.embedding_dim) as backend_a,
        SQLiteBackend(backend_b_path, dim=config.embedding_dim) as backend_b,
    ):
        initialize_canaries(config, backend=backend_a)
        initialize_canaries(config, backend=backend_b)

        canary_a = backend_a.get("canary-001")
        assert canary_a is not None
        backend_a.store(canary_a.model_copy(update={"content": "tampered"}))

        with pytest.raises(CanaryTamperError):
            probe_canaries(config, backend=backend_a)

        probe_canaries(config, backend=backend_b)


def test_should_halt_recalls_per_backend(tmp_path: Path) -> None:
    """halt-mode tamper on backend A must not halt backend B."""
    config = MemoryConfig(
        storage_path=str(tmp_path / "storage"),
        canary_probe_interval=1,
        canary_fail_mode="halt",
    )
    backend_a_path = tmp_path / "backend-a" / "memory.db"
    backend_b_path = tmp_path / "backend-b" / "memory.db"

    with (
        SQLiteBackend(backend_a_path, dim=config.embedding_dim) as backend_a,
        SQLiteBackend(backend_b_path, dim=config.embedding_dim) as backend_b,
    ):
        initialize_canaries(config, backend=backend_a)
        initialize_canaries(config, backend=backend_b)

        canary_a = backend_a.get("canary-001")
        assert canary_a is not None
        backend_a.store(canary_a.model_copy(update={"content": "tampered"}))
        with pytest.raises(CanaryTamperError):
            probe_canaries(config, backend=backend_a)

        assert should_halt_recalls(config, backend=backend_a) is True
        assert should_halt_recalls(config, backend=backend_b) is False
