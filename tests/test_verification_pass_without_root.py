"""A verification refresh with no project root observes nothing and changes nothing.

trw-mcp reads these rows at recall time and reports ``unknown`` for a never-observed
assertion and ``last_known_failure`` for a dated one (CORE268), so a root-less refresh
must neither invent an observation nor clear a recorded failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.lifecycle.verification_pass import run_maintain_verify
from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    store.store(
        MemoryEntry(
            id="L-test",
            content="test claim",
            assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="my_func", target="source.py")],
        )
    )
    yield store
    store.close()


def _refresh(backend: SQLiteBackend, root: Path | None) -> None:
    run_maintain_verify(
        backend,
        assertion_failure_penalty=0.15,
        assertion_stale_threshold_days=7,
        anchor_validity_verified_floor=0.8,
        batch_limit=1,
        project_root=root,
    )


def _assertions(backend: SQLiteBackend) -> list[dict[str, object]]:
    entry = backend.get("L-test", namespace="default")
    assert entry is not None
    return [assertion.model_dump(mode="json") for assertion in entry.assertions]


def test_a_refresh_without_a_root_records_no_observation(backend: SQLiteBackend) -> None:
    _refresh(backend, None)

    (assertion,) = _assertions(backend)
    assert assertion["last_result"] is None
    assert assertion["last_verified_at"] is None


def test_a_refresh_without_a_root_keeps_a_dated_failure(backend: SQLiteBackend, tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("missing = 1\n", encoding="utf-8")
    _refresh(backend, tmp_path)
    failed = _assertions(backend)
    assert failed[0]["last_result"] is False and failed[0]["first_failed_at"]

    _refresh(backend, None)

    assert _assertions(backend) == failed
