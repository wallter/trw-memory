"""Unit tests for trw_memory.security.canary (PRD-SEC-001 FR-004, FR-007, FR-009)."""

from __future__ import annotations

from trw_memory.security.canary import (
    PINNED_HASHES,
    seed_canaries,
    verify_canaries,
)


class FakeStore:
    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    def seed(self, canary_id: str, content: str) -> None:
        self._rows[canary_id] = content

    def read(self, canary_id: str) -> str | None:
        return self._rows.get(canary_id)


def test_pinned_hashes_frozen_at_boot() -> None:
    # MappingProxyType is read-only.
    assert len(PINNED_HASHES) == 10
    import pytest

    with pytest.raises(TypeError):
        PINNED_HASHES["canary-001"] = "tampered"  # type: ignore[index]


def test_seed_canaries_inserts_expected_count() -> None:
    store = FakeStore()
    seeded = seed_canaries(store, count=5)
    assert len(seeded) == 5
    for canary in seeded:
        assert canary.expected_hash == PINNED_HASHES[canary.canary_id]


def test_seed_default_count_10() -> None:
    store = FakeStore()
    seeded = seed_canaries(store)
    assert len(seeded) == 10


def test_verify_clean_store_all_ok() -> None:
    store = FakeStore()
    seed_canaries(store)
    result = verify_canaries(store)
    assert result.ok == 10
    assert result.tampered == []
    assert result.missing == []


def test_verify_detects_tampering() -> None:
    store = FakeStore()
    seed_canaries(store)
    # Tamper with canary-003 content
    store._rows["canary-003"] = "tampered content"
    result = verify_canaries(store)
    assert "canary-003" in result.tampered
    assert result.ok == 9


def test_verify_detects_missing() -> None:
    store = FakeStore()
    seed_canaries(store, count=5)
    result = verify_canaries(store)
    # Canaries 6-10 were not seeded
    assert len(result.missing) == 5
    assert result.ok == 5


def test_seed_clamps_negative_count() -> None:
    store = FakeStore()
    seeded = seed_canaries(store, count=-1)
    assert seeded == []


def test_seed_clamps_overcount() -> None:
    store = FakeStore()
    seeded = seed_canaries(store, count=100)
    assert len(seeded) == 10
