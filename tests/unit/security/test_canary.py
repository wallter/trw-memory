"""Unit tests for trw_memory.security.canary (PRD-SEC-001 FR-004, FR-007, FR-009)."""

from __future__ import annotations

from trw_memory.security.canary import PINNED_HASHES


def test_pinned_hashes_frozen_at_boot() -> None:
    # MappingProxyType is read-only.
    assert len(PINNED_HASHES) == 10
    import pytest

    with pytest.raises(TypeError):
        PINNED_HASHES["canary-001"] = "tampered"  # type: ignore[index]
