"""Simulate a store swapped after ``connect_registered`` pinned it (PRD-SEC-016).

A real swap is only detectable on Linux (the pin), so tests that exercise what a
caller does with the refusal fake one: once the pre-connect pin on *path* is
taken, every later identity lookup of *path* answers a different file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest


def swap_after_pin(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    import trw_memory._live_stores as live_stores

    real_pin = live_stores.pinned_identity
    real_identity = live_stores.current_identity
    pinned: set[Path] = set()

    @contextmanager
    def _pin(target: Path | str, **kwargs: object) -> Iterator[tuple[int, int] | None]:
        with real_pin(target, **kwargs) as identity:  # type: ignore[arg-type]
            pinned.add(Path(target))
            yield identity

    def _identity(target: Path | str, **kwargs: object) -> tuple[int, int] | None:
        if Path(target) == path and path in pinned:
            return (999999, 999999)  # the identity of a file that is not the one pinned
        return real_identity(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_stores, "pinned_identity", _pin)
    monkeypatch.setattr(live_stores, "current_identity", _identity)
