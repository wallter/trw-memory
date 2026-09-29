"""A read of the warm tier never creates ``warm.db`` (or its directory); a write still does.

``get_embedding`` and a dense ``warm_search`` used to open the warm backend in create mode, so
recalling from a fresh store left ``memory/warm.db``, its WAL, shm and oplock behind.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.models.config import MemoryConfig

pytestmark = pytest.mark.unit

_VECTOR = [0.25] * 384


def _manager(root: Path) -> TierManager:
    return TierManager(root, MemoryConfig(storage_path=str(root)))


def _listing(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


_READS: dict[str, Callable[[TierManager], object]] = {
    "get_embedding": lambda m: m._warm_store.get_embedding("missing"),
    "warm_search keywords only": lambda m: m.warm_search(["anything"], None),
    "warm_search dense": lambda m: m.warm_search(["anything"], _VECTOR),
    "warm_remove of an absent entry": lambda m: m.warm_remove("missing"),
    "purge_sidecar_entry of an absent entry": lambda m: m._warm_store.purge_sidecar_entry("missing"),
}


@pytest.mark.parametrize("read", sorted(_READS))
def test_a_read_on_a_fresh_store_leaves_no_new_files(tmp_path: Path, read: str) -> None:
    manager = _manager(tmp_path)
    before = _listing(tmp_path)

    _READS[read](manager)

    assert _listing(tmp_path) == before


def test_a_write_still_creates_the_warm_tier_and_a_later_read_opens_it(tmp_path: Path) -> None:
    writer = _manager(tmp_path)
    writer.warm_add("entry-1", {"content": "the warm tier mirrors this row"}, _VECTOR)
    assert (tmp_path / "memory" / "warm.db").is_file()
    writer.close()

    reader = _manager(tmp_path)  # a fresh manager: it must open the existing file without creating
    assert reader._warm_store.get_embedding("entry-1") is not None
    assert reader._warm_store.get_embedding("missing") is None
