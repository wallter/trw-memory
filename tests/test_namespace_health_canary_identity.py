"""UF-MEM-15: namespace health excludes the store's canaries by identity, not by a caller-writable flag.

``memory_status`` counted rows by ``metadata['system_canary'] == 'true'`` while the store probe used the pinned
``classify_canary`` (id and content), so a row carrying the flag could hide itself from the count. Both now use the
same classifier.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_canary import CANARY_NAMESPACE, _seeded_canary
from trw_memory.security.canary import PINNED_HASHES
from trw_memory.storage._namespace_health import namespace_health
from trw_memory.storage.sqlite_backend import SQLiteBackend

_NS = "project:health-demo"


def _backend(tmp_path: Path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "memory.db", dim=8)


def test_a_row_carrying_the_canary_flag_still_counts(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        backend.store(MemoryEntry(id="M-plain", content="an ordinary lesson", namespace=_NS))
        backend.store(
            MemoryEntry(
                id="M-forged", content="a lesson that tries to hide", namespace=_NS, metadata={"system_canary": "true"}
            )
        )

        health = namespace_health(backend, _NS, MemoryConfig())
    finally:
        backend.close()

    assert health["entries"] == 2, "a flag alone must not remove a row from the namespace's census"
    assert health["types"] == {"pattern": 2}


def test_the_stores_own_canary_is_still_excluded(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        canary_id = sorted(PINNED_HASHES)[0]
        backend.store(_seeded_canary(canary_id))
        backend.store(MemoryEntry(id="M-real", content="real knowledge", namespace=CANARY_NAMESPACE))

        health = namespace_health(backend, CANARY_NAMESPACE, MemoryConfig())
    finally:
        backend.close()

    assert health["entries"] == 1, health


def test_a_pinned_id_with_other_content_is_data_not_a_canary(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        canary_id = sorted(PINNED_HASHES)[0]
        backend.store(
            MemoryEntry(
                id=canary_id,
                content="not the pinned content",
                namespace=CANARY_NAMESPACE,
                metadata={"system_canary": "true"},
            )
        )

        health = namespace_health(backend, CANARY_NAMESPACE, MemoryConfig())
    finally:
        backend.close()

    assert health["entries"] == 1, "a pinned id carrying other content is user data (fail closed)"
