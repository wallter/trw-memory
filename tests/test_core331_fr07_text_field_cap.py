"""PRD-CORE-331 FR07 / B71-94: the local store/update path caps content/detail like the daemon.

Before this change, ``security/poisoning.py::validate_store_inputs`` had no length cap on
``content``/``detail`` -- a local store or update could write a field of any size, while the
daemon's ``_arg_bounds.py::TEXT`` middleware caps the same fields at 64 KiB for every served
tool call. Verified empirically against archived commit 613d34db4 (this lane's integration
base): with ``max_entry_chars`` raised high enough to isolate the per-field check, a
65,537-character (64 KiB + 1) ``detail`` was accepted in full --
``{'status': 'stored', ...}`` with the stored row's ``detail`` measuring 65537 characters.

A field over the cap is now refused through the existing ``SchemaValidationError`` path
(``status: "invalid"``, the field named in ``failed_fields``), never truncated. The cap is
``trw_memory.models.memory.MAX_TEXT_FIELD_CHARS`` (65536 = 64 * 1024), the single source the
daemon's ``_arg_bounds.TEXT`` now imports instead of restating.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator

import pytest

from trw_memory.client import MemoryClient
from trw_memory.daemon._arg_bounds import TEXT as DAEMON_TEXT_BOUND
from trw_memory.exceptions import SchemaValidationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.correction import LearningPatch, parse_patch
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MAX_TEXT_FIELD_CHARS, MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.store import memory_store_impl
from trw_memory.tools.update import memory_update_impl

NAMESPACE = "project:default"
OVER_CAP = MAX_TEXT_FIELD_CHARS + 1
AT_CAP = MAX_TEXT_FIELD_CHARS


def test_shared_constant_not_two_numbers() -> None:
    """The daemon's per-field bound IS the local cap, not a second number that could drift."""
    assert MAX_TEXT_FIELD_CHARS == 64 * 1024
    assert DAEMON_TEXT_BOUND is MAX_TEXT_FIELD_CHARS


@pytest.fixture()
def store() -> Iterator[tuple[StorageBackend, MemoryConfig]]:
    with tempfile.TemporaryDirectory() as td:
        # Raised well above the field cap so the generic size-ceiling check
        # (poisoning.validate_entry_payload's max_chars) never masks the
        # field-specific cap under test.
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=td, max_entry_chars=10_000_000)
        with create_backend_from_config(cfg, NAMESPACE) as backend:
            backend.store(
                MemoryEntry(
                    id="L-1",
                    content="original content",
                    detail="original detail",
                    importance=0.4,
                    namespace=NAMESPACE,
                )
            )
            yield backend, cfg


class TestLocalStoreCap:
    def test_content_one_char_over_cap_is_refused_and_writes_nothing(
        self, store: tuple[StorageBackend, MemoryConfig]
    ) -> None:
        backend, cfg = store
        result = memory_store_impl(
            content="x" * OVER_CAP,
            namespace=NAMESPACE,
            backend=backend,
            config=cfg,
        )
        assert result["status"] == "invalid"
        assert "content" in result.get("failed_fields", []) or "content" in result.get("error", "")
        assert backend.get(result.get("memory_id", "M-does-not-exist"), namespace=NAMESPACE) is None

    def test_detail_one_char_over_cap_is_refused_and_writes_nothing(
        self, store: tuple[StorageBackend, MemoryConfig]
    ) -> None:
        backend, cfg = store
        entry_id = "M-fr07-detail-overcap"
        result = memory_store_impl(
            content="fits fine",
            detail="x" * OVER_CAP,
            namespace=NAMESPACE,
            backend=backend,
            config=cfg,
            entry_id=entry_id,
        )
        assert result["status"] == "invalid"
        assert "detail" in result.get("failed_fields", []) or "detail" in result.get("error", "")
        assert backend.get(entry_id, namespace=NAMESPACE) is None

    def test_content_exactly_at_cap_is_accepted(self, store: tuple[StorageBackend, MemoryConfig]) -> None:
        backend, cfg = store
        entry_id = "M-fr07-content-at-cap"
        result = memory_store_impl(
            content="y" * AT_CAP,
            namespace=NAMESPACE,
            backend=backend,
            config=cfg,
            entry_id=entry_id,
        )
        assert result["status"] == "stored"
        stored = backend.get(entry_id, namespace=NAMESPACE)
        assert stored is not None
        assert len(stored.content) == AT_CAP

    def test_detail_exactly_at_cap_is_accepted(self, store: tuple[StorageBackend, MemoryConfig]) -> None:
        backend, cfg = store
        entry_id = "M-fr07-detail-at-cap"
        result = memory_store_impl(
            content="fits fine",
            detail="z" * AT_CAP,
            namespace=NAMESPACE,
            backend=backend,
            config=cfg,
            entry_id=entry_id,
        )
        assert result["status"] == "stored"
        stored = backend.get(entry_id, namespace=NAMESPACE)
        assert stored is not None
        assert len(stored.detail) == AT_CAP


class TestLocalUpdateCap:
    def _update(self, store: tuple[StorageBackend, MemoryConfig], entry_id: str, **fields: object) -> dict[str, str]:
        backend, cfg = store
        parsed = parse_patch(fields)
        if not isinstance(parsed, LearningPatch):
            return parsed
        return memory_update_impl(entry_id, parsed, NAMESPACE, backend=backend, config=cfg)

    def test_detail_one_char_over_cap_on_update_is_refused_and_writes_nothing(
        self, store: tuple[StorageBackend, MemoryConfig]
    ) -> None:
        backend, _ = store
        result = self._update(store, "L-1", detail="x" * OVER_CAP)
        assert result["status"] == "invalid"
        entry = backend.get("L-1", namespace=NAMESPACE)
        assert entry is not None
        assert entry.detail == "original detail"

    def test_summary_one_char_over_cap_on_update_is_refused_and_writes_nothing(
        self, store: tuple[StorageBackend, MemoryConfig]
    ) -> None:
        backend, _ = store
        result = self._update(store, "L-1", summary="x" * OVER_CAP)
        assert result["status"] == "invalid"
        entry = backend.get("L-1", namespace=NAMESPACE)
        assert entry is not None
        assert entry.content == "original content"

    def test_detail_exactly_at_cap_on_update_is_accepted(self, store: tuple[StorageBackend, MemoryConfig]) -> None:
        backend, _ = store
        result = self._update(store, "L-1", detail="z" * AT_CAP)
        assert result["status"] == "updated"
        entry = backend.get("L-1", namespace=NAMESPACE)
        assert entry is not None
        assert len(entry.detail) == AT_CAP

    def test_a_long_value_cannot_be_patched_in_later_even_when_the_row_was_created_small(
        self, store: tuple[StorageBackend, MemoryConfig]
    ) -> None:
        """The original row was created well under the cap; patching detail past it is still refused."""
        backend, _ = store
        result = self._update(store, "L-1", detail="q" * OVER_CAP)
        assert result["status"] == "invalid"
        assert backend.get("L-1", namespace=NAMESPACE).detail == "original detail"  # type: ignore[union-attr]


class TestSdkClientCap:
    """MemoryClient.store is the SDK's served local entrypoint for memory_store."""

    async def test_client_store_content_over_cap_raises_schema_validation_error(
        self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_MAX_ENTRY_CHARS", "10000000")
        client._config = MemoryConfig()
        with pytest.raises(SchemaValidationError, match="content"):
            await client.store("x" * OVER_CAP)

    async def test_client_store_detail_over_cap_raises_schema_validation_error(
        self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_MAX_ENTRY_CHARS", "10000000")
        client._config = MemoryConfig()
        with pytest.raises(SchemaValidationError, match="detail"):
            await client.store("fits fine", detail="x" * OVER_CAP)

    async def test_client_store_at_cap_succeeds(self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_MAX_ENTRY_CHARS", "10000000")
        client._config = MemoryConfig()
        result = await client.store("fits fine", detail="z" * AT_CAP)
        assert result["status"] == "stored"


class TestBulkStoreCap:
    """``store_many``/bulk store go through the same ``validate_store_inputs`` choke point."""

    async def test_bulk_store_over_cap_item_is_rejected_not_stored(
        self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_MAX_ENTRY_CHARS", "10000000")
        client._config = MemoryConfig()
        written = await client.store_many(
            [
                {"content": "fine entry one"},
                {"content": "fine entry two", "detail": "x" * OVER_CAP},
            ]
        )
        assert written == 1
        results = await client.recall("fine entry", limit=10)
        rows = results["results"] if isinstance(results, dict) else results
        detail_lengths = [len(row.get("detail") or "") for row in rows]
        assert all(length <= AT_CAP for length in detail_lengths)
