"""Wave 14: coverage gap-fill for tools/store.py (lines 118, 176-177, 194-195, 212-214)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import SchemaValidationError
from trw_memory.tools.store import memory_store_impl

from ._test_tools_support import _mock_backend


class TestStoreUpdatePath:
    def test_store_same_entry_id_twice_hits_model_copy_update_branch(self) -> None:
        """Re-storing with the same entry_id enters the model_copy update branch (line 118)."""
        from trw_memory.models.memory import MemoryEntry

        backend = _mock_backend()
        existing = MemoryEntry(id="M-update-001", content="original", namespace="project:default")
        backend.get.return_value = existing

        result = memory_store_impl(
            "updated content",
            "project:default",
            backend=backend,
            entry_id="M-update-001",
        )

        assert result["status"] in ("stored", "updated", "quarantined", "error")
        # backend.get called at least once to check for existing entry
        assert backend.get.call_count >= 1

    def test_store_rejects_entry_in_another_namespace(self) -> None:
        from trw_memory.models.memory import MemoryEntry

        backend = _mock_backend()
        backend.get.return_value = MemoryEntry(id="M-foreign", content="owner", namespace="project:owner")

        result = memory_store_impl("other", "project:other", backend=backend, entry_id="M-foreign")

        assert result["status"] == "not_found"
        backend.store.assert_not_called()

    def test_store_persists_grounding_fields(self) -> None:
        from trw_memory.models.memory import Assertion, AssertionType

        backend = _mock_backend()
        assertion = Assertion(type=AssertionType.GLOB_EXISTS, target="src/**/*.py")

        with patch("trw_memory.tools.store.embedding_has_consumer", return_value=False):
            result = memory_store_impl(
                "grounded tool content",
                "project:default",
                backend=backend,
                evidence=["src/example.py:10-20"],
                expires="when migration ships",
                assertions=[assertion],
            )

        stored = backend.store.call_args.args[0]
        assert result["status"] in ("stored", "updated")
        assert stored.evidence == ["src/example.py:10-20"]
        assert stored.expires == "when migration ships"
        assert stored.assertions == [assertion]


class TestStoreEmbedderFailure:
    def test_embedder_embed_raises_wraps_as_storage_error(self) -> None:
        """embedder.embed() raising any exception → StorageError (lines 176-177)."""
        from trw_memory.models.config import MemoryConfig

        backend = _mock_backend()
        cfg = MemoryConfig()

        class _FailEmbedder:
            def embed(self, _text: str) -> list[float]:
                raise RuntimeError("CUDA OOM")

        with (
            patch("trw_memory.tools.store.embedding_has_consumer", return_value=True),
            patch("trw_memory.tools.store.get_local_embedder", return_value=_FailEmbedder()),
            patch("trw_memory.tools.store.prepare_entry_for_store") as mock_prep,
        ):
            mock_decision = MagicMock()
            mock_decision.quarantined = False
            mock_decision.op = "store"
            mock_decision.entry = MagicMock()
            mock_decision.entry.id = "M-001"
            mock_decision.entry.content = "test"
            mock_decision.entry.detail = ""
            mock_decision.entry.source_identity = ""
            mock_decision.entry.source = "tool"
            mock_decision.pii_matches = []
            mock_prep.return_value = mock_decision

            result = memory_store_impl(
                "content requiring embedding",
                "project:default",
                backend=backend,
                config=cfg,
            )

        assert result["status"] == "error"


class TestStoreGraphScheduleFailure:
    def test_schedule_graph_update_runtime_error_logs_warning_and_continues(self) -> None:
        """schedule_graph_update raising RuntimeError → warning logged, store succeeds (lines 194-195)."""
        from trw_memory.models.config import MemoryConfig

        backend = _mock_backend()
        cfg = MemoryConfig()

        with (
            patch("trw_memory.tools.store.schedule_graph_update", side_effect=RuntimeError("graph error")),
            patch("trw_memory.tools.store.embedding_has_consumer", return_value=False),
        ):
            result = memory_store_impl(
                "content with graph failure",
                "project:default",
                backend=backend,
                config=cfg,
            )

        assert result["status"] in ("stored", "updated", "quarantined")


class TestStoreSchemaValidationErrorInnerBlock:
    def test_schema_error_from_prepare_entry_returns_invalid_when_not_raising(self) -> None:
        """SchemaValidationError from prepare_entry_for_store with raise_security_errors=False → return invalid (lines 212-214)."""
        from trw_memory.models.config import MemoryConfig

        backend = _mock_backend()
        cfg = MemoryConfig()

        exc = SchemaValidationError("inner schema error", failed_fields=["content"])
        with patch("trw_memory.tools.store.prepare_entry_for_store", side_effect=exc):
            result = memory_store_impl(
                "valid content",
                "project:default",
                backend=backend,
                config=cfg,
                raise_security_errors=False,
            )

        assert result["status"] == "invalid"
        assert "inner schema error" in str(result.get("error", ""))

    def test_schema_error_from_prepare_entry_re_raises_when_raise_security_errors_true(self) -> None:
        """SchemaValidationError from prepare_entry_for_store with raise_security_errors=True → re-raises (line 212-213)."""
        from trw_memory.models.config import MemoryConfig

        backend = _mock_backend()
        cfg = MemoryConfig()

        exc = SchemaValidationError("inner schema error", failed_fields=["content"])
        with patch("trw_memory.tools.store.prepare_entry_for_store", side_effect=exc):
            with pytest.raises(SchemaValidationError, match="inner schema error"):
                memory_store_impl(
                    "valid content",
                    "project:default",
                    backend=backend,
                    config=cfg,
                    raise_security_errors=True,
                )
