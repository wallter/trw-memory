# ruff: noqa: F401,F811
"""Round-trip integration tests for the shipped adapters."""

from __future__ import annotations

from typing import Any

from ._test_integrations_support import tmp_backend


class TestIntegration:
    """Integration tests with real storage backend."""

    def test_vscode_store_and_status(self, tmp_backend: Any) -> None:
        """IT-04: VSCode store + status round-trip."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        adapter.store_selection("use fixtures", "/test.py", ["testing"])

        status = adapter.get_status()
        assert status["entry_count"] == 1

    def test_vscode_owned_backend_persists_across_reopen(
        self,
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """IT-05: an adapter-owned backend survives close + reopen.

        The adapter-owned path (no ``backend=`` argument) resolves and opens its
        own store, so it is the one that can lose data on close. Injecting a
        backend, as IT-04 does, never exercises it.
        """
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        storage_path = str(tmp_path / "vscode-store")

        from trw_memory.integrations.vscode import LocalMemoryAdapter

        with LocalMemoryAdapter(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        ) as adapter:
            adapter.store_selection("retry backoff is capped at 30s", "src/app.py", ["note"])

        reopened = LocalMemoryAdapter(
            namespace="project:adapter-e2e",
            storage_path=storage_path,
        )
        try:
            assert reopened.get_status()["entry_count"] == 1
        finally:
            reopened.close()
