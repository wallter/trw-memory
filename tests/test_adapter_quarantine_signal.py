"""A held write must never look like a stored one to a caller.

``guarded_store`` reports a quarantine in its RETURN VALUE (``stored=False,
quarantined=True``) rather than by raising, and ``guarded_store_or_raise``
turns that into ``MemoryQuarantinedError`` for a caller that cannot itself
surface the distinction.

Historical context: this was written for a set of chat-framework adapters
(LangChain, CrewAI, LlamaIndex, and later a VSCode adapter) that called the
non-raising ``guarded_store`` and discarded its return value, so a quarantined
turn silently vanished from the transcript instead of raising. All of those
adapters have since been removed as unused surface (zero production callers);
the tests below now exercise the shared ``guarded_store_or_raise`` seam
directly.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import pytest

from trw_memory.exceptions import MemoryQuarantinedError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.write_gate import GuardedStoreResult, guarded_store_or_raise
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture()
def gate_config() -> MemoryConfig:
    tmp = pathlib.Path(tempfile.mkdtemp())
    return MemoryConfig(
        audit_log_path=str(tmp / "audit.jsonl"),
        rate_limit_state_path=str(tmp / "rate.yaml"),
    )


@pytest.fixture()
def backend() -> SQLiteBackend:
    return SQLiteBackend(pathlib.Path(tempfile.mkdtemp()) / "m.db")


def _force_quarantine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the intake pipeline return a quarantine decision.

    Patched at the ``write_gate`` seam rather than by constructing a genuinely
    anomalous corpus: the behaviour under test is what ``guarded_store``'s CALLER
    does with a quarantine verdict, not how the verdict is reached. The real
    quarantine path itself is covered by ``test_poisoning_runtime.py``.
    """
    import trw_memory.security.write_gate as write_gate

    def _quarantined(entry: MemoryEntry, **_: Any) -> Any:
        from trw_memory.security.runtime import PreparedStoreEntry

        return PreparedStoreEntry(
            entry=entry,
            op="store",
            pii_matches=(),
            quarantined=True,
            anomaly_dimension="trust_score",
            anomaly_z_score=0.25,
        )

    monkeypatch.setattr(write_gate, "prepare_entry_for_store", _quarantined)
    monkeypatch.setattr(write_gate, "store_quarantined_entry", lambda *a, **k: None)


class TestGuardedStoreOrRaise:
    """The shared seam, so three adapters need not hand-roll the same check."""

    def test_quarantine_raises_with_the_entry_and_dimension(
        self, backend: SQLiteBackend, gate_config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_quarantine(monkeypatch)
        entry = MemoryEntry(id="M-held", content="a routine note")

        with pytest.raises(MemoryQuarantinedError) as excinfo:
            guarded_store_or_raise(backend, entry, config=gate_config)

        # The caller needs enough to point an operator at the review store.
        assert excinfo.value.entry_id == "M-held"
        assert excinfo.value.anomaly_dimension == "trust_score"

    def test_quarantine_is_not_reported_as_poisoning(
        self, backend: SQLiteBackend, gate_config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A held entry is durable and may be approved — calling it a rejection
        would misstate what happened to the caller's data."""
        from trw_memory.exceptions import PoisoningError

        _force_quarantine(monkeypatch)
        with pytest.raises(MemoryQuarantinedError) as excinfo:
            guarded_store_or_raise(backend, MemoryEntry(id="M-held", content="note"), config=gate_config)
        assert not isinstance(excinfo.value, PoisoningError)

    def test_a_clean_write_returns_normally(self, backend: SQLiteBackend, gate_config: MemoryConfig) -> None:
        """Control: the seam must not turn ordinary writes into errors."""
        result = guarded_store_or_raise(backend, MemoryEntry(id="M-ok", content="a routine note"), config=gate_config)
        assert isinstance(result, GuardedStoreResult)
        assert result.stored is True
        assert backend.get("M-ok", namespace="default") is not None
