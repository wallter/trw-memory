"""A held write must never look like a stored one to a chat adapter's caller.

``guarded_store`` reports a quarantine in its RETURN VALUE (``stored=False,
quarantined=True``) rather than by raising — correct for a caller that can
surface the distinction, which ``vscode.LocalMemoryAdapter.store_selection``
does via ``status``.

The LangChain, CrewAI and LlamaIndex adapters all return ``None``, and all three
discarded that result. So once an operator promotes ``trust_scoring_mode`` past
``observe`` — the promotion ``security/CLAUDE.md`` documents as the planned next
step — a quarantined turn vanished from the transcript while the method returned
normally. Each of those three docstrings already named that exact failure as the
reason they raise ("a censored transcript indistinguishable from a complete
one"); nothing held them to it.

These tests hold them to it, and pin the VSCode adapter's different-but-correct
contract so the two are not accidentally unified later.
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
        assert backend.get("M-ok") is not None


class TestAdaptersDoNotSwallowAQuarantine:
    """Derivation: every adapter whose return channel cannot express "held" must
    route through the raising seam. Checked structurally so a NEW adapter written
    against ``guarded_store`` is caught rather than needing its own test."""

    #: Adapters returning ``None`` (or a dict a caller reads as success) from the
    #: write, so a quarantine has nowhere to go but an exception.
    NO_SIGNAL_CHANNEL = ("langchain.py", "crewai.py", "llamaindex.py")

    @pytest.mark.parametrize("module", NO_SIGNAL_CHANNEL)
    def test_adapter_uses_the_raising_seam(self, module: str) -> None:
        import trw_memory.integrations as integrations

        source = (pathlib.Path(integrations.__file__).parent / module).read_text(encoding="utf-8")
        assert "guarded_store_or_raise(" in source, f"{module} must route writes through the raising seam"
        # The bare call must not survive alongside it — that is the swallow.
        bare = [
            line
            for line in source.splitlines()
            if "guarded_store(" in line and "guarded_store_or_raise(" not in line and not line.lstrip().startswith("#")
        ]
        assert bare == [], f"{module} still calls guarded_store directly: {bare}"

    def test_vscode_keeps_its_reporting_contract(self) -> None:
        """The counter-example. VSCode CAN express "held" in its ``status`` field,
        so it correctly uses the non-raising form; unifying the two would lose a
        real distinction rather than fix one."""
        import trw_memory.integrations as integrations

        source = (pathlib.Path(integrations.__file__).parent / "vscode.py").read_text(encoding="utf-8")
        assert "result = guarded_store(" in source
        assert "quarantined" in source
