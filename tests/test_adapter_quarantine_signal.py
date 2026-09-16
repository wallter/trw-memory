"""A held write must never look like a stored one to a chat adapter's caller.

``guarded_store`` reports a quarantine in its RETURN VALUE (``stored=False,
quarantined=True``) rather than by raising — correct for a caller that can
surface the distinction, which ``vscode.LocalMemoryAdapter.store_selection``
does via ``status``.

The three chat adapters this was written for (LangChain, CrewAI, LlamaIndex) all
returned ``None`` and discarded that result. So once an operator promotes
``trust_scoring_mode`` past ``observe`` — the promotion ``security/CLAUDE.md``
documents as the planned next step — a quarantined turn vanished from the
transcript while the method returned normally. Those three adapters have since
been removed as unused surface, so the structural check below is DERIVED from the
adapter modules that actually ship rather than naming them: a new adapter written
with the same defect is caught without editing this file.
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


def _adapter_sources() -> dict[str, str]:
    """Every shipped adapter module, keyed on file name.

    Discovered from the package directory, not listed here: a hand-written list
    is exactly the subset-registry defect this test exists to prevent, and it
    goes silently stale the moment an adapter is added or removed.
    """
    import trw_memory.integrations as integrations

    root = pathlib.Path(integrations.__file__).parent
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(root.glob("*.py"))
        if not path.name.startswith("_") and path.name != "factory.py"
    }


def _bare_guarded_store_lines(source: str) -> list[str]:
    """Lines calling the NON-raising ``guarded_store`` in *source*."""
    return [
        line
        for line in source.splitlines()
        if "guarded_store(" in line and "guarded_store_or_raise(" not in line and not line.lstrip().startswith("#")
    ]


class TestAdaptersDoNotSwallowAQuarantine:
    """Derivation: every adapter must either route writes through the raising
    seam or report the quarantine in its own return value. Checked structurally
    so a NEW adapter written against ``guarded_store`` is caught rather than
    needing its own test."""

    def test_every_adapter_can_report_a_held_write(self) -> None:
        sources = _adapter_sources()
        assert sources, "non-vacuity: no adapter modules were discovered"

        offenders = []
        for name, source in sources.items():
            if not _bare_guarded_store_lines(source):
                continue  # routes through the raising seam, or does not write
            if "quarantined" not in source:
                offenders.append(name)
        assert offenders == [], (
            "adapter(s) call the non-raising guarded_store without reading the "
            f"quarantine verdict, so a held write looks stored: {offenders}"
        )

    def test_the_check_sees_the_shipped_adapter(self) -> None:
        """Non-vacuity control: the discovery must actually find vscode.py.

        Without this, a glob that stopped matching would report zero offenders
        out of zero files — the "proved absence by not looking" failure.
        """
        sources = _adapter_sources()
        assert "vscode.py" in sources
        assert _bare_guarded_store_lines(sources["vscode.py"]), (
            "vscode.py no longer calls guarded_store, so this class no longer checks anything"
        )

    def test_vscode_keeps_its_reporting_contract(self) -> None:
        """The counter-example. VSCode CAN express "held" in its ``status`` field,
        so it correctly uses the non-raising form; unifying the two would lose a
        real distinction rather than fix one."""
        import trw_memory.integrations as integrations

        source = (pathlib.Path(integrations.__file__).parent / "vscode.py").read_text(encoding="utf-8")
        assert "result = guarded_store(" in source
        assert "quarantined" in source
