"""Unit tests for Ed25519 provenance signing (PRD-SEC-001 FR-002).

Sprint-96 carry-forward-b. Covers :func:`append_signed` and
:func:`verify_signed` including PyNaCl-unavailable degrade.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from trw_memory.security import provenance as prov_mod
from trw_memory.security.provenance import (
    ProvenanceEntry,
    append_signed,
    build_entry_provenance,
    derive_verify_key,
    verify_entry_provenance,
    verify_signed,
)

nacl_signing = pytest.importorskip("nacl.signing")
SigningKey = nacl_signing.SigningKey


def _entry(i: int) -> ProvenanceEntry:
    return ProvenanceEntry(
        learning_id=f"L-{i:03d}",
        content_hash=f"hash-{i}",
        source_identity="agent-1",
    )


def test_append_signed_writes_signature(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    key = SigningKey.generate()
    head = append_signed(chain, _entry(1), key)
    assert chain.exists()
    assert len(head) == 64
    # Read the record back and check the signature is a 128-char hex string
    raw = chain.read_text().strip()
    stored = ProvenanceEntry.model_validate_json(raw)
    assert stored.signature
    assert len(stored.signature) == 128  # 64-byte signature in hex
    bytes.fromhex(stored.signature)  # parses


def test_verify_signed_passes_clean_chain(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    key = SigningKey.generate()
    append_signed(chain, _entry(1), key)
    append_signed(chain, _entry(2), key)
    append_signed(chain, _entry(3), key)
    assert verify_signed(chain, key.verify_key) is None


def test_verify_signed_returns_id_of_tampered_content(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    key = SigningKey.generate()
    append_signed(chain, _entry(1), key)
    append_signed(chain, _entry(2), key)
    append_signed(chain, _entry(3), key)
    # Tamper with record 2's content_hash -- signature won't match
    lines = chain.read_text().splitlines()
    lines[1] = lines[1].replace('"hash-2"', '"hash-X"')
    chain.write_text("\n".join(lines) + "\n")
    broken = verify_signed(chain, key.verify_key)
    # Hash-chain link breaks first (record 2's recomputed hash differs from
    # record 3's prev_hash) OR signature breaks on record 2. Either way
    # we must identify record L-002 as the earliest broken link.
    assert broken == "L-002"


def test_verify_signed_detects_wrong_key(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    key = SigningKey.generate()
    wrong_key = SigningKey.generate()
    append_signed(chain, _entry(1), key)
    append_signed(chain, _entry(2), key)
    broken = verify_signed(chain, wrong_key.verify_key)
    assert broken == "L-001"


def test_verify_signed_missing_file_returns_none(tmp_path: Path) -> None:
    key = SigningKey.generate()
    assert verify_signed(tmp_path / "nope.jsonl", key.verify_key) is None


def test_verify_signed_with_unsigned_entry_fails(tmp_path: Path) -> None:
    """If a caller expected signed verification but entries have no signature, fail."""
    from trw_memory.security.provenance import append

    chain = tmp_path / "chain.jsonl"
    key = SigningKey.generate()
    append(chain, _entry(1))  # unsigned
    broken = verify_signed(chain, key.verify_key)
    assert broken == "L-001"


def test_append_signed_degrades_when_pynacl_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If PyNaCl flag is false, the entry is still written, just unsigned."""
    monkeypatch.setattr(prov_mod, "_NACL_AVAILABLE", False)
    chain = tmp_path / "chain.jsonl"
    # Pass a None signing key -- module must not raise AttributeError
    head = append_signed(chain, _entry(1), None)
    assert chain.exists()
    assert len(head) == 64
    stored = ProvenanceEntry.model_validate_json(chain.read_text().strip())
    assert stored.signature == ""


def test_module_reload_preserves_api() -> None:
    """Guard against accidental API breakage in future refactors."""
    reloaded = importlib.reload(prov_mod)
    assert hasattr(reloaded, "append_signed")
    assert hasattr(reloaded, "verify_signed")


def test_verify_entry_provenance_requires_real_verify_key() -> None:
    key = SigningKey.generate()
    metadata = build_entry_provenance(
        learning_id="L-row-001",
        content="safe",
        detail="detail",
        author="agent-1",
        session_id="sess-1",
        ts="2026-04-24T00:00:00+00:00",
        signing_key=key,
    )
    entry = {
        "learning_id": "L-row-001",
        "_content_for_verify": "safedetail",
        **metadata,
    }

    assert verify_entry_provenance(entry, derive_verify_key(key)) is True
    assert verify_entry_provenance(entry, derive_verify_key(SigningKey.generate())) is False
    assert verify_entry_provenance(entry, None) is False
