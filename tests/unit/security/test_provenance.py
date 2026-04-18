"""Unit tests for trw_memory.security.provenance (PRD-SEC-001 FR-003)."""

from __future__ import annotations

from pathlib import Path

from trw_memory.security.provenance import ProvenanceEntry, append, verify


def _entry(i: int) -> ProvenanceEntry:
    return ProvenanceEntry(
        learning_id=f"L-{i:03d}",
        content_hash=f"hash-{i}",
        source_identity="agent-1",
    )


def test_verify_on_missing_file_returns_true(tmp_path: Path) -> None:
    assert verify(tmp_path / "does_not_exist.jsonl") is True


def test_append_creates_genesis_entry(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    head = append(chain, _entry(1))
    assert chain.exists()
    assert len(head) == 64  # sha256 hex
    assert verify(chain) is True


def test_append_chains_multiple_entries(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    h1 = append(chain, _entry(1))
    h2 = append(chain, _entry(2))
    h3 = append(chain, _entry(3))
    assert len({h1, h2, h3}) == 3
    assert verify(chain) is True


def test_tampered_chain_fails_verify(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    append(chain, _entry(2))
    append(chain, _entry(3))
    # Corrupt the middle line: flip a char in content_hash
    lines = chain.read_text().splitlines()
    lines[1] = lines[1].replace("hash-2", "hash-X")
    chain.write_text("\n".join(lines) + "\n")
    assert verify(chain) is False


def test_malformed_json_fails_verify(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    with chain.open("a") as fh:
        fh.write("{not json\n")
    assert verify(chain) is False


def test_empty_lines_ignored(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    with chain.open("a") as fh:
        fh.write("\n\n")
    append(chain, _entry(2))
    assert verify(chain) is True
