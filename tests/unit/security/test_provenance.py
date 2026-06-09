"""Unit tests for trw_memory.security.provenance (PRD-SEC-001 FR-003)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.security.provenance import ProvenanceEntry, append, append_signed, verify


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


def test_append_on_corrupt_tail_fails_closed(tmp_path: Path) -> None:
    """A corrupt tail line must abort the write, not re-root the chain."""
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    with chain.open("a", encoding="utf-8") as fh:
        fh.write("{not valid provenance json\n")
    with pytest.raises(StorageError):
        append(chain, _entry(2))


def test_append_signed_on_corrupt_tail_fails_closed(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    with chain.open("a", encoding="utf-8") as fh:
        fh.write('{"learning_id": 123}\n')  # valid JSON, schema-invalid
    with pytest.raises(StorageError):
        append_signed(chain, _entry(2), signing_key=None)


def test_append_on_non_utf8_tail_fails_closed(tmp_path: Path) -> None:
    """Non-UTF-8 bytes must surface as a typed StorageError, not a raw decode crash."""
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    with chain.open("ab") as fh:
        fh.write(b"\xff\xfe not utf-8\n")
    with pytest.raises(StorageError):
        append(chain, _entry(2))


def test_corrupt_tail_error_is_content_free(tmp_path: Path) -> None:
    """The raised error must not leak the corrupt record's payload."""
    chain = tmp_path / "chain.jsonl"
    append(chain, _entry(1))
    secret = "L-SECRET-leaky-learning-id"
    with chain.open("a", encoding="utf-8") as fh:
        # Valid JSON, missing required fields -> pydantic ValidationError would
        # normally embed this content in its message.
        fh.write(f'{{"learning_id": "{secret}"}}\n')
    with pytest.raises(StorageError) as exc_info:
        append(chain, _entry(2))
    rendered = f"{exc_info.value}{exc_info.value.__cause__}"
    assert secret not in rendered
