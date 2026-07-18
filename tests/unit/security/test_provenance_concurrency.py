"""Concurrent provenance writers must preserve one linear hash chain."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trw_memory.security import provenance
from trw_memory.security.provenance import ProvenanceEntry


def _entry(index: int) -> ProvenanceEntry:
    return ProvenanceEntry(
        learning_id=f"L-{index:03d}",
        content_hash=f"hash-{index}",
        source_identity="concurrent-test",
    )


@pytest.mark.parametrize("signed", [False, True])
def test_concurrent_appends_preserve_linear_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signed: bool,
) -> None:
    chain = tmp_path / "chain.jsonl"
    real_read_last = provenance._read_last

    def slow_read_last(path: Path) -> ProvenanceEntry | None:
        result = real_read_last(path)
        time.sleep(0.002)
        return result

    monkeypatch.setattr(provenance, "_read_last", slow_read_last)
    signing_key = None
    if signed:
        signing_key = pytest.importorskip("nacl.signing").SigningKey.generate()

    def write(index: int) -> None:
        if signing_key is None:
            provenance.append(chain, _entry(index))
        else:
            provenance.append_signed(chain, _entry(index), signing_key)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write, range(60)))

    assert len(chain.read_text(encoding="utf-8").splitlines()) == 60
    if signing_key is None:
        assert provenance.verify(chain) is True
    else:
        assert provenance.verify_signed(chain, signing_key.verify_key) is None
