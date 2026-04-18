"""FR-003 — Hash-chained provenance log for learning origins.

Sprint 96 W1-E scaffolding. Distinct from :mod:`trw_memory.security.audit`
(a generic audit log): this module specifically tracks *learning origins*
(who wrote what, when, and the chain of hashes linking successive writes).

On-disk format: JSON Lines. Each line is one :class:`ProvenanceEntry`.
The ``prev_hash`` of each record equals the SHA-256 of the canonical
JSON of the preceding record (``"GENESIS"`` for the first entry).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ProvenanceChain", "ProvenanceEntry", "append", "verify"]

_LOG = structlog.get_logger(__name__)
_GENESIS = "GENESIS"


class ProvenanceEntry(BaseModel):
    """A single record in the provenance chain."""

    model_config = ConfigDict(strict=True)

    learning_id: str
    content_hash: str
    prev_hash: str = _GENESIS
    source_identity: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProvenanceChain(BaseModel):
    """In-memory view of a provenance chain."""

    model_config = ConfigDict(strict=True)

    entries: list[ProvenanceEntry] = Field(default_factory=list)


def _canonical(entry: ProvenanceEntry) -> bytes:
    return json.dumps(entry.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(entry: ProvenanceEntry) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


def _read_last(chain_path: Path) -> ProvenanceEntry | None:
    if not chain_path.exists():
        return None
    last_line = ""
    with chain_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return ProvenanceEntry.model_validate_json(last_line)


def append(chain_path: Path, entry: ProvenanceEntry) -> str:
    """Atomically append *entry* to the chain at *chain_path*.

    The entry's ``prev_hash`` is rewritten to match the SHA-256 of the
    prior record (or ``"GENESIS"`` if this is the first). Returns the
    new chain head hash (``sha256(entry)``).
    """
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_last(chain_path)
    prev_hash = _hash(prior) if prior is not None else _GENESIS
    linked = entry.model_copy(update={"prev_hash": prev_hash})
    line = json.dumps(linked.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    # Atomic append via O_APPEND on POSIX; write+flush for Windows parity.
    with chain_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
    head = _hash(linked)
    _LOG.info(
        "provenance.append",
        learning_id=linked.learning_id,
        head=head,
        prev_hash=prev_hash,
    )
    return head


def verify(chain_path: Path) -> bool:
    """Walk the chain, recomputing each record's ``prev_hash`` link.

    Returns ``True`` if every link is consistent; ``False`` otherwise.
    Missing file returns ``True`` (empty chain is trivially valid).
    """
    if not chain_path.exists():
        return True
    expected_prev = _GENESIS
    with chain_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = ProvenanceEntry.model_validate_json(raw)
            except Exception:
                _LOG.warning("provenance.verify_parse_error", lineno=lineno)
                return False
            if entry.prev_hash != expected_prev:
                _LOG.warning(
                    "provenance.verify_chain_break",
                    lineno=lineno,
                    expected=expected_prev,
                    got=entry.prev_hash,
                )
                return False
            expected_prev = _hash(entry)
    return True
