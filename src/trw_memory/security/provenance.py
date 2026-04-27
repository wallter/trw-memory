"""FR-002/FR-003 — Hash-chained provenance log for learning origins.

Sprint 96 W1-E scaffolding. Distinct from :mod:`trw_memory.security.audit`
(a generic audit log): this module specifically tracks *learning origins*
(who wrote what, when, and the chain of hashes linking successive writes).

On-disk format: JSON Lines. Each line is one :class:`ProvenanceEntry`.
The ``prev_hash`` of each record equals the SHA-256 of the canonical
JSON of the preceding record (``"GENESIS"`` for the first entry).

FR-002 (Sprint-96 carry-forward-b): entries MAY additionally carry an
Ed25519 ``signature`` over ``learning_id + content_hash + prev_hash``.
:func:`append_signed` writes a signed record; :func:`verify_signed`
walks the chain and returns the first broken link (or ``None``).
Signatures are optional — legacy SHA-256-only chains verify unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

try:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import SigningKey, VerifyKey

    _NACL_AVAILABLE = True
except ImportError:  # pragma: no cover — PyNaCl is optional
    _NACL_AVAILABLE = False
    SigningKey = Any  # type: ignore[misc,assignment]
    VerifyKey = Any  # type: ignore[misc,assignment]
    BadSignatureError = Exception  # type: ignore[misc,assignment]

__all__ = [
    "ProvenanceChain",
    "ProvenanceEntry",
    "append",
    "append_signed",
    "build_entry_provenance",
    "derive_verify_key",
    "verify",
    "verify_entry_provenance",
    "verify_signed",
]

_LOG = structlog.get_logger(__name__)
_GENESIS = "GENESIS"


class ProvenanceEntry(BaseModel):
    """A single record in the provenance chain.

    ``signature`` is populated only when the entry was written via
    :func:`append_signed`. Hex-encoded Ed25519 signature over the
    concatenation ``learning_id + content_hash + prev_hash``.
    """

    model_config = ConfigDict(strict=True)

    learning_id: str
    content_hash: str
    prev_hash: str = _GENESIS
    source_identity: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""


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


def _sign_message(learning_id: str, content_hash: str, prev_hash: str) -> bytes:
    """Canonical signing payload for an entry."""
    return f"{learning_id}|{content_hash}|{prev_hash}".encode()


def build_entry_provenance(
    *,
    learning_id: str,
    content: str,
    detail: str,
    author: str,
    session_id: str,
    ts: str,
    signing_key: Any,
) -> dict[str, str]:
    """Build signed per-row provenance metadata for a memory entry."""
    content_hash = hashlib.sha256(f"{content}{detail}".encode()).hexdigest()
    payload = f"{learning_id}|{author}|{session_id}|{ts}|{content_hash}".encode()
    signature = ""
    if signing_key is not None:
        signed = signing_key.sign(payload)
        signature_bytes = signed.signature if hasattr(signed, "signature") else signed
        signature = bytes(signature_bytes).hex()
    return {
        "provenance_author": author,
        "provenance_session_id": session_id,
        "provenance_ts": ts,
        "provenance_content_hash": content_hash,
        "provenance_signature": signature,
    }


def derive_verify_key(signing_key: Any | None) -> Any | None:
    """Derive a public verify key from a loaded signing key object."""
    if signing_key is None:
        return None
    verify_key = getattr(signing_key, "verify_key", None)
    if verify_key is not None:
        return verify_key
    public_key = getattr(signing_key, "public_key", None)
    if callable(public_key):
        return public_key()
    return None


def verify_entry_provenance(entry: dict[str, str] | Any, verify_key: Any | None) -> bool:
    """Return True when per-row provenance metadata matches the current content."""
    metadata = entry if isinstance(entry, dict) else dict(getattr(entry, "metadata", {}))
    content = metadata.get("_content_for_verify", "") if isinstance(entry, dict) else f"{entry.content}{entry.detail}"
    content_hash = metadata.get("provenance_content_hash", "")
    signature = metadata.get("provenance_signature", "")
    author = metadata.get("provenance_author", "")
    session_id = metadata.get("provenance_session_id", "")
    ts = metadata.get("provenance_ts", "")
    if not content_hash or not author or not session_id or not ts:
        return False
    current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if current_hash != content_hash:
        return False
    if not signature:
        return False
    if verify_key is None:
        return False
    payload = (
        f"{getattr(entry, 'id', metadata.get('learning_id', ''))}|{author}|{session_id}|{ts}|{content_hash}".encode()
    )
    try:
        signature_bytes = bytes.fromhex(signature)
    except ValueError:
        return False
    verify = getattr(verify_key, "verify", None)
    if not callable(verify):
        return False
    for args in ((payload, signature_bytes), (signature_bytes, payload)):
        try:
            verified = verify(*args)
        except Exception:  # noqa: S112 — try alternate arg ordering on next iteration
            continue
        if verified is False:
            continue
        return True
    return False


def append_signed(
    chain_path: Path,
    entry: ProvenanceEntry,
    signing_key: Any,
) -> str:
    """Append *entry* to the chain with an Ed25519 signature.

    The signature is taken over ``learning_id + content_hash + prev_hash``
    (pipe-joined). If PyNaCl is unavailable, the entry is appended
    via :func:`append` (degraded to SHA-256-only) and a warning is
    emitted.

    Returns the new chain head hash.
    """
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_last(chain_path)
    prev_hash = _hash(prior) if prior is not None else _GENESIS

    if not _NACL_AVAILABLE or signing_key is None:
        _LOG.warning(
            "provenance.append_signed_degraded",
            reason="pynacl_unavailable" if not _NACL_AVAILABLE else "no_signing_key",
            learning_id=entry.learning_id,
        )
        linked = entry.model_copy(update={"prev_hash": prev_hash, "signature": ""})
    else:
        msg = _sign_message(entry.learning_id, entry.content_hash, prev_hash)
        # nacl SigningKey.sign returns SignedMessage whose .signature is bytes
        sig_bytes: bytes = signing_key.sign(msg).signature
        linked = entry.model_copy(update={"prev_hash": prev_hash, "signature": sig_bytes.hex()})

    line = json.dumps(linked.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    with chain_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
    head = _hash(linked)
    _LOG.info(
        "provenance.append_signed",
        learning_id=linked.learning_id,
        head=head,
        prev_hash=prev_hash,
        signed=bool(linked.signature),
    )
    return head


def verify_signed(chain_path: Path, verify_key: Any) -> str | None:
    """Walk the chain; return the first broken link's ``learning_id``.

    Returns ``None`` if every entry's hash-link AND Ed25519 signature
    verify cleanly. Entries without a signature are treated as broken
    *only if* a verify_key was provided (signed chain was expected).
    A missing file returns ``None`` (empty chain is trivially valid).

    If PyNaCl is unavailable, this degrades to plain :func:`verify` and
    returns ``None`` on pass / the first broken ``learning_id`` on fail.
    """
    if not chain_path.exists():
        return None

    expected_prev = _GENESIS
    with chain_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = ProvenanceEntry.model_validate_json(raw)
            except Exception:
                _LOG.warning("provenance.verify_signed_parse_error", lineno=lineno)
                return f"lineno:{lineno}"

            if entry.prev_hash != expected_prev:
                _LOG.warning(
                    "provenance.verify_signed_chain_break",
                    lineno=lineno,
                    learning_id=entry.learning_id,
                )
                return entry.learning_id

            if _NACL_AVAILABLE and verify_key is not None:
                if not entry.signature:
                    _LOG.warning(
                        "provenance.verify_signed_missing_signature",
                        learning_id=entry.learning_id,
                    )
                    return entry.learning_id
                try:
                    sig = bytes.fromhex(entry.signature)
                    msg = _sign_message(entry.learning_id, entry.content_hash, entry.prev_hash)
                    verify_key.verify(msg, sig)
                except (BadSignatureError, ValueError):
                    _LOG.warning(
                        "provenance.verify_signed_bad_signature",
                        learning_id=entry.learning_id,
                    )
                    return entry.learning_id

            expected_prev = _hash(entry)
    return None
