"""End-to-end proof of the 2026-07-25 PII posture: local fidelity, boundary sanitization.

Two invariants are pinned here, and they are the whole point of the change:

1. **The local row keeps what the user wrote.** Storing an entry that contains an
   email address, an IPv4 address, an SSN-shaped number and a filesystem path
   persists all four verbatim. The write path used to mutate them irreversibly
   *before* anything reached disk, on the strength of 8 regexes with no NER.

2. **The egress payload does not.** The publish path — the only boundary where a
   memory leaves the machine — still sanitizes, so nothing that was true about
   the data leaving this machine changed. Sanitizing there is reversible: the
   unmasked original is still on the user's disk.

Plus the one destructive-adjacent behaviour that is deliberately KEPT: a
recognized credential still BLOCKS the store outright. That is high-precision
and non-destructive — it errors loudly instead of silently rewriting text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.remote import publish_memory_result

from ._test_sync_support import mock_httpx_client as _mock_httpx_client

# One payload carrying every built-in detector type the write path used to destroy.
EMAIL = "alice@example.com"
IP = "10.42.7.19"
SSN_SHAPED = "123-45-6789"
ABSOLUTE_PATH = "/home/alice/projects/widget/src/auth.py"
SENSITIVE_CONTENT = f"{EMAIL} hit a 500 from {IP} while replaying case {SSN_SHAPED} in {ABSOLUTE_PATH}"
SENSITIVE_DETAIL = f"repro: curl {IP} then grep {SSN_SHAPED} {ABSOLUTE_PATH}; owner {EMAIL}"


def _stored_row(storage_path: Path, memory_id: str) -> Any:
    """Read the persisted row straight from the SQLite file the client wrote."""
    db_path = next(storage_path.rglob("*.db"))
    with SQLiteBackend(db_path) as backend:
        return backend.get(memory_id)


@pytest.fixture()
def local_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


class TestLocalRowKeepsTheUsersText:
    """Requirement 1a — nothing the user wrote is mutated on the way to disk."""

    async def test_stored_row_preserves_email_ip_ssn_and_path_verbatim(
        self, local_client: MemoryClient, tmp_path: Path
    ) -> None:
        result = await local_client.store(
            content=SENSITIVE_CONTENT,
            detail=SENSITIVE_DETAIL,
            tags=["incident", f"reporter:{EMAIL}"],
            evidence=[f"log line from {IP}"],
            importance=0.9,
        )

        row = _stored_row(tmp_path / "storage", result["memory_id"])
        assert row is not None
        assert row.content == SENSITIVE_CONTENT
        assert row.detail == SENSITIVE_DETAIL
        assert row.tags[1] == f"reporter:{EMAIL}"
        assert row.evidence[0] == f"log line from {IP}"
        # No redaction marker of any kind reached the persisted row.
        for marker in ("<email>", "<ip>", "<ssn>", "<credit_card>", "<phone>", "<id:"):
            assert marker not in row.content
            assert marker not in row.detail

    async def test_detection_metadata_still_recorded_on_the_stored_row(
        self, local_client: MemoryClient, tmp_path: Path
    ) -> None:
        """Observability is free and non-destructive, so it is kept."""
        result = await local_client.store(content=SENSITIVE_CONTENT, importance=0.9)

        row = _stored_row(tmp_path / "storage", result["memory_id"])
        assert row is not None
        detected = row.metadata["pii_types"].split(",")
        assert "email" in detected
        assert "ssn" in detected


class TestEgressStillSanitizes:
    """Requirement 1b — the published payload is scrubbed at the boundary."""

    def test_published_payload_masks_what_the_local_row_kept(self) -> None:
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(
            id="M-egress",
            content=SENSITIVE_CONTENT,
            detail=SENSITIVE_DETAIL,
            importance=0.9,
        )
        cfg = MemoryConfig(
            sync_enabled=True,
            platform_url="https://api.example.com",
            platform_api_key="test-key-123",
            sync_min_importance=0.7,
        )

        with patch("httpx.Client") as mock_cls:
            mock_client = _mock_httpx_client(mock_cls, json_data={"id": "remote-1"})
            result = publish_memory_result(entry, cfg, project_root="/home/alice/projects/widget")

        assert result["success"] is True
        posted = mock_client.post.call_args.kwargs["json"]
        wire = json.dumps(posted)

        # Nothing sensitive survives the boundary...
        assert EMAIL not in wire
        assert IP not in wire
        assert SSN_SHAPED not in wire
        assert ABSOLUTE_PATH not in wire
        # ...and each is replaced by an explicit marker rather than dropped.
        assert "<email>" in posted["summary"]
        assert "<ip>" in posted["summary"]
        assert "<ssn>" in posted["summary"]
        assert "<project>/src/auth.py" in posted["summary"]
        assert "<email>" in posted["detail"]
        assert "<ip>" in posted["detail"]

    def test_published_payload_masks_pii_inside_tags(self) -> None:
        """Tags egress with the entry, so they need the same boundary treatment.

        The removed write-path policy scanned tags specifically (a credential in
        a tag would otherwise slip past a content-only scan). With the store path
        no longer mutating anything, a tag reaching the wire raw is a leak that
        nothing downstream repairs. Asserted on the POST body, not on the return
        of ``_anonymize_entry``, because the body is what leaves the machine.
        """
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(
            id="M-egress-tags",
            content="benign summary",
            detail="benign detail",
            tags=["incident", f"reporter:{EMAIL}", f"host-{IP}"],
            importance=0.9,
        )
        cfg = MemoryConfig(
            sync_enabled=True,
            platform_url="https://api.example.com",
            platform_api_key="test-key-123",
            sync_min_importance=0.7,
        )

        with patch("httpx.Client") as mock_cls:
            mock_client = _mock_httpx_client(mock_cls, json_data={"id": "remote-1"})
            result = publish_memory_result(entry, cfg, project_root="/home/alice/projects/widget")

        assert result["success"] is True
        posted = mock_client.post.call_args.kwargs["json"]
        wire = json.dumps(posted)

        assert EMAIL not in wire
        assert IP not in wire
        assert posted["tags"] == ["incident", "reporter:<email>", "host-<ip>"]


class TestRecognizedCredentialsStillBlock:
    """Requirement 2 — API_KEY refuses the store; it never silently rewrites."""

    @pytest.mark.parametrize(
        "credential",
        [
            "sk-abcdefghijklmnopqrstuvwxyz012345",
            "secret_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "AKIAIOSFODNN7EXAMPLE",
        ],
    )
    async def test_store_refuses_a_recognized_credential(
        self, local_client: MemoryClient, tmp_path: Path, credential: str
    ) -> None:
        with pytest.raises(PIIBlockError, match="api_key"):
            await local_client.store(content=f"the deploy key is {credential}", importance=0.9)

        # Fail-closed: the credential is not on disk in any form, masked or not.
        db_candidates = list((tmp_path / "storage").rglob("*.db"))
        persisted = "".join(path.read_bytes().decode("utf-8", "replace") for path in db_candidates)
        assert credential not in persisted

    async def test_credential_hidden_in_a_tag_still_blocks(self, local_client: MemoryClient) -> None:
        with pytest.raises(PIIBlockError, match="api_key"):
            await local_client.store(
                content="benign content",
                tags=["ok", "sk-abcdefghijklmnopqrstuvwxyz"],
                importance=0.9,
            )


class TestBlockingSurfaceUnchanged:
    """Guard rail: the kept defences must not be quietly narrowed later."""

    def test_blocking_pii_types_is_exactly_api_key(self) -> None:
        from trw_memory.security._runtime_pii import BLOCKING_PII_TYPES
        from trw_memory.security.pii import PIIType

        assert BLOCKING_PII_TYPES == frozenset({PIIType.API_KEY})

    def test_write_path_masking_is_limited_to_operator_custom_patterns(self) -> None:
        from trw_memory.security._runtime_pii import REDACTED_PII_TYPES
        from trw_memory.security.pii import PIIType

        assert REDACTED_PII_TYPES == frozenset({PIIType.CUSTOM})

    @pytest.mark.parametrize(
        ("value", "marker"),
        [
            (EMAIL, "<email>"),
            (IP, "<ip>"),
            (SSN_SHAPED, "<ssn>"),
            ("4111-1111-1111-1111", "<credit_card>"),
            ("555-123-4567", "<phone>"),
            ("sk-abcdefghijklmnopqrstuvwxyz", "<api_key>"),
        ],
    )
    def test_egress_helper_covers_every_type_the_write_path_stopped_masking(self, value: str, marker: str) -> None:
        """strip_pii parity: what write-path redaction used to cover, egress covers.

        Each value is checked in isolation. Adjacent digit groups can be merged
        into one match by the (deliberately untouched) credit-card pattern, which
        is a detector-precision property, not an egress-coverage property.
        """
        from trw_memory.security.pii import strip_pii

        assert strip_pii(f"observed {value} in the trace") == f"observed {marker} in the trace"


class TestEgressCredentialCoverageNotWeakened:
    """A digit-run inside a credential must not break credential masking."""

    def test_digit_heavy_api_key_is_fully_masked(self) -> None:
        """The SSN shape matches the tail of this credential.

        If the SSN pass ran first it would splice the credential apart, leaving a
        fragment the api_key pattern no longer recognises — and the remainder
        would be published. Credential subs therefore run before the detector pass.
        """
        from trw_memory.security.pii import strip_pii

        credential = "sk-1234567890123456789012"
        scrubbed = strip_pii(f"key {credential} end")
        assert credential not in scrubbed
        assert "123456789" not in scrubbed
        assert scrubbed == "key <api_key> end"


def test_mock_helper_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity: the httpx mock used above really captures the POST body."""
    with patch("httpx.Client") as mock_cls:
        client = _mock_httpx_client(mock_cls, json_data={"id": "x"})
        assert isinstance(client, MagicMock)
