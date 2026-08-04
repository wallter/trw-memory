"""Secret shapes are handled by threat class, not by which regex happened to match.

Two defects motivate this file, both confirmed by execution on 2026-07-30.

**Provider keys were detected but mis-typed.** ``sk_live_…`` (Stripe) and
``sk-proj-…`` (OpenAI) scored above the Shannon-entropy backstop, so they were
detected — as ``HIGH_ENTROPY``. Only ``PIIType.API_KEY`` appears in
``_runtime_pii.BLOCKING_PII_TYPES``, and only the regex types are masked by
``strip_pii``. So a live Stripe key was persisted verbatim AND published verbatim
to the platform, while an AWS key in the identical position was blocked and
masked. ``sk`` was already in ``_SECRET_PREFIX_PATTERN`` — the pattern simply
could not span the ``live``/``proj`` segment every real provider key carries.

**The search body was never sanitized.** The 2026-07-25 egress pass (6cf5b97f29)
covered the publish direction only. ``fetch_shared_memories`` put the raw recall
query on the wire, and a recall query quotes whatever is broken.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import detect_pii, strip_pii
from trw_memory.security.write_gate import guarded_store
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync._remote_fetch import fetch_shared_memories

#: Canonical published shapes, one per provider. ``aws``/``github`` are the
#: controls that were already handled — they pin that this file's assertions are
#: about *parity*, not about the two shapes that always worked.
#:
#: Every value here is synthetic — invented for this test, matching no live
#: account. They are nonetheless assembled from a prefix and a body rather than
#: written as one literal, because **this package is published to a public
#: mirror and GitHub push protection scans the file text, not the runtime
#: value**. A contiguous ``sk_live_51…`` on disk matches Stripe's real key shape
#: closely enough to block the release push outright (it blocked trw-memory
#: 0.15.0 on 2026-08-04) with no way to fix it from the pushing side. The
#: concatenation is invisible at runtime, so the assertions below are unchanged;
#: ``github_pat`` has always been written this way for the same reason.
PROVIDER_SECRETS: dict[str, str] = {
    "stripe_live": "sk_" + "live_51H8xQzGvNqL2pRsT4uVwXyZaBcDeFgHi",
    "stripe_test": "sk_" + "test_51H8xQzGvNqL2pRsT4uVwXyZaBcDeFgHi",
    "stripe_restricted": "rk_" + "live_51H8xQzGvNqL2pRsT4uVwXyZaBcDeFgHi",
    "openai_project": "sk-" + "proj-abc123XYZdef456UVWghi789RSTjkl012MNO",
    "openai_classic": "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0",
    "slack_bot": "xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    "google_api": "AIza" + "SyD-1234567890abcdefghijklmnopqrstu",
    "aws_access_key": "AKIA" + "IOSFODNN7EXAMPLE",
    "github_pat": "ghp_" + "a" * 36,
}

#: Engineering prose that must survive untouched. Widening a credential regex is
#: only safe with a false-positive control beside it — each of these contains a
#: token from the prefix vocabulary in an ordinary sentence.
BENIGN_PROSE: tuple[str, ...] = (
    "token_for_the_admin_account is set in the env file",
    "the api_key config field is read at startup",
    "secret-rotation runbook lives in docs/deployment",
    "key_derivation uses HKDF with a 32 byte salt",
    "see docs/requirements-aare-f/prds/PRD-CORE-185-user-space-memory.md",
    "pk_test is the publishable key prefix documented by the provider",
)

#: Ordinary snake_case identifiers of the exact shape `<prefix>_<word>_<longword>`.
#: The first draft of the scope-segment widening used a generic
#: `[a-zA-Z0-9]{1,12}` for the middle, which is indistinguishable from one of
#: these — and `API_KEY` BLOCKS the write, so each of these lost the learning
#: outright rather than merely logging. The segment is now a closed set of real
#: provider scope words, which covers every shape above.
BENIGN_IDENTIFIERS: tuple[str, ...] = (
    "pk_users_organizationmembership",
    "key_error_troubleshootingnotes",
    "token_cache_invalidationstrategy",
    "secret_store_rotationprocedure",
    "api_gateway_requestthrottling",
    "sk_deployment_configurationvalues",
)


def _pii_types(text: str) -> set[str]:
    return {getattr(match.pii_type, "value", match.pii_type) for match in detect_pii(text)}


@pytest.fixture()
def gate_config() -> MemoryConfig:
    tmp = pathlib.Path(tempfile.mkdtemp())
    return MemoryConfig(
        audit_log_path=str(tmp / "audit.jsonl"),
        rate_limit_state_path=str(tmp / "rate.yaml"),
    )


class TestProviderSecretsAreClassifiedAsCredentials:
    """Classification is what decides blocking and masking, so pin it directly."""

    @pytest.mark.parametrize("name", sorted(PROVIDER_SECRETS))
    def test_shape_is_typed_api_key_not_high_entropy(self, name: str) -> None:
        """``HIGH_ENTROPY`` neither blocks nor masks — the type IS the behaviour."""
        assert "api_key" in _pii_types(PROVIDER_SECRETS[name])

    @pytest.mark.parametrize("name", sorted(PROVIDER_SECRETS))
    def test_shape_is_masked_by_strip_pii(self, name: str) -> None:
        secret = PROVIDER_SECRETS[name]
        assert secret not in strip_pii(f"the key {secret} was rotated")

    @pytest.mark.parametrize("name", sorted(PROVIDER_SECRETS))
    def test_shape_blocks_the_write(self, name: str, gate_config: MemoryConfig) -> None:
        """Attribution anchor: narrow the regex back and this goes red per shape."""
        backend = SQLiteBackend(pathlib.Path(tempfile.mkdtemp()) / "m.db")
        entry = MemoryEntry(id=f"M-{name}", content=f"deploy failed, key was {PROVIDER_SECRETS[name]}")
        with pytest.raises(PIIBlockError):
            guarded_store(backend, entry, config=gate_config)

    @pytest.mark.parametrize("prose", BENIGN_PROSE)
    def test_engineering_prose_is_untouched(self, prose: str) -> None:
        """The false-positive control that makes the widening defensible."""
        assert strip_pii(prose) == prose

    @pytest.mark.parametrize("prose", BENIGN_PROSE)
    def test_engineering_prose_still_stores(self, prose: str, gate_config: MemoryConfig) -> None:
        backend = SQLiteBackend(pathlib.Path(tempfile.mkdtemp()) / "m.db")
        result = guarded_store(backend, MemoryEntry(id="M-benign", content=prose), config=gate_config)
        assert result.stored is True

    @pytest.mark.parametrize("identifier", BENIGN_IDENTIFIERS)
    def test_snake_case_identifiers_are_not_credentials(self, identifier: str) -> None:
        """A generic scope segment cannot tell these apart from a provider key."""
        assert "api_key" not in _pii_types(identifier)

    @pytest.mark.parametrize("identifier", BENIGN_IDENTIFIERS)
    def test_snake_case_identifiers_still_store(self, identifier: str, gate_config: MemoryConfig) -> None:
        """API_KEY blocks the write, so a false positive here LOSES the learning."""
        backend = SQLiteBackend(pathlib.Path(tempfile.mkdtemp()) / "m.db")
        entry = MemoryEntry(id="M-ident", content=f"the {identifier} index is the one that got dropped")
        assert guarded_store(backend, entry, config=gate_config).stored is True


def _mock_client(mock_cls: MagicMock, *, status_code: int = 200, payload: object = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = [] if payload is None else payload
    client.post.return_value = response
    mock_cls.return_value.__enter__.return_value = client
    return client


SYNC_CONFIG = MemoryConfig(
    sync_enabled=True,
    platform_url="https://api.example.com",
    platform_api_key="test-key-123",
)


class TestSearchQueryIsSanitizedOnTheWire:
    """The fetch direction egresses caller text too — 6cf5b97f29 covered publish only."""

    def test_secret_in_the_query_never_reaches_the_request_body(self) -> None:
        secret = PROVIDER_SECRETS["stripe_live"]
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            client = _mock_client(mock_cls)
            fetch_shared_memories(f"why did {secret} start 401ing", SYNC_CONFIG)

        body = json.dumps(client.post.call_args.kwargs["json"])
        assert secret not in body
        assert "<api_key>" in body

    def test_email_in_the_query_never_reaches_the_request_body(self) -> None:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            client = _mock_client(mock_cls)
            fetch_shared_memories("ask alice@example.com about the retry budget", SYNC_CONFIG)

        body = json.dumps(client.post.call_args.kwargs["json"])
        assert "alice@example.com" not in body
        assert "<email>" in body

    @pytest.mark.parametrize(
        "query",
        [
            "how do I tune the recall dedup threshold",
            # A query is the caller's SEARCH INTENT, so the masking is deliberately
            # narrower than strip_pii, which also masks PHONE/SSN/CREDIT_CARD/IP.
            # Those detectors are shape-based and eat legitimate search terms: a
            # bare 10-digit epoch matches the PHONE shape, and an internal IP is a
            # perfectly good thing to search for. Masking them would leave no
            # remote hit possible. Publish-direction egress keeps full strip_pii —
            # a published learning is durable, a query is not.
            "scheduler stalled at 1753833600",
            "why does 10.0.0.5 refuse connections",
            "why does /home/alice/proj/src/auth.py fail",
        ],
    )
    def test_a_legitimate_search_term_is_transmitted_unchanged(self, query: str) -> None:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            client = _mock_client(mock_cls)
            fetch_shared_memories(query, SYNC_CONFIG)

        assert client.post.call_args.kwargs["json"]["query"] == query


class TestFailedFetchIsDistinguishableFromAnEmptyCorpus:
    """P5: a failed fetch and 'nothing matched' both return ``[]``.

    The caller merges either identically, so without a log an operator cannot tell
    a rotated API key from a quiet corpus.
    """

    def test_non_200_emits_a_warning_with_the_status(self) -> None:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _mock_client(mock_cls, status_code=401)
            with patch("trw_memory.sync._remote_fetch.logger") as mock_logger:
                assert fetch_shared_memories("anything", SYNC_CONFIG) == []

        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.kwargs["status_code"] == 401

    def test_malformed_body_emits_a_warning(self) -> None:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _mock_client(mock_cls, payload="not-a-container")
            with patch("trw_memory.sync._remote_fetch.logger") as mock_logger:
                assert fetch_shared_memories("anything", SYNC_CONFIG) == []

        mock_logger.warning.assert_called_once()

    def test_a_genuinely_empty_corpus_logs_nothing(self) -> None:
        """The control that makes the warning meaningful: no warning on success."""
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _mock_client(mock_cls, payload=[])
            with patch("trw_memory.sync._remote_fetch.logger") as mock_logger:
                assert fetch_shared_memories("anything", SYNC_CONFIG) == []

        mock_logger.warning.assert_not_called()
