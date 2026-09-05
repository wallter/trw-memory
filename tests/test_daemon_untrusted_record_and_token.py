"""A daemon secret that cannot be READ is never mistaken for one that is ABSENT.

Two bugs of one shape, PRD-CORE-253 FR03/FR08:

*Discovery* — ``read_discovery`` folded unreadable, malformed and
schema-mismatched records into ``None``, and the single-instance claim only ran
its liveness gate when the record parsed. So an invalid record walked straight
past the gate: a second daemon bound a fresh port and overwrote the record while
the first was still serving, leaving two writers on one ``memory.db``.

*Token* — ``read_secret_file`` returned ``None`` for a planted symlink, a
permission failure or non-UTF-8 bytes, and ``ensure_token`` reads ``None`` as
first run. It then ``os.replace``d the live daemon's token, so every subsequent
client call was rejected: the automatic rotation FR08 clause 3 forbids, reached
from the create path instead of the reject path.

Both are now three-valued, and the middle value refuses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trw_memory.daemon import (
    DaemonInfo,
    DaemonPaths,
    DiscoveryAbsent,
    DiscoveryInvalid,
    claim_single_instance,
    ensure_token,
    read_discovery,
    read_discovery_result,
)
from trw_memory.exceptions import (
    DaemonRecordInvalidError,
    TokenUnreadableError,
)

#: Corruptions an operator, a crash or a version skew can genuinely leave
#: behind. Each proves NOTHING about whether a daemon is serving the store.
_UNTRUSTED_RECORDS = [
    pytest.param("}{ not json at all", id="malformed"),
    pytest.param('["a", "list"]', id="not_an_object"),
    pytest.param(json.dumps({"schema_version": 99, "pid": 1}), id="future_schema"),
    pytest.param(json.dumps({"schema_version": 1, "pid": -4, "url": "u"}), id="field_invalid"),
]


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    resolved = DaemonPaths.resolve()
    resolved.user_memory_dir.mkdir(parents=True, exist_ok=True)
    return resolved


# ── W02: an untrusted discovery record ───────────────────────────────────────


@pytest.mark.parametrize("record", _UNTRUSTED_RECORDS)
def test_untrusted_record_reads_as_invalid_not_absent(paths: DaemonPaths, record: str) -> None:
    """The read says "I cannot tell", which is a different answer from "nobody"."""
    paths.discovery.write_text(record, encoding="utf-8")

    result = read_discovery_result(paths)

    assert isinstance(result, DiscoveryInvalid)
    assert result.path == paths.discovery
    assert result.reason, "an invalid record must carry an operator-readable reason"


def test_absent_record_reads_as_absent(paths: DaemonPaths) -> None:
    """The other two arms still resolve, so the union is not vacuously invalid."""
    assert isinstance(read_discovery_result(paths), DiscoveryAbsent)

    info = DaemonInfo(
        pid=os.getpid(),
        url="http://127.0.0.1:1/mcp",
        token="t",
        started_at="2026-09-03T00:00:00+00:00",
        version="test",
    )
    paths.discovery.write_text(info.model_dump_json(), encoding="utf-8")
    assert read_discovery_result(paths) == info


@pytest.mark.parametrize("record", _UNTRUSTED_RECORDS)
def test_claim_refuses_on_an_untrusted_record_without_binding_or_overwriting(paths: DaemonPaths, record: str) -> None:
    """The claim gate must fire for an unparseable record, not only a live one."""
    paths.discovery.write_text(record, encoding="utf-8")
    before = paths.discovery.read_bytes()

    with pytest.raises(DaemonRecordInvalidError) as refusal:
        claim_single_instance(paths, port=0, token="a-token", version="test")

    message = str(refusal.value)
    assert str(paths.discovery) in message, "the refusal must name the file to inspect"
    assert paths.discovery.read_bytes() == before, "a refused claim rewrote the discovery record"


def test_claim_still_succeeds_when_the_slot_is_genuinely_free(paths: DaemonPaths) -> None:
    """The refusal is conditional on evidence: a clean slot is still claimable."""
    claim = claim_single_instance(paths, port=0, token="a-token", version="test")
    try:
        assert claim.info.pid == os.getpid()
        assert read_discovery(paths) == claim.info
    finally:
        claim.sock.close()


@pytest.mark.parametrize("record", _UNTRUSTED_RECORDS)
def test_client_refuses_to_auto_start_over_an_untrusted_record(
    paths: DaemonPaths, record: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-start is gated on ABSENT, so an untrusted record spawns nothing."""
    pytest.importorskip("fastmcp")
    from trw_memory.daemon import client as client_module

    paths.discovery.write_text(record, encoding="utf-8")
    before = paths.discovery.read_bytes()
    spawns: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", spawns.append)

    daemon_client = client_module.DaemonClient(paths=paths)
    with pytest.raises(DaemonRecordInvalidError):
        daemon_client._attach()

    assert spawns == [], "a second daemon was spawned over a record that may name a live one"
    assert paths.discovery.read_bytes() == before
    assert not paths.token.exists(), "a refused attach minted a token"


def test_the_diagnostic_probe_still_collapses_to_none(paths: DaemonPaths) -> None:
    """``read_discovery`` stays two-valued for read-only callers that act on nothing."""
    paths.discovery.write_text("}{ not json", encoding="utf-8")
    assert read_discovery(paths) is None


# ── W03: an unreadable token file ────────────────────────────────────────────


def _plant_symlink(paths: DaemonPaths, tmp_path: Path) -> Path:
    victim = tmp_path / "victim-token"
    victim.write_text("a-secret-that-must-not-be-read-or-replaced", encoding="utf-8")
    paths.token.symlink_to(victim)
    return victim


def test_a_symlinked_token_raises_and_is_never_replaced(paths: DaemonPaths, tmp_path: Path) -> None:
    """The O_NOFOLLOW refusal must reach the caller, not become a rotation."""
    victim = _plant_symlink(paths, tmp_path)
    victim_before = victim.read_bytes()

    with pytest.raises(TokenUnreadableError) as refusal:
        ensure_token(paths)

    message = str(refusal.value)
    assert str(paths.token) in message, "the refusal must name the file"
    assert "NOT regenerated" in message
    assert paths.token.is_symlink(), "the token path was replaced"
    assert victim.read_bytes() == victim_before, "the symlink target was overwritten"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the mode bits this asserts on")
def test_an_unreadable_token_raises_and_is_never_replaced(paths: DaemonPaths) -> None:
    """A permission change locks the operator out of a rotation, not into one."""
    paths.token.write_text("the-live-daemons-token", encoding="utf-8")
    paths.token.chmod(0o000)
    before = paths.token.stat().st_ino
    try:
        with pytest.raises(TokenUnreadableError, match="NOT regenerated"):
            ensure_token(paths)
        assert paths.token.stat().st_ino == before, "the token file was replaced"
    finally:
        paths.token.chmod(0o600)
    assert paths.token.read_text(encoding="utf-8") == "the-live-daemons-token"


@pytest.mark.parametrize(
    ("content", "case"),
    [(b"\xff\xfe not utf-8 \x80", "invalid_utf8"), (b"   \n", "empty")],
)
def test_an_undecodable_or_empty_token_raises_and_is_never_replaced(
    paths: DaemonPaths, content: bytes, case: str
) -> None:
    """``write_secret_file`` is atomic, so neither state is a partial write."""
    paths.token.write_bytes(content)

    with pytest.raises(TokenUnreadableError, match="NOT regenerated"):
        ensure_token(paths)

    assert paths.token.read_bytes() == content, f"the {case} token file was rewritten"


def test_a_genuinely_absent_token_is_still_generated_at_0600(paths: DaemonPaths) -> None:
    """First run is still first run: the refusal is conditional, not blanket."""
    assert not paths.token.exists()

    token = ensure_token(paths)

    assert token and paths.token.read_text(encoding="utf-8") == token
    assert paths.token.stat().st_mode & 0o777 == 0o600
    assert ensure_token(paths) == token, "a second call minted a new token"
