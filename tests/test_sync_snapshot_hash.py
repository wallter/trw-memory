"""Tests for PRD-INFRA-066 off-box snapshot hash publish (C1)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig
from trw_memory.sync.remote import publish_snapshot_hash

if TYPE_CHECKING:
    from pytest import MonkeyPatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_snapshot(path: Path, payload: bytes = b"sqlite-fake\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class _RecordingClient:
    """Stand-in for httpx.Client that records calls and returns a scripted response."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[dict[str, Any]] = []
        self._status_code = status_code

    def __enter__(self) -> _RecordingClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(status_code=self._status_code, json={})


def _cfg(**overrides: Any) -> MemoryConfig:
    base = {
        "local_only": False,
        "sync_enabled": True,
        "memory_snapshot_publish_hash": True,
        "platform_url": "https://trw.example.com",
        "platform_api_key": "k123",
    }
    base.update(overrides)
    return MemoryConfig(**base)


# ---------------------------------------------------------------------------
# Gating: opt-in defaults MUST NOT touch the network
# ---------------------------------------------------------------------------


def test_skips_when_sync_disabled(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg(sync_enabled=False)
    called: list[Any] = []
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: called.append((a, kw)) or _RecordingClient())
    result = publish_snapshot_hash(snap, cfg)
    assert result == {"success": False, "remote_id": None, "retryable": False}
    assert called == [], "sync_enabled=False MUST NOT instantiate an HTTP client"


def test_skips_when_publish_hash_knob_off(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg(memory_snapshot_publish_hash=False)
    called: list[Any] = []
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: called.append((a, kw)) or _RecordingClient())
    result = publish_snapshot_hash(snap, cfg)
    assert result["success"] is False
    assert called == [], "publish-hash knob off MUST NOT instantiate an HTTP client"


def test_skips_when_platform_url_empty(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg(platform_url="")
    called: list[Any] = []
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: called.append((a, kw)) or _RecordingClient())
    result = publish_snapshot_hash(snap, cfg)
    assert result["success"] is False
    assert called == []


def test_raises_local_only_violation(tmp_path: Path) -> None:
    """local_only=True MUST raise immediately — never attempt network."""
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    # Construct with local_only=True; validator sets sync_enabled=False anyway,
    # so use explicit kwargs.
    cfg = MemoryConfig(
        local_only=True,
        platform_url="https://trw.example.com",
    )
    with pytest.raises(LocalOnlyViolationError):
        publish_snapshot_hash(snap, cfg)


def test_skips_when_url_invalid_http(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """http:// URLs are rejected unless TRW_DEBUG=true."""
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    monkeypatch.delenv("TRW_DEBUG", raising=False)
    cfg = _cfg(platform_url="http://insecure.example.com")
    called: list[Any] = []
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: called.append((a, kw)) or _RecordingClient())
    result = publish_snapshot_hash(snap, cfg)
    assert result["success"] is False
    assert called == []


def test_skips_when_snapshot_missing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    missing = tmp_path / "ghost.db"
    cfg = _cfg()
    called: list[Any] = []
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: called.append((a, kw)) or _RecordingClient())
    result = publish_snapshot_hash(missing, cfg)
    assert result["success"] is False
    assert called == []


# ---------------------------------------------------------------------------
# Success path — structural payload guards
# ---------------------------------------------------------------------------


def test_publish_succeeds_and_posts_metadata_only(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    expected_digest = _write_fake_snapshot(snap, b"known-content")
    cfg = _cfg()

    recorder = _RecordingClient(status_code=200)
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)

    result = publish_snapshot_hash(snap, cfg, installation_id="site-42")
    assert result["success"] is True
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"].endswith("/v1/memory/snapshot-hash")
    payload = call["json"]
    assert payload["digest"] == expected_digest
    assert payload["size_bytes"] == len(b"known-content")
    assert "created_at" in payload
    # installation_id is anonymized, NOT raw.
    assert payload["installation_id"] != "site-42"


def test_publish_hash_never_includes_snapshot_bytes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Structural guard: payload MUST NOT contain any byte-leak field.

    This is the sprint exit-criterion regression test — it fails loudly if a
    future refactor accidentally ships contents. Forbidden field suffixes:
    ``_contents``, ``_data``, ``_bytes``, ``_payload``.
    """
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap, b"secret-data")
    cfg = _cfg()

    recorder = _RecordingClient(status_code=200)
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)

    publish_snapshot_hash(snap, cfg)
    payload = recorder.calls[0]["json"]

    allowed = {"digest", "size_bytes", "created_at", "installation_id"}
    for key in payload:
        assert key in allowed, f"unexpected payload field: {key}"

    forbidden_suffixes = ("_contents", "_data", "_bytes", "_payload", "_raw")
    # NOTE: size_bytes is an allowed integer — do not flag it even though it
    # ends with _bytes. Scan only unknown keys.
    for key in payload:
        if key in allowed:
            continue
        for suffix in forbidden_suffixes:
            assert not key.endswith(suffix), f"forbidden suffix {suffix!r} in key {key!r} — potential contents leak"


def test_publish_sends_api_key_header(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg(platform_api_key="secret-key")
    recorder = _RecordingClient(status_code=200)
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)
    publish_snapshot_hash(snap, cfg)
    headers = recorder.calls[0]["headers"]
    assert headers.get("Authorization") == "Bearer secret-key"


# ---------------------------------------------------------------------------
# Failure paths — fail-open, retryable flags
# ---------------------------------------------------------------------------


def test_publish_returns_retryable_on_5xx(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg()
    recorder = _RecordingClient(status_code=503)
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)
    result = publish_snapshot_hash(snap, cfg)
    assert result == {"success": False, "remote_id": None, "retryable": True}


def test_publish_fails_open_on_httpx_error(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Connection errors must NEVER raise to the caller."""
    snap = tmp_path / "snapshot.db"
    _write_fake_snapshot(snap)
    cfg = _cfg()

    class _ExplodingClient:
        def __enter__(self) -> _ExplodingClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: _ExplodingClient())
    result = publish_snapshot_hash(snap, cfg)
    assert result["success"] is False
    assert result["retryable"] is True


# ---------------------------------------------------------------------------
# Hash correctness
# ---------------------------------------------------------------------------


def test_hash_matches_sha256_of_bytes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    snap = tmp_path / "snapshot.db"
    body = b"abc" * 10_000  # 30KB — exercises chunked hashing
    expected = _write_fake_snapshot(snap, body)
    cfg = _cfg()
    recorder = _RecordingClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)

    publish_snapshot_hash(snap, cfg)
    payload = recorder.calls[0]["json"]
    assert payload["digest"] == expected
    assert payload["size_bytes"] == len(body)


def test_hash_stable_for_real_snapshot_content(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """End-to-end: create a real SQLite snapshot, verify hash matches stored bytes."""
    src = tmp_path / "memory.db"
    conn = sqlite3.connect(str(src))
    conn.execute("CREATE TABLE memories (id TEXT, content TEXT)")
    conn.execute("INSERT INTO memories VALUES ('a', 'hello')")
    conn.commit()
    conn.close()

    from trw_memory.storage._snapshot import create_snapshot

    snap = tmp_path / "snap.db"
    create_snapshot(src, snap)
    expected = hashlib.sha256(snap.read_bytes()).hexdigest()

    cfg = _cfg()
    recorder = _RecordingClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: recorder)
    publish_snapshot_hash(snap, cfg)
    assert recorder.calls[0]["json"]["digest"] == expected
