"""``DaemonInfo.url`` must name the daemon's own loopback binding, never a forged host.

PRD-FIX-157-FR01: a discovery record is read from
``<user_memory_dir>/daemon.json``, a file an attacker with same-user
filesystem access could plant (R13). Before this validator, ``url`` was a
plain ``str`` field, so a forged record naming any host would construct a
live ``DaemonInfo`` and a client would trust it as a real endpoint. The only
production writer, ``endpoint_url()`` (``trw_memory.daemon._loopback``),
always emits ``http://127.0.0.1:<port>/mcp``, so a validator that trusts only
the loopback forms changes no production behaviour -- it only refuses a
record today's code never produces.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pydantic
import pytest

from trw_memory.daemon import DaemonInfo, DaemonPaths
from trw_memory.daemon._discovery import DISCOVERY_SCHEMA_VERSION, DiscoveryInvalid, read_discovery_result
from trw_memory.daemon._loopback import bind_loopback_socket, endpoint_url


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    resolved = DaemonPaths.resolve()
    resolved.user_memory_dir.mkdir(parents=True, exist_ok=True)
    return resolved


# ── FR01: reject a non-loopback or non-http(s) url ──────────────────────────

_REJECTED_URLS = [
    pytest.param("http://93.184.216.34:41234/mcp", id="public_ip"),
    pytest.param("http://example.com:41234/mcp", id="external_host"),
    pytest.param("https://attacker.invalid/mcp", id="external_host_https"),
    pytest.param("http://127.0.0.1.evil.com:41234/mcp", id="loopback_suffix_trick"),
    pytest.param("http://127.0.0.1@evil.com:41234/mcp", id="loopback_userinfo_trick"),
    pytest.param("file:///etc/passwd", id="file_scheme"),
    pytest.param("http:///mcp", id="missing_host"),
    pytest.param("http://:41234/mcp", id="empty_host"),
    pytest.param("http://127.1:41234/mcp", id="non_canonical_loopback_shorthand"),
    pytest.param("http://LOCALHOST.evil.com:41234/mcp", id="localhost_suffix_trick"),
    pytest.param("ftp://127.0.0.1:41234/mcp", id="non_http_scheme"),
    pytest.param("http://evil.com:999999/mcp", id="malformed_port"),
    # A trusted host with a bad port: only the port check can refuse these.
    pytest.param("http://127.0.0.1:999999/mcp", id="loopback_out_of_range_port"),
    pytest.param("http://localhost:abc/mcp", id="loopback_nonnumeric_port"),
]


@pytest.mark.parametrize("url", _REJECTED_URLS)
def test_daemon_info_rejects_non_loopback_url(url: str) -> None:
    """A forged or attacker-chosen url must fail pydantic validation, not construct."""
    with pytest.raises(pydantic.ValidationError):
        DaemonInfo(pid=1, url=url, started_at="2026-01-01T00:00:00Z", version="x")


# ── FR01: every real endpoint format still constructs ───────────────────────

_ACCEPTED_URLS = [
    pytest.param("http://127.0.0.1:41234/mcp", id="loopback_ip"),
    pytest.param("http://localhost:41234/mcp", id="localhost_hostname"),
    pytest.param("http://[::1]:41234/mcp", id="bracketed_ipv6_loopback"),
    pytest.param("http://LOCALHOST:41234/mcp", id="localhost_case_folded"),
    pytest.param("https://127.0.0.1:41234/mcp", id="https_scheme"),
]


@pytest.mark.parametrize("url", _ACCEPTED_URLS)
def test_daemon_info_accepts_trusted_loopback_forms(url: str) -> None:
    """Every real endpoint format the daemon could ever advertise still constructs."""
    info = DaemonInfo(pid=1, url=url, started_at="2026-01-01T00:00:00Z", version="x")
    assert info.url == url


# ── FR01: a forged on-disk record reads as invalid, not live and not absent ─


def test_forged_non_loopback_record_reads_as_invalid_not_live(paths: DaemonPaths) -> None:
    """A record naming an attacker-chosen host must not resolve to a live daemon."""
    forged = json.dumps(
        {
            "schema_version": DISCOVERY_SCHEMA_VERSION,
            "pid": os.getpid(),
            "url": "http://93.184.216.34:41234/mcp",
            "started_at": "2026-09-26T00:00:00+00:00",
            "version": "test",
        }
    )
    paths.discovery.write_text(forged, encoding="utf-8")

    result = read_discovery_result(paths)

    assert isinstance(result, DiscoveryInvalid)
    assert not isinstance(result, DaemonInfo)
    assert result.path == paths.discovery
    assert result.reason, "an invalid record must carry an operator-readable reason"


# ── FR01: the only production writer still round-trips ──────────────────────


def test_endpoint_url_round_trips_through_daemon_info() -> None:
    """``endpoint_url()`` -- the only production writer of ``DaemonInfo.url`` -- must still validate.

    A regression here would mean the validator rejects the daemon's own real
    binding, breaking every daemon start.
    """
    sock = bind_loopback_socket(0)
    try:
        url = endpoint_url(sock)
    finally:
        sock.close()

    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/mcp")

    info = DaemonInfo(pid=os.getpid(), url=url, started_at="2026-01-01T00:00:00Z", version="test")
    assert info.url == url


def test_endpoint_url_matches_bound_socket_port() -> None:
    """Sanity: ``endpoint_url`` names the port the OS actually assigned, not a fixed one."""
    sock = bind_loopback_socket(0)
    try:
        _, port = sock.getsockname()[:2]
        assert isinstance(sock, socket.socket)
        assert endpoint_url(sock) == f"http://127.0.0.1:{port}/mcp"
    finally:
        sock.close()
