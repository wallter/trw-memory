"""F5/P1-C parity for trw-memory's own platform-egress trust gate.

trw-memory cannot import trw-mcp's ``_platform_trust`` module (it is a
standalone, publicly-distributed package trw-mcp depends on, not the other
way around), so it carries its own small trust boundary in
``sync/_remote_common.py`` (``bearer_allowed_for`` / ``build_platform_headers``)
mirroring trw-mcp's policy: https to a trusted host, or http to a loopback
dev host, and never trust derived from a project's tracked
``MemoryConfig.platform_url``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from trw_memory.sync._remote_common import (
    DEFAULT_TRUSTED_PLATFORM_HOST,
    bearer_allowed_for,
    build_platform_headers,
    trusted_platform_hosts,
)

#: The single sanctioned chokepoint, plus modules whose "Bearer" header is a
#: different, unrelated credential (not the TRW platform egress this census
#: guards).
_CENSUS_EXEMPTIONS = {
    "sync/_remote_common.py",
    # jev's judge HTTP client authenticates to an operator-configured
    # OpenAI-compatible judge endpoint (self._base_url/self._api_key) --
    # a distinct credential and threat model from TRW_PLATFORM_API_KEY /
    # MemoryConfig.platform_url, which is what this census guards.
    "decisions/_jev_http.py",
}


def test_default_host_always_trusted() -> None:
    assert DEFAULT_TRUSTED_PLATFORM_HOST in trusted_platform_hosts()


def test_project_tracked_platform_url_never_adds_trust() -> None:
    assert "attacker.host" not in trusted_platform_hosts()


def test_bearer_allowed_for_untrusted_https_refused() -> None:
    assert not bearer_allowed_for("https://attacker.host/v1/x")


def test_bearer_allowed_for_default_host_allowed() -> None:
    assert bearer_allowed_for(f"https://{DEFAULT_TRUSTED_PLATFORM_HOST}/v1/x")


def test_bearer_allowed_for_http_loopback_allowed() -> None:
    assert bearer_allowed_for("http://127.0.0.1:5002/v1/x")


def test_bearer_allowed_for_http_non_loopback_refused() -> None:
    assert not bearer_allowed_for("http://attacker.host/v1/x")


def test_build_platform_headers_withholds_bearer_from_untrusted_url() -> None:
    headers = build_platform_headers("secret", "https://attacker.host/v1/learnings")
    assert "Authorization" not in headers


def test_build_platform_headers_attaches_bearer_to_trusted_url() -> None:
    headers = build_platform_headers("secret", f"https://{DEFAULT_TRUSTED_PLATFORM_HOST}/v1/learnings")
    assert headers["Authorization"] == "Bearer secret"


def test_build_platform_headers_no_key_yields_no_header() -> None:
    headers = build_platform_headers(None, f"https://{DEFAULT_TRUSTED_PLATFORM_HOST}/v1/learnings")
    assert "Authorization" not in headers


def test_no_other_module_builds_a_raw_bearer_header() -> None:
    """AST census: an f-string with a literal ``Bearer `` prefix lives only in the exempted file."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "trw_memory"
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root).as_posix()
        if rel in _CENSUS_EXEMPTIONS:
            continue
        text = path.read_text(encoding="utf-8")
        if "Bearer " not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            if any(
                isinstance(value, ast.Constant) and isinstance(value.value, str) and "Bearer " in value.value
                for value in node.values
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], f"raw f'Bearer {{...}}' header construction outside build_platform_headers: {offenders}"
