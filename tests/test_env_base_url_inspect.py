"""PRD-FIX-156 residual (B71-100 burn-down): ``_inspect_base_url``'s malformed-URL
except handler used to degrade to ``(False, "", "")`` with no trace. It now logs
one structured debug event with the error type (never the raw URL, which may
carry userinfo/credentials in the malformed case)."""

from __future__ import annotations

import structlog

from trw_memory.decisions._env import _inspect_base_url


def test_malformed_url_logs_without_echoing_the_url() -> None:
    # An unbalanced IPv6 bracket makes urlsplit raise ValueError.
    malformed = "https://[::1:secret-token@evil.example/"

    with structlog.testing.capture_logs() as captured:
        result = _inspect_base_url(malformed)

    assert result == (False, "", "")
    events = [e for e in captured if e.get("event") == "jev_base_url_malformed"]
    assert len(events) == 1
    assert "secret-token" not in str(events[0])


def test_malformed_port_logs_the_error_type_not_the_port_text() -> None:
    """urlsplit's port error quotes the port string, which here is a secret."""
    with structlog.testing.capture_logs() as captured:
        result = _inspect_base_url("https://openrouter.ai:hunter2-secret/api")

    assert result == (False, "", "")
    (event,) = [e for e in captured if e.get("event") == "jev_base_url_malformed"]
    assert event["error_type"] == "ValueError"
    assert "hunter2-secret" not in str(event)


def test_well_formed_allowed_url_is_accepted_without_warning() -> None:
    with structlog.testing.capture_logs() as captured:
        result = _inspect_base_url("https://openrouter.ai/api/v1")

    assert result == (True, "openrouter.ai", "https")
    assert not any(e.get("event") == "jev_base_url_malformed" for e in captured)


def test_disallowed_host_is_rejected_without_logging() -> None:
    """Negative control: a syntactically valid but non-allowlisted host is
    rejected via the normal `allowed=False` path, not the malformed branch."""
    with structlog.testing.capture_logs() as captured:
        result = _inspect_base_url("https://evil.example/")

    assert result == (False, "evil.example", "https")
    assert not any(e.get("event") == "jev_base_url_malformed" for e in captured)
