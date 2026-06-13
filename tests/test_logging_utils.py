"""Wave 12: coverage gap-fill for trw_memory._logging (lines 54, 70-74, 80, 98, 106, 122, 148-149)."""
from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import pytest

from trw_memory._logging import (
    _add_component,
    _redact_secrets,
    _verbosity_to_level,
    configure_logging,
)


def _make_event(**kwargs: Any) -> MutableMapping[str, Any]:
    return dict(kwargs)


class TestRedactSecrets:
    def test_sensitive_key_is_redacted(self) -> None:
        event = _make_event(password="s3cr3t", event="login")
        result = _redact_secrets(None, "info", event)
        assert result["password"] == "***REDACTED***"

    def test_bearer_token_in_value_is_redacted(self) -> None:
        event = _make_event(authorization="Bearer abc123", event="req")
        result = _redact_secrets(None, "info", event)
        assert "***REDACTED***" in result["authorization"]

    def test_non_sensitive_key_unchanged(self) -> None:
        event = _make_event(user="alice", event="login")
        result = _redact_secrets(None, "info", event)
        assert result["user"] == "alice"


class TestAddComponent:
    def test_trw_memory_module_name_extracts_component(self) -> None:
        event = _make_event(_logger_name="trw_memory.storage.sqlite_backend", event="x")
        result = _add_component(None, "info", event)
        assert result["component"] == "storage.sqlite_backend"

    def test_non_trw_memory_logger_uses_full_name(self) -> None:
        event = _make_event(_logger_name="third_party.lib", event="x")
        result = _add_component(None, "info", event)
        assert result["component"] == "third_party.lib"

    def test_no_logger_name_no_component(self) -> None:
        event = _make_event(event="x")
        result = _add_component(None, "info", event)
        assert "component" not in result

    def test_existing_component_not_overwritten(self) -> None:
        event = _make_event(_logger_name="trw_memory.foo", component="custom", event="x")
        result = _add_component(None, "info", event)
        assert result["component"] == "custom"


class TestVerbosityToLevel:
    def test_negative_verbosity_returns_warning(self) -> None:
        assert _verbosity_to_level(-1) == logging.WARNING

    def test_zero_verbosity_returns_info(self) -> None:
        assert _verbosity_to_level(0) == logging.INFO

    def test_one_verbosity_returns_debug(self) -> None:
        assert _verbosity_to_level(1) == logging.DEBUG


class TestConfigureLogging:
    def test_explicit_log_level_override(self) -> None:
        configure_logging(log_level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_json_output_false_uses_console_renderer(self) -> None:
        configure_logging(json_output=False)

    def test_json_output_true_uses_json_renderer(self) -> None:
        configure_logging(json_output=True)

    def test_version_bind_exception_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.metadata as meta

        monkeypatch.setattr(meta, "version", lambda _: (_ for _ in ()).throw(Exception("no ver")))
        configure_logging()
