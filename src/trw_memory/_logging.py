"""Structured logging configuration for trw-memory CLI.

As a library, trw-memory does NOT configure logging globally — the consuming
application (trw-mcp, user projects) owns that. This module is only called
by the ``trw-memory`` CLI entry point (``cli.py:main``).

Environment variables:
    TRW_LOG_LEVEL   -- explicit level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL        -- fallback for generic deployments
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_PATTERNS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "private_key",
        "access_key",
        "session_id",
    }
)

_SENSITIVE_VALUE_RE = re.compile(
    r"((?:Bearer|Basic|Token)\s+)\S+",
    re.IGNORECASE,
)


def _redact_secrets(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        key_lower = key.lower()
        if any(pat in key_lower for pat in _SENSITIVE_PATTERNS):
            event_dict[key] = "***REDACTED***"
        elif isinstance(event_dict[key], str):
            event_dict[key] = _SENSITIVE_VALUE_RE.sub(
                r"\1***REDACTED***",
                event_dict[key],
            )
    return event_dict


def _add_component(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    logger_name = event_dict.get("_logger_name") or event_dict.get("logger")
    if logger_name and "component" not in event_dict:
        parts = str(logger_name).split(".")
        if len(parts) > 1 and parts[0] == "trw_memory":
            event_dict["component"] = ".".join(parts[1:])
        else:
            event_dict["component"] = str(logger_name)
    return event_dict


def _verbosity_to_level(verbosity: int) -> int:
    if verbosity < 0:
        return logging.WARNING
    return {0: logging.INFO, 1: logging.DEBUG}.get(verbosity, logging.DEBUG)


def configure_logging(
    *,
    verbosity: int = 0,
    log_level: str | None = None,
    json_output: bool | None = None,
) -> None:
    """Configure structlog for the trw-memory CLI only.

    Args:
        verbosity: CLI verbosity (-1=quiet, 0=INFO, 1=DEBUG).
        log_level: Explicit override (e.g. "WARNING").
        json_output: Force JSON/console. None=auto from TTY.
    """
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        env_level = os.environ.get("TRW_LOG_LEVEL") or os.environ.get("LOG_LEVEL")
        level = getattr(logging, env_level.upper(), logging.INFO) if env_level else _verbosity_to_level(verbosity)

    if json_output is None:
        use_json = not sys.stderr.isatty()
    else:
        use_json = json_output

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_component,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_secrets,
    ]

    renderer: structlog.types.Processor
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # Suppress noisy third-party loggers below WARNING
    for noisy in ("sentence_transformers", "huggingface_hub", "torch", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # Bind service version to all log records for incident triage
    try:
        from importlib.metadata import version as _get_version

        structlog.contextvars.bind_contextvars(
            service_version=_get_version("trw-memory"),
        )
    except Exception:  # justified: best-effort — version binding is non-critical
        pass
