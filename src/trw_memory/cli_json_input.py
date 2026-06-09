"""Structural JSON-file input seam for the trw-memory CLI.

A small deep module whose Interface is three functions and one exception, and
whose Implementation hides every way reading a JSON document from disk can fail.
Callers (``import``, ``wiki-lint``) hand it a path and receive either valid data
or a single :class:`JsonInputError` whose message is *content-free*: it names the
source and the structural reason (error class / shape) but never echoes payload
bytes, failing JSON snippets, or the raw interpreter exception string.

This keeps the leak-prone ``json.loads(path.read_text(...))`` idiom — and its
catch-all CLI error boundary — out of the command handlers, giving the JSON-file
input concern a single deep Seam with high locality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonInputError(Exception):
    """A structural failure loading a JSON-file CLI input.

    The message is deliberately content-free: it identifies the source and the
    structural reason (missing, unreadable, non-UTF-8, malformed, wrong shape)
    without reproducing the payload or the underlying interpreter exception text.
    """


def read_source_text(path: Path, *, source: str) -> str:
    """Read ``path`` as UTF-8 text, mapping every read failure to a structural error.

    Covers missing, directory-as-file, otherwise-unreadable (permission, etc.),
    and non-UTF-8 inputs. The raised message names only the source and the failure
    class — never the bytes that could not be decoded.
    """
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise JsonInputError(f"file not found: {source}") from exc
    except IsADirectoryError as exc:
        raise JsonInputError(f"{source} is a directory, not a file") from exc
    except OSError as exc:
        raise JsonInputError(f"cannot read {source} ({type(exc).__name__})") from exc

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonInputError(f"{source} is not valid UTF-8 text (UnicodeDecodeError)") from exc


def load_json_document(path: Path, *, source: str) -> Any:
    """Read and parse a UTF-8 JSON document, returning its top-level value.

    Read/decode failures surface via :func:`read_source_text`; a malformed
    document raises a structural :class:`JsonInputError` carrying the decoder's
    reason and position (line/column) only — not the offending text.
    """
    text = read_source_text(path, source=source)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonInputError(
            f"{source} is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
        ) from exc


def load_json_array(path: Path, *, source: str) -> list[Any]:
    """Load a JSON document and require its top-level value to be an array.

    Raises a structural :class:`JsonInputError` (naming the actual JSON type,
    not its contents) when the document is not a list.
    """
    value = load_json_document(path, source=source)
    if not isinstance(value, list):
        raise JsonInputError(f"{source} must be a JSON array, got {json_type_name(value)}")
    return value


def json_type_name(value: object) -> str:
    """Return the JSON type name for ``value`` for use in structural diagnostics."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__
