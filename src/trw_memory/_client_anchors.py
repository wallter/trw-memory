"""Public-boundary validation of the ``anchors`` argument to ``MemoryClient.store``.

Belongs to ``_client_store.py``. An anchor names the code a stored entry is about, and
it is stored repo-relative: a machine path never goes in (the ``trw_learn`` writer
enforces the same rule before it reaches this package). The check runs before anything
is written, and refuses with :class:`~trw_memory.exceptions.SchemaValidationError`
naming the offending value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import Anchor
from trw_memory.storage._anchor_index import normalize_anchor_file

#: The model's own bound (``MemoryEntry.anchors`` ``max_length``); restated so the refusal names the field.
MAX_ANCHORS = 3

MAX_FILE_CHARS = 1_024
_SHOWN_CHARS = 120
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _shown(value: object) -> str:
    """``repr`` of at most :data:`_SHOWN_CHARS` characters of *value*, so a control character cannot break a log line."""
    return repr(value[:_SHOWN_CHARS]) if isinstance(value, str) else repr(value)[:_SHOWN_CHARS]


def _refuse(message: str) -> SchemaValidationError:
    return SchemaValidationError(
        f"memory store schema invalid for fields: anchors ({message})", failed_fields=["anchors"]
    )


def _check_file(file: object) -> None:
    """Refuse any ``file`` that is not a plain repo-relative, forward-slash path."""
    shown = _shown(file)
    if not isinstance(file, str):
        raise _refuse(f"anchor file must be a string, got {shown}")
    if len(file) > MAX_FILE_CHARS:
        raise _refuse(f"anchor file is longer than {MAX_FILE_CHARS} characters: {shown}")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in file):
        raise _refuse(f"anchor file contains a control character (NUL, newline, tab...): {shown}")
    if not file.strip() or file != file.strip():
        raise _refuse(f"anchor file is empty or has leading or trailing whitespace: {shown}")
    if "\\" in file:
        raise _refuse(f"anchor file must use forward slashes, not a backslash: {shown}")
    if file.startswith(("/", "~")) or _URI_SCHEME.match(file):
        raise _refuse(f"anchor file must be repo-relative, not absolute, home-relative, a drive or a URI: {shown}")
    rest = file.removeprefix("./")  # one leading ``./`` is normalised away downstream
    if any(segment in ("", ".", "..") for segment in rest.split("/")):
        raise _refuse(f"anchor file has an empty, '.' or '..' path segment: {shown}")
    if normalize_anchor_file(file) is None:  # pragma: no cover - defence in depth: the single posting-key definition
        raise _refuse(f"anchor file is not a usable repo-relative path: {shown}")


def validate_store_anchors(anchors: Sequence[Anchor | Mapping[str, object]] | None) -> list[Anchor] | None:
    """Return *anchors* as validated :class:`Anchor` models (``None`` stays ``None``), or raise.

    Accepts the two shapes the writers already use: ``Anchor`` models (``memory_store``)
    and their dict form (``trw_learn``). URL-encoded text (``%2F``, ``%2e%2e``) is never decoded and is
    not refused: it is a literal file name here. Raises ``SchemaValidationError`` for a non-sequence,
    more than :data:`MAX_ANCHORS` anchors, a malformed anchor, or a ``file`` that is not repo-relative.
    """
    if anchors is None:
        return None
    if isinstance(anchors, (str, bytes, Mapping)) or not isinstance(anchors, Sequence):
        raise _refuse(f"anchors must be a list of Anchor, got {type(anchors).__name__}")
    if len(anchors) > MAX_ANCHORS:
        raise _refuse(f"at most {MAX_ANCHORS} anchors per entry, got {len(anchors)}")
    checked: list[Anchor] = []
    for item in anchors:
        raw_file = item.get("file") if isinstance(item, Mapping) else getattr(item, "file", None)
        _check_file(raw_file)
        try:
            checked.append(Anchor.model_validate(item.model_dump() if isinstance(item, Anchor) else item))
        except ValidationError as exc:
            raise _refuse(f"invalid anchor {_shown(raw_file)}: {exc.error_count()} field error(s)") from exc
    return checked
