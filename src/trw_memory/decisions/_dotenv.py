"""A tiny, scoped ``.env`` line parser — no ``python-dotenv`` dependency.

Deliberately narrow: it only ever returns the keys a caller allowlists by exact
name, so a ``.env`` full of unrelated project secrets never has more of itself
parsed or held in memory than the one key the decision seam actually reads
(``OPENROUTER_API_KEY``). There is no prefix matching: a prefix is a wildcard,
and a wildcard over a repo-controlled file is how a ``.env`` came to be able to
redirect credential egress (release-verify R1).

**The path is repo-controlled, so opening it is the risky part** (release-verify
N3/N4). Git cannot commit a FIFO but CAN commit a symlink, so a cloned repo may
carry ``.env -> /dev/zero`` (an unbounded read), ``.env -> some-fifo`` (a read
that blocks the MCP handler forever), or ``.env -> ~/.config/<tool>/creds`` (a
read of a file the operator never meant to expose). This module therefore opens
with ``O_NOFOLLOW | O_NONBLOCK``, requires a REGULAR file by ``fstat`` on the
open descriptor — not a pre-open ``stat``, which is a TOCTOU — and reads a
bounded number of bytes from that same descriptor. A symlink at ``.env`` is
REFUSED outright rather than resolved: the convenience of one is not worth
re-opening the class of bug, and an operator who wants a shared dotenv can
point ``dotenv_path`` at the real file.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

#: A ``.env`` this large is not the small key/value file this parser is for;
#: refuse it rather than read an arbitrary blob into memory.
_MAX_DOTENV_BYTES = 64 * 1024

#: Read granularity. Irrelevant to behavior — only to how many syscalls a
#: 64 KiB ceiling costs.
_CHUNK_BYTES = 8192

#: Not every platform defines these. Absent, the flag contributes nothing and
#: the ``S_ISREG`` check below is still the gate that matters.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def _read_small_regular_file(path: str | Path) -> str | None:
    """Return the text of a small, regular, non-symlinked file, else ``None``.

    ``None`` covers every refusal — missing, symlink, FIFO/device/directory,
    over :data:`_MAX_DOTENV_BYTES`, not valid UTF-8, or any OS error — because
    the caller treats them all identically: there is no dotenv to read.
    """
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK)
    # trw-fail-silent-allow: missing, unopenable, a symlink (O_NOFOLLOW -> ELOOP) or a bad path type all mean "no optional dotenv to read"; raising would let a hostile repo .env fail the tool call
    except (OSError, ValueError, TypeError):
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        remaining = _MAX_DOTENV_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, _CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_DOTENV_BYTES:
            return None
        return raw.decode("utf-8")
    # trw-fail-silent-allow: same contract after the open -- a read error or non-UTF-8 bytes in an untrusted file mean there is no key to be had, which the caller already handles as an abstention
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        os.close(fd)


def _parse_dotenv_lines(text: str, *, allowed_keys: frozenset[str]) -> dict[str, str]:
    """Parse ``KEY=value`` lines already read into ``text``, keeping only allowlisted keys.

    Pure (no I/O) so a caller that needs its OWN existence/refusal classification around the read
    -- e.g. :mod:`trw_memory.decisions._enablement`, which must distinguish a genuinely absent
    file from one the hardened reader refused -- can supply ``text`` itself rather than going
    through :func:`parse_dotenv_subset`'s "any refusal yields empty" contract. Handles
    ``export KEY=value``, ``#`` comments, blank lines, and one layer of matching quotes around the
    value; nothing more elaborate (no multiline values, no variable interpolation) is needed for
    the one key shape this reads.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key not in allowed_keys:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def parse_dotenv_subset(path: str | Path, *, allowed_keys: frozenset[str]) -> dict[str, str]:
    """Parse ``KEY=value`` lines from ``path``, keeping only allowlisted keys.

    Anything :func:`_read_small_regular_file` refuses yields an empty mapping
    rather than raising — a dotenv is an optional convenience, never a required
    input, and the file is repo-controlled so both its contents and its kind are
    untrusted. See :func:`_parse_dotenv_lines` for the line-parsing rules.
    """
    text = _read_small_regular_file(path)
    if text is None:
        return {}
    return _parse_dotenv_lines(text, allowed_keys=allowed_keys)


__all__ = ["parse_dotenv_subset"]
