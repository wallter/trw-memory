"""Where this test suite lives: the trw-memory package root, and the monorepo root if any.

The public ``wallter/trw-memory`` repository is the package alone, so anything
under ``src/trw_memory`` must be addressed from :data:`PACKAGE_ROOT`; only tests
that need the monorepo (its git history, sibling packages, e.g. trw-mcp's
source tree) use :data:`MONOREPO_ROOT` and must carry :data:`requires_monorepo`
rather than trust an unconditional ``parents[2]`` (B71-127b: the public repo's
``tests/`` sits one level shallower than the monorepo checkout's, so a bare
``parents[2]`` there resolves to whatever happens to be one directory above the
checkout — not a deliberate skip, a silent false pass or a wrong path).

Mirrors ``trw-mcp/tests/_layout.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
_candidate = PACKAGE_ROOT.parent
# A GitHub Actions checkout lives at ``<work>/trw-memory/trw-memory``, so the parent
# of the package root DOES contain a ``trw-memory`` directory there too — that is
# the checkout itself, not a monorepo. The monorepo is the only layout with the
# independent release manifest or the sibling trw-mcp/trw-memory + CLAUDE.md identity.
MONOREPO_ROOT: Path | None = (
    _candidate
    if (
        _candidate.name != "site-packages"
        and (
            (_candidate / "release-packages.yaml").is_file()
            or (
                (_candidate / "trw-mcp" / "pyproject.toml").is_file()
                and (_candidate / "trw-memory" / "pyproject.toml").is_file()
                and (_candidate / "CLAUDE.md").is_file()
            )
        )
    )
    else None
)
requires_monorepo = pytest.mark.skipif(
    MONOREPO_ROOT is None, reason="needs the monorepo checkout (public repo is the package alone)"
)
