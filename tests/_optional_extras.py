"""Skip markers for tests that exercise an optional trw-memory extra.

The isolated release gate installs only ``trw-memory[dev]``. A test whose
behavior needs an extra must say so and skip there, not fail -- and it must
not pass there by accident either, which is why the markers name the extra.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

requires_sqlite_vec = pytest.mark.skipif(
    find_spec("sqlite_vec") is None,
    reason="needs sqlite-vec: the SQLite backend reports supports_vectors() False without it; it is a base dependency, so reinstall trw-memory",
)
