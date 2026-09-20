"""Give mem0's extraction the session date instead of the wall clock (side-effect free to import).

mem0 OSS 2.0.x builds its extraction prompt with ``Current Date`` and ``Observation Date``
sections (``generate_additive_extraction_prompt(current_date=, timestamp=)``), but
``Memory.add`` never passes either, so both default to the benchmark machine's clock and
"yesterday" resolves to 2026. Mem0 Cloud takes the session timestamp. The shim wraps each
``add`` in :func:`mem0_extraction_date` so mem0 is measured the way its own prompt intends.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator


@contextlib.contextmanager
def mem0_extraction_date(day: str) -> Iterator[None]:
    """Pass ``day`` (YYYY-MM-DD) as mem0's extraction current/observation date for one add().

    Callers must serialise adds (the shim holds its write lock), so the module attribute swap
    cannot leak into a concurrent add.
    """
    import mem0.memory.main as mem0_main

    original = mem0_main.generate_additive_extraction_prompt
    mem0_main.generate_additive_extraction_prompt = functools.partial(original, current_date=day, timestamp=day)
    try:
        yield
    finally:
        mem0_main.generate_additive_extraction_prompt = original
