"""Pytest plugin: eager-load the numpy/transformers stack at bootstrap time.

Registered via ``-p tests._cov_numpy_warmup`` on coverage gates whose ``--cov``
scope does **not** transitively import numpy at startup — specifically the
INFRA-020 security gate ``--cov=trw_memory.security`` (the security subpackage
imports no numpy of its own).

Why this is needed
------------------
``-p`` plugins are imported during pytest's initial bootstrap, *before*
``pytest-cov`` starts its coverage tracer. Importing the numpy /
``sentence_transformers`` C-extension stack here loads it while no tracer is
active, so numpy's compiled modules load exactly once and get cached.

Without this warm-up, numpy first loads *mid-collection* — when
``test_bench_hype`` / the retrieval tests import ``sentence_transformers`` — at
which point the coverage tracer is already engaged and numpy's
``_multiarray_umath`` C extension raises::

    ImportError: cannot load module more than once per process

The main ``--cov=trw_memory`` Test gate does not need this because coverage
imports the whole ``trw_memory`` package (which eagerly loads numpy) during its
own startup; only the narrower ``--cov=trw_memory.security`` gate misses that
early import.

IMPORTANT — coverage correctness
--------------------------------
This warm-up imports ONLY the third-party numpy stack, never ``trw_memory`` (or
``trw_memory.security``). Pre-importing a measured module *before* the coverage
tracer starts would make coverage miss its import-time lines and badly
under-report the security branch coverage (observed: ~66% instead of ~90%+).
Keeping the warm-up to third-party packages caches numpy's C extensions without
touching any measured module. Test-harness ordering shim only — no production
behaviour change; a no-op when the numpy stack is unavailable.
"""

from __future__ import annotations

try:  # pragma: no cover - environment-dependent warm-up, intentionally silent
    import numpy  # noqa: F401  (load numpy's C extensions before the cov tracer)
    import sentence_transformers  # noqa: F401  (the transformers->numpy re-import path)
except Exception:
    pass
