"""The default embedding model and the Hub commit each known model is pinned to.

PRD-CORE-302 FR06: the embedder, the cache probe and the installer's download
all pass :func:`model_revision`, so the three cannot load different weights.
A pinned commit writes no ``refs/main`` in the HF cache, which is why the probe
must ask for the revision explicitly rather than resolve ``main``.

Stdlib-only leaf: ``models.config`` and the ``embeddings`` package both import it.
"""

from __future__ import annotations

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
UNPINNED_REVISION = "main"

# Keyed by lowercased Hub id. The CI model cache key (trw-memory/.github/workflows/ci.yml)
# names the same commit.
_PINNED_REVISIONS = {
    "baai/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
}


def pinned_revision(model: str) -> str | None:
    """Return the pinned Hub commit for *model*, or ``None`` when it is unpinned."""
    return _PINNED_REVISIONS.get(model.lower())


def model_revision(model: str) -> str:
    """Return the revision to load *model* at: its pinned commit, else ``main``."""
    return pinned_revision(model) or UNPINNED_REVISION
