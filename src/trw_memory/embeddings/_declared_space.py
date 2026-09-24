"""Declared embedding space for a loaded encoder whose state cannot be measured.

The measured identity (``_runtime_identity`` + ``_loaded_state``) hashes the
loaded CPU tensors, so it is unavailable whenever the encoder runs on an
accelerator (Apple ``mps``, CUDA) or has a non-BERT shape. Without some identity
every vector such a machine writes would be unqualified, and a recall path that
refuses unqualified vectors would have no dense retrieval at all.

This module supplies the weaker, declared tier: the model id the provider was
asked to load, the Hub snapshot revision it resolved to (when the local cache
names one), the output dimension and the fixed document-encoding contract. It
is recorded by the producer at generation time, never inferred afterwards for a
stored vector, and its ``encoding`` prefix keeps it distinguishable from a
measured space: the two never compare equal, so a vector is only ever scored
against a query from the same tier, model and revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trw_memory.embeddings.provenance import EmbeddingSpace

__all__ = ["DECLARED_ENCODING_PREFIX", "declared_embedding_space", "snapshot_revision"]

DECLARED_ENCODING_PREFIX = "trw-declared-encoder-v1:"

#: Hugging Face cache layout: ``models--org--name/snapshots/<commit>/``.
_SNAPSHOTS_DIR = "snapshots"


def snapshot_revision(snapshot_path: str) -> str:
    """Return the Hub commit a cached snapshot directory names, else ``""``.

    A plain local model directory has no revision; its path is deliberately not
    used, because the same weights copied elsewhere must not change identity.
    """
    if not snapshot_path:
        return ""
    path = Path(snapshot_path)
    return path.name if path.parent.name == _SNAPSHOTS_DIR else ""


def declared_embedding_space(model_name: str, revision: str, dimensions: int) -> EmbeddingSpace:
    """Build the declared space for a normalized float32 document encoding."""
    contract = json.dumps(
        {
            "contract": "trw-declared-encoder-v1",
            "model": model_name,
            "revision": revision,
            "dimensions": dimensions,
            "document_input": "verbatim",
            "normalize_embeddings": True,
            "precision": "float32",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return EmbeddingSpace(
        artifact_sha256=hashlib.sha256(contract.encode("utf-8")).hexdigest(),
        encoding=f"{DECLARED_ENCODING_PREFIX}{model_name}",
        dimensions=dimensions,
        model_id=model_name,
    )
