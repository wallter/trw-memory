"""Model-aware cosine thresholds for document-to-document similarity decisions.

Every similarity threshold in trw-memory and trw-mcp (dedup skip/merge, graph
similarity edges, consolidation clusters, cross-project validation, shared and
remote dedup, recall near-duplicate collapse, skill duplicates) was tuned on
``all-MiniLM-L6-v2`` cosines. Encoders do not share a cosine scale:
``bge-small-en-v1.5`` scores unrelated learnings around 0.6 where MiniLM scores
them around 0.2, so a MiniLM value applied to bge vectors merges and links
learnings that MiniLM kept apart.

Thresholds are therefore stated on the MiniLM REFERENCE scale -- defaults and
explicit config values alike -- and :func:`calibrated_threshold` translates one
into the scale of the encoder that produced the vectors being compared. The map
is monotone and piecewise linear through measured anchors; a reference value
between anchors is interpolated. MiniLM itself, any model absent from the table,
and an unidentifiable encoder keep the value unchanged, which is the behaviour
before this table existed. Values outside [-1, 1] (for example a threshold set
above 1.0 to disable a decision) are returned unchanged.

How the bge anchors were measured (2026-09-18): 512 labelled learning pairs --
90 near-exact edits, 73 paraphrases, 168 same-topic distinct, 160 unrelated, 21
templated distinct learnings -- scored under both encoders. For each reference
threshold the bge value is the larger of (a) the lowest threshold at which bge
precision on duplicates (paraphrase + near-exact; near-exact only for the 0.95
skip) is at least MiniLM's precision at the reference value, and (b) the highest
threshold at which bge recall is at least MiniLM's. The 0.0 and 0.5 anchors
match the share of all pairs at or above the value. The per-pair scores are in
``tests/data/similarity_calibration_scores.json`` and
``tests/test_similarity_calibration.py`` re-derives every anchor from them.
"""

from __future__ import annotations

from bisect import bisect_right

from trw_memory.embeddings._declared_space import DECLARED_ENCODING_PREFIX
from trw_memory.embeddings.provenance import EmbeddingSpace

__all__ = ["REFERENCE_MODEL", "calibrated_threshold", "encoder_model", "register_space_model"]

#: The encoder every threshold default is stated for.
REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Model id (lower-case) -> ascending (reference cosine, model cosine) anchors.
#: bge maps 0.75 and 0.85 to the same value: short templated distinct learnings
#: score up to 0.886 under bge, so precision caps both there (recall would allow
#: 0.805 for 0.75); the precision floor wins, the higher of the two.
_ANCHORS: dict[str, tuple[tuple[float, float], ...]] = {
    "baai/bge-small-en-v1.5": (
        (-1.0, -1.0),
        (0.0, 0.495),
        (0.5, 0.745),
        (0.75, 0.89),
        (0.85, 0.89),
        (0.90, 0.945),
        (0.92, 0.955),
        (0.95, 0.98),
        (1.0, 1.0),
    ),
}

#: Embedding spaces this process has loaded, mapped to the model that encodes
#: into them, so a caller holding only a vector's space can still be calibrated.
_SPACE_MODELS: dict[EmbeddingSpace, str] = {}


def register_space_model(space: EmbeddingSpace, model_name: str) -> None:
    """Record that *model_name* encodes into *space* (called by the provider on load)."""
    _SPACE_MODELS[space] = model_name


def encoder_model(encoder: object | None) -> str | None:
    """Model id of *encoder*: a model id, a provider with ``model_name``, or a loaded space."""
    if isinstance(encoder, str):
        return encoder
    name = getattr(encoder, "model_name", None)
    if isinstance(name, str):
        return name
    if not isinstance(encoder, EmbeddingSpace):
        return None
    registered = _SPACE_MODELS.get(encoder)
    if registered is not None:
        return registered
    # A declared space names its model in its encoding, so a process that never
    # loaded that encoder (a reader of another process's vectors) still knows it.
    if encoder.encoding.startswith(DECLARED_ENCODING_PREFIX):
        return encoder.encoding[len(DECLARED_ENCODING_PREFIX) :] or None
    return None


def _anchors(model_name: str | None) -> tuple[tuple[float, float], ...] | None:
    if not model_name:
        return None
    key = model_name.strip().lower()
    if key in _ANCHORS:
        return _ANCHORS[key]
    for known, anchors in _ANCHORS.items():
        if key == known.rsplit("/", 1)[-1]:
            return anchors
    return None


def calibrated_threshold(threshold: float, encoder: object | None) -> float:
    """Translate a reference-scale *threshold* to the scale of *encoder*'s vectors.

    *encoder* is whatever identifies the vectors' producer at the call site: an
    embedding provider, its model id, or the ``EmbeddingSpace`` the vectors were
    recorded in. Unknown or reference encoders return *threshold* unchanged.
    """
    anchors = _anchors(encoder_model(encoder))
    if anchors is None or not -1.0 <= threshold <= 1.0:
        return threshold
    refs = [ref for ref, _ in anchors]
    i = min(max(bisect_right(refs, threshold) - 1, 0), len(anchors) - 2)
    (x0, y0), (x1, y1) = anchors[i], anchors[i + 1]
    return round(y0 + (y1 - y0) * (threshold - x0) / (x1 - x0), 4)
