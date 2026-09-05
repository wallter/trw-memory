"""HyPE store-path expansion (PRD-CORE-195 FR03 + FR05 update purge).

Belongs to ``_client_store.py``. Kept here so the store module stays under the
effective-LOC gate and the HyPE branch is independently testable.

Single entry point :func:`expand_hype_siblings` runs INSIDE the caller's open
``backend.transaction()`` block, after the primary vector upsert:

1. (FR05) purge any existing ``{parent_id}#hype{n}`` siblings — overwrite, not
   append, so re-stores never accumulate stale duplicates.
2. generate hypothetical questions via the injected generator,
3. dedup + length-filter + cap at ``hype_questions_per_entry``,
4. embed each kept question with the SAME embedder as the primary vector,
5. upsert each as a secondary ``{parent_id}#hype{n}`` vector (skip_commit=True so
   it batches into the caller's outermost COMMIT).

The whole body is wrapped fail-open: any exception logs ``hype_generation_failed``
at warning and returns, leaving the canonical row + primary vector to commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from trw_memory._hype_ids import (
    HYPE_ID_SEPARATOR as HYPE_ID_SEPARATOR,
)
from trw_memory._hype_ids import (
    hype_sibling_id as hype_sibling_id,
)
from trw_memory._hype_ids import (
    is_hype_id as is_hype_id,
)
from trw_memory._hype_ids import (
    parent_of_hype_id as parent_of_hype_id,
)
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)


def _filter_questions(questions: list[str], *, min_chars: int, cap: int) -> list[str]:
    """Dedup (order-preserving), drop short/blank questions, cap the count."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw in questions:
        question = raw.strip()
        if len(question) < min_chars:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(question)
        if len(kept) >= cap:
            break
    return kept


def expand_hype_siblings(
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    entry: MemoryEntry,
    embedder: EmbeddingProvider | None,
    generator: Any,
) -> None:
    """Generate + store HyPE sibling vectors for *entry* (fail-open).

    Must be called INSIDE an open ``backend.transaction()`` block so sibling
    upserts batch into the caller's outermost COMMIT. A no-op when HyPE is
    disabled, no embedding consumer exists, or the generator yields nothing.

    Boundary semantics (FR05): existing siblings are purged FIRST, so an UPDATE
    overwrites rather than appends.
    """
    if not config.hype_enabled:
        return
    try:
        # FR05: purge-then-regenerate. Idempotent; no-op without vec support.
        backend.delete_hype_siblings(entry.id)

        if embedder is None:
            return
        questions = generator.generate(entry)
        if not questions:
            logger.debug("hype_expansion_complete", op="store", parent_id=entry.id, generated=0, stored=0, skipped=0)
            return
        kept = _filter_questions(
            questions,
            min_chars=config.hype_min_question_chars,
            cap=config.hype_questions_per_entry,
        )
        stored = 0
        for index, question in enumerate(kept):
            sibling_id = hype_sibling_id(entry.id, index)
            if backend.get(sibling_id, namespace=entry.namespace) is not None:
                logger.warning("hype_sibling_id_collision", op="store", parent_id=entry.id, sibling_id=sibling_id)
                continue
            q_embedding = embedder.embed(question)
            if q_embedding is None:
                continue
            backend.upsert_vector(sibling_id, q_embedding, namespace=entry.namespace)
            stored += 1
        logger.debug(
            "hype_expansion_complete",
            op="store",
            parent_id=entry.id,
            generated=len(questions),
            stored=stored,
            skipped=len(questions) - stored,
        )
    except Exception:
        # Fail-open: the canonical row + primary vector still commit. Never log
        # question text or entry content (may carry secrets) — structural only.
        logger.warning(
            "hype_generation_failed",
            op="store",
            outcome="failure",
            parent_id=entry.id,
            exc_info=True,
        )
