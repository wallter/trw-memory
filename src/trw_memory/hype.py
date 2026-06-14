"""HyPE — hypothetical-question generation seam (PRD-CORE-195).

trw-memory ships **no** LLM dependency. HyPE (index-time hypothetical-question
expansion) needs a way to turn a stored entry into a small set of questions the
entry could answer; that generation is the caller's concern. This module
defines the structural contract for such a generator and a default no-op
implementation so the engine works out of the box with zero questions.

A caller wires a real generator (e.g. an Ollama-backed adapter) by passing any
object that implements :class:`QuestionGenerator` to
``MemoryClient(question_generator=...)``. The engine never imports or requires
an LLM runtime to satisfy this contract.

Design mirrors :class:`trw_memory.embeddings.interface.EmbeddingProvider`: a
``@runtime_checkable`` Protocol plus a default implementation, injected at the
client boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry

__all__ = ["NoOpQuestionGenerator", "QuestionGenerator"]


@runtime_checkable
class QuestionGenerator(Protocol):
    """Structural protocol for HyPE hypothetical-question generators.

    Any object exposing a ``generate(entry) -> list[str]`` method satisfies
    this protocol; subclassing is not required. The protocol is
    ``@runtime_checkable`` so ``isinstance(obj, QuestionGenerator)`` works for
    dependency-injection checks at the client boundary.
    """

    def generate(self, entry: MemoryEntry) -> list[str]:
        """Return hypothetical questions that *entry* could answer.

        Args:
            entry: The stored memory entry to derive questions from. The
                generator typically reads ``entry.content`` / ``entry.detail``.

        Returns:
            A list of question strings (possibly empty). The store path embeds
            and stores each as a secondary retrieval vector, after filtering by
            ``hype_min_question_chars`` and capping at
            ``hype_questions_per_entry``. Generators need not deduplicate or
            length-filter; the engine does both.
        """
        ...


class NoOpQuestionGenerator:
    """Default :class:`QuestionGenerator` that produces no questions.

    Bound by ``MemoryClient`` when no generator is injected, keeping the engine
    LLM-free: with HyPE enabled but this generator active, ``store`` writes the
    primary vector exactly as today and zero HyPE siblings.
    """

    def generate(self, entry: MemoryEntry) -> list[str]:
        """Return an empty list — no hypothetical questions are generated."""
        return []
