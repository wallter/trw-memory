"""Result types and caller-error exceptions for the decision toolkit.

Belongs to the ``toolkit.py`` facade; every public name here is re-exported there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from trw_memory.decisions._models import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionFailure,
    NoulAnswer,
    ScoreAnswer,
)

#: Server-enforced ceiling; 256+ returns 400 "Too many choices. Must have at most 255 choices."
MAX_CHOICE_OPTIONS = 255

#: Measured repeat SD 0.012; a threshold decision inside this band is not distinguishable.
DEAD_BAND = 0.03

#: 1 in 10 identical repeat sets returned a different top label; below this gap, do not act.
MIN_MARGIN = 0.05

#: Where a batched item's text travels. See :meth:`Toolkit.batch_items`.
BatchSchema = Literal["embedded", "keyed"]

Route = Literal["act", "escalate", "abstain"]


class InvalidRequest(ValueError):
    """The caller built a request the provider would reject or silently degrade. Fix it; do not retry."""


class InvalidCriteria(InvalidRequest):
    """A noul question was given criteria the transport would discard."""


@dataclass(frozen=True)
class ClassifyResult:
    label: str | None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    failure: DecisionFailure | None = None

    @property
    def answered(self) -> bool:
        return self.label is not None

    @property
    def margin(self) -> float:
        """Gap from the CHOSEN label to its nearest rival (may be negative if they disagree)."""
        if not self.probabilities:
            return 0.0
        if self.label is None or self.label not in self.probabilities:
            ranked = sorted(self.probabilities.values(), reverse=True)
            return ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        chosen = self.probabilities[self.label]
        rivals = [p for key, p in self.probabilities.items() if key != self.label]
        return chosen - max(rivals) if rivals else chosen

    def is_decisive(self, *, min_probability: float, min_margin: float = MIN_MARGIN) -> bool:
        if not self.answered:
            return False
        return self.probabilities.get(self.label or "", 0.0) >= min_probability and self.margin >= min_margin


@dataclass(frozen=True)
class ScoreResult:
    score: float | None
    legend: Mapping[str, Any] = field(default_factory=dict)
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    failure: DecisionFailure | None = None

    @property
    def answered(self) -> bool:
        return self.score is not None


Outcome = DecisionAnswer | DecisionFailure


@dataclass(frozen=True)
class AskResult:
    """One outcome per requested question id, plus the call's provenance.

    ``status`` is ``complete`` only when every id has a typed answer. A caller that wants to
    ignore a partial result has to do so on purpose.
    """

    outcomes: dict[str, Outcome]
    model: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def answers(self) -> dict[str, DecisionAnswer]:
        return {k: v for k, v in self.outcomes.items() if not isinstance(v, DecisionFailure)}

    @property
    def failures(self) -> dict[str, DecisionFailure]:
        return {k: v for k, v in self.outcomes.items() if isinstance(v, DecisionFailure)}

    @property
    def status(self) -> Literal["complete", "partial", "failed"]:
        if not self.answers:
            return "failed"
        return "complete" if not self.failures else "partial"

    def noul(self, question_id: str) -> float | None:
        answer = self.outcomes.get(question_id)
        return answer.noul if isinstance(answer, NoulAnswer) else None

    def choice(self, question_id: str) -> ClassifyResult:
        answer = self.outcomes.get(question_id)
        if isinstance(answer, ChoiceAnswer):
            return ClassifyResult(answer.choice, answer.probabilities, answer.confidence)
        failure = answer if isinstance(answer, DecisionFailure) else _wrong_type(question_id, answer, "choice")
        return ClassifyResult(None, failure=failure)

    def score(self, question_id: str) -> ScoreResult:
        answer = self.outcomes.get(question_id)
        if isinstance(answer, ScoreAnswer):
            return ScoreResult(answer.score, answer.legend, answer.probabilities, answer.confidence)
        failure = answer if isinstance(answer, DecisionFailure) else _wrong_type(question_id, answer, "score")
        return ScoreResult(None, failure=failure)

    def to_wire(self) -> dict[str, Any]:
        """A JSON-friendly rendering: answers as dicts, failures as ``{"failure": {...}}``."""
        out: dict[str, Any] = {}
        for key, value in self.outcomes.items():
            if isinstance(value, DecisionFailure):
                out[key] = {"failure": value.model_dump()}
            else:
                rendered = value.model_dump()
                if isinstance(value, ChoiceAnswer):
                    rendered["margin"] = round(self.choice(key).margin, 4)
                out[key] = rendered
        return out


def _wrong_type(question_id: str, answer: object, expected: str) -> DecisionFailure:
    got = "missing" if answer is None else getattr(answer, "type", type(answer).__name__)
    return DecisionFailure(
        kind="malformed_response", detail=f"expected a {expected} answer for {question_id!r}, got {got}"
    )


@dataclass(frozen=True)
class RankedItem:
    key: str
    probability: float
    item: Any
    #: Which request this item was scored in. Items in different chunks were judged in
    #: different company; a global sort across chunks is a convenience, not a validated scale.
    chunk: int = 0


@dataclass(frozen=True)
class RankResult:
    ranked: list[RankedItem]
    unanswered: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unanswered

    @property
    def status(self) -> Literal["complete", "partial", "failed"]:
        if not self.ranked:
            return "failed"
        return "complete" if self.complete else "partial"


@dataclass(frozen=True)
class BatchResult:
    """``batch_items`` output: an :class:`AskResult` per item, keyed by the caller's item key."""

    per_item: dict[str, AskResult]
    chunk_of: dict[str, int]
    schema: BatchSchema

    @property
    def unanswered(self) -> list[str]:
        return [k for k, r in self.per_item.items() if r.status == "failed"]

    @property
    def status(self) -> Literal["complete", "partial", "failed"]:
        statuses = {r.status for r in self.per_item.values()}
        if statuses <= {"failed"}:
            return "failed"
        return "complete" if statuses == {"complete"} else "partial"
