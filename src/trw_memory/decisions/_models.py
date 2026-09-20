"""Typed wire models for the trw-jev decision seam (opt-in, off by default).

Question models mirror the OpenRouter Decisions API (``POST
/api/alpha/decisions``): a ``noul`` is a yes/no probability, a ``choice`` picks
one of N labeled options, and a ``score`` is a probability-weighted position on
an ordered rubric, per the Jev decision-backend wire protocol this mirrors.
Nothing here performs network I/O — that lives in
``_jev_http.py`` — so these models are safe to import from any caller that
wants to describe a decision without depending on httpx.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

#: A criterion / instruction value may be plain prose or, per the wire
#: protocol, arbitrary JSON that the transport layer stringifies before send.
CriterionValue = JsonValue


class NoulQuestion(BaseModel):
    """A yes/no question; the answer is a probability of ``true``.

    ``criteria`` is optional. OpenRouter requires BOTH ``true`` and ``false``
    keys when criteria are present at all — the transport layer normalizes a
    single-sided criteria dict by filling the missing side, so a caller may
    supply just the side it cares about.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: CriterionValue
    criteria: dict[str, CriterionValue] | None = None


class ChoiceQuestion(BaseModel):
    """Pick one of N labeled options; each option carries a rubric."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: CriterionValue
    criteria: dict[str, CriterionValue] = Field(min_length=1)


class ScoreQuestion(BaseModel):
    """An ordered rubric; the answer is a probability-weighted position on it."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["score"] = "score"
    instructions: CriterionValue
    criteria: list[CriterionValue] = Field(min_length=1)


#: Discriminated by ``type`` — matches the wire protocol's three question
#: shapes exactly, so a mapping of these round-trips through JSON unchanged.
DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class NoulAnswer(BaseModel):
    """``noul`` is the probability of ``true``, in ``[0, 1]``."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    """A picked option plus the full probability distribution over options."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    """A probability-weighted score plus the rubric legend it was scored against."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


DecisionAnswer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class DecisionResult(BaseModel):
    """The outcome of one judge call: every question answered in one pass.

    ``usage`` carries whatever the backend reports (``input_tokens``,
    ``output_tokens``, ``cost``) verbatim rather than a typed subset, because a
    backend beyond Jev may report different fields and this is a pass-through,
    not a billing record.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    model: str
    answers: dict[str, DecisionAnswer]
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    backend: str
    latency_ms: float = Field(ge=0.0)


__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAnswer",
    "DecisionQuestion",
    "DecisionResult",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
]
