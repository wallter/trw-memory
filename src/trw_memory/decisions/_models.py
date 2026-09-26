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

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator

#: A criterion / instruction value may be plain prose or, per the wire
#: protocol, arbitrary JSON that the transport layer stringifies before send.
CriterionValue = JsonValue


class _StrictModel(BaseModel):
    """Shared config for every wire model below: no coercion, no unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class InvalidRequest(ValueError):
    """The caller built a request the provider would reject or silently degrade. Fix it; do not retry."""


class InvalidCriteria(InvalidRequest):
    """A noul question was given criteria the transport would discard."""


class QuestionShapeError(InvalidRequest):
    """A question dict used a shape pydantic's own message can't turn into a useful fix on its own —
    a ``type`` tag no submodel would ever match (a bad tag makes the discriminated union fail before
    any submodel validator runs, so this is caught here, first). Named for the mistake, one line,
    with the field to use instead — see the 2026-09-24 ``trw_assess`` usage audit (worker-1, 38 calls).
    """


#: One compact question set covering all three types — the single example every question-shape
#: error and the tool description itself point to, so a caller sees the same shape everywhere.
QUESTION_EXAMPLE = (
    '{"q1": {"type": "noul", "instructions": "..", "criteria": {"true": "..", "false": ".."}}, '
    '"q2": {"type": "choice", "instructions": "..", "criteria": {"a": "..", "b": ".."}}, '
    '"q3": {"type": "score", "instructions": "..", "criteria": ["low", "high"]}}'
)

#: The three valid ``type`` tags. A tag outside this set (most often ``"screen"``, mistaken for a
#: batch-screening mode) fails the discriminated union with a generic "no matching tag" message
#: unless caught first, by name, in :func:`_check_type_tag`.
_KNOWN_TYPES = frozenset({"noul", "choice", "score"})

#: Field names worker calls used in place of the real ones, and a one-line example of the right
#: one (2026-09-24 usage audit).
_FIELD_RENAMES: dict[str, tuple[str, str]] = {
    "question": ("instructions", '"instructions": "the question text"'),
    "criterion": ("criteria", '"criteria": {"true": "..", "false": ".."}'),
}


def _reject_renamed_fields(data: Mapping[str, Any]) -> None:
    for wrong, (right, example) in _FIELD_RENAMES.items():
        if wrong in data and right not in data:
            raise ValueError(f"use {right!r}, not {wrong!r}, e.g. {example}")


def _check_type_tag(question_id: str, raw: Mapping[str, Any]) -> None:
    """Reject a ``type`` no submodel would match, before the union dispatch obscures which field is wrong."""
    qtype = raw.get("type")
    if qtype is None or (isinstance(qtype, str) and qtype in _KNOWN_TYPES):
        return
    if qtype == "screen":
        raise QuestionShapeError(
            f"questions.{question_id}.type: 'screen' is not a question type; batch screening goes "
            'through the items= argument, not a question "type", e.g. items={"key": "..state.."}. '
            'Use "noul", "choice" or "score" here.'
        )
    raise QuestionShapeError(f"questions.{question_id}.type: must be 'noul', 'choice' or 'score', got {qtype!r}.")


class NoulQuestion(_StrictModel):
    """A yes/no question; the answer is a probability of ``true``.

    ``criteria`` is optional. OpenRouter requires BOTH ``true`` and ``false``
    keys when criteria are present at all — the transport layer normalizes a
    single-sided criteria dict by filling the missing side, so a caller may
    supply just the side it cares about.
    """

    type: Literal["noul"] = "noul"
    instructions: CriterionValue
    criteria: dict[str, CriterionValue] | None = None

    @model_validator(mode="before")
    @classmethod
    def _check_shape(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _reject_renamed_fields(data)
            criteria = data.get("criteria")
            if isinstance(criteria, Mapping) and criteria and not set(criteria) <= {"true", "false"}:
                raise ValueError(
                    "noul criteria keys must be exactly 'true'/'false', got "
                    f'{sorted(map(str, criteria))}, e.g. "criteria": {{"true": "..", "false": ".."}}'
                )
        return data


class ChoiceQuestion(_StrictModel):
    """Pick one of N labeled options; each option carries a rubric."""

    type: Literal["choice"] = "choice"
    instructions: CriterionValue
    criteria: dict[str, CriterionValue] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _check_shape(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _reject_renamed_fields(data)
            if "options" in data and "criteria" not in data:
                raise ValueError(
                    "'options' is not a field; choice options go in 'criteria' as {option: rubric}, "
                    'e.g. "criteria": {"a": "rubric for a", "b": "rubric for b"}'
                )
        return data


class ScoreQuestion(_StrictModel):
    """An ordered rubric; the answer is a probability-weighted position on it."""

    type: Literal["score"] = "score"
    instructions: CriterionValue
    criteria: list[CriterionValue] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _check_shape(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _reject_renamed_fields(data)
        return data


#: Discriminated by ``type`` — matches the wire protocol's three question
#: shapes exactly, so a mapping of these round-trips through JSON unchanged.
DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

_QUESTION_ADAPTER: TypeAdapter[DecisionQuestion] = TypeAdapter(DecisionQuestion)
_QUESTIONS_ADAPTER: TypeAdapter[dict[str, DecisionQuestion]] = TypeAdapter(dict[str, DecisionQuestion])


def parse_question(question_id: str, raw: Any) -> DecisionQuestion:
    """Validate one question dict, raising :class:`QuestionShapeError` for a ``type`` tag no
    submodel would ever match (checked before the adapter, since the union dispatch never runs a
    submodel's own validator for an unmatched tag)."""
    if isinstance(raw, Mapping):
        _check_type_tag(question_id, raw)
    return _QUESTION_ADAPTER.validate_python(raw)


def parse_questions(questions: Mapping[str, Any]) -> dict[str, DecisionQuestion]:
    """Validate a caller's whole question mapping in one pass (one :class:`ValidationError` names
    every bad field across every question, same as validating them individually would scatter);
    a ``type`` tag no submodel matches is still caught per-question, first, by name."""
    for qid, raw in questions.items():
        if isinstance(raw, Mapping):
            _check_type_tag(str(qid), raw)
    return _QUESTIONS_ADAPTER.validate_python(dict(questions))


def format_validation_error(exc: ValidationError, questions: Mapping[str, Any]) -> str:
    """Render a discriminated-union :class:`ValidationError` as a caller-facing message: each
    error names its question id and field path, never the raw pydantic dump (which would echo the
    caller's own criteria/instructions values back at them), plus the one shared worked example."""
    paths = []
    for error in exc.errors()[:3]:
        loc = list(error["loc"])
        qid = str(loc[0]) if loc else "?"
        # A tagged union puts the question's own type tag second in the path; the caller never wrote that key.
        if len(loc) > 1 and isinstance(questions.get(qid), Mapping) and loc[1] == questions[qid].get("type"):
            del loc[1]
        msg = error["msg"].removeprefix("Value error, ")
        paths.append(f"questions.{'.'.join(str(part) for part in loc)}: {msg}")
    return f"invalid questions ({exc.error_count()} error(s)): {'; '.join(paths)}. Valid example: {QUESTION_EXAMPLE}"


class NoulAnswer(_StrictModel):
    """``noul`` is the probability of ``true``, in ``[0, 1]``."""

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(_StrictModel):
    """A picked option plus the full probability distribution over options."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


class ScoreAnswer(_StrictModel):
    """A probability-weighted score plus the rubric legend it was scored against."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


DecisionAnswer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


#: Substring of a failure's ``detail`` when a request was refused for exceeding the token
#: ceiling. Lives here (not ``_jev_http.py``) so ``toolkit.py`` can test for it without importing
#: httpx on the disabled path.
OVER_CEILING_HINT = "request exceeds the token ceiling; trim state or split the batch"


#: Why a judge produced no answer. ``invalid_request`` and ``auth`` are the caller's
#: to fix; ``rate_limited``/``timeout``/``provider_error`` are transient or upstream;
#: ``disabled`` is the configured-off path; ``malformed_response`` is a 200 whose body
#: could not be trusted.
FailureKind = Literal[
    "invalid_request",
    "auth",
    "rate_limited",
    "timeout",
    "provider_error",
    "disabled",
    "malformed_response",
]


class DecisionFailure(_StrictModel):
    """Why no answer came back. A caller must be able to tell its own bug from an outage.

    ``detail`` is a short, non-echoing description: a status code and error type, never
    the state or the response body, both of which may carry the caller's data.
    """

    kind: FailureKind
    detail: str = ""

    @property
    def retryable(self) -> bool:
        return self.kind in ("rate_limited", "timeout", "provider_error")


class DecisionResult(_StrictModel):
    """The outcome of one judge call: every question answered in one pass.

    ``usage`` carries whatever the backend reports (``input_tokens``,
    ``output_tokens``, ``cost``) verbatim rather than a typed subset, because a
    backend beyond Jev may report different fields and this is a pass-through,
    not a billing record.
    """

    model: str
    answers: dict[str, DecisionAnswer]
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    backend: str
    latency_ms: float = Field(ge=0.0)
    #: Question ids whose answer member failed validation. The envelope was fine and the valid
    #: siblings are in ``answers``; one bad member must not erase them.
    malformed_ids: list[str] = Field(default_factory=list)
    #: Exception class name from the first malformed member (type only, never the value).
    malformed_error_type: str = ""


#: What a detailed decide returns: every question answered, or one reason nothing was.
DecisionOutcome = DecisionResult | DecisionFailure


__all__ = [
    "OVER_CEILING_HINT",
    "QUESTION_EXAMPLE",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAnswer",
    "DecisionFailure",
    "DecisionOutcome",
    "DecisionQuestion",
    "DecisionResult",
    "FailureKind",
    "InvalidCriteria",
    "InvalidRequest",
    "NoulAnswer",
    "NoulQuestion",
    "QuestionShapeError",
    "ScoreAnswer",
    "ScoreQuestion",
    "format_validation_error",
    "parse_question",
    "parse_questions",
]
