"""Wire encoding/decoding for the OpenRouter Decisions API.

Pure functions, no I/O — kept separate from ``_jev_http.py`` so the payload
shape and the transport (retries, timeouts) are independently testable.
Protocol reference: the Jev decision-backend design (OpenRouter Decisions API).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue, TypeAdapter

from trw_memory.decisions._judge import DecisionState
from trw_memory.decisions._models import (
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResult,
    NoulQuestion,
    ScoreQuestion,
)

_ANSWER_ADAPTER: TypeAdapter[DecisionAnswer] = TypeAdapter(DecisionAnswer)
_QUESTION_ADAPTER: TypeAdapter[DecisionQuestion] = TypeAdapter(DecisionQuestion)


def _stringify(value: JsonValue) -> str:
    """OpenRouter validates ``instructions``/criteria values as strings — a
    non-string value (structured guidance) is JSON-encoded, per the primer."""
    return value if isinstance(value, str) else json.dumps(value)


def normalize_noul_criteria(criteria: Mapping[str, JsonValue] | None) -> dict[str, JsonValue] | None:
    """Fill the missing side of a one-sided noul ``criteria`` dict.

    OpenRouter requires BOTH ``true`` and ``false`` keys whenever ``criteria``
    is present at all. A caller who only cares about describing one side would
    otherwise have their question rejected by the transport; this fills the
    other side with a neutral, generated description instead.
    """
    if criteria is None or len(criteria) == 0:
        return None
    true_value = criteria.get("true")
    false_value = criteria.get("false")
    if true_value is None and false_value is None:
        # Sending the question with no rubric would still return a plausible probability;
        # descriptive criteria are the largest measured quality lever (AUC 0.955 vs 0.737),
        # so a misspelt key is a caller error, not something to paper over (learning L-BPZq).
        raise ValueError(
            f"noul criteria must use the keys 'true' and 'false'; got {sorted(map(str, criteria))}. "
            "Other keys would be discarded on the wire, leaving the question with no rubric."
        )
    if true_value is None:
        true_value = f"Not: {_stringify(false_value)}"
    if false_value is None:
        false_value = f"Not: {_stringify(true_value)}"
    return {"true": true_value, "false": false_value}


def question_to_wire(question: DecisionQuestion) -> dict[str, Any]:
    """Render one typed question as its OpenRouter wire dict."""
    wire: dict[str, Any] = {"type": question.type, "instructions": _stringify(question.instructions)}
    if isinstance(question, NoulQuestion):
        normalized = normalize_noul_criteria(question.criteria)
        if normalized is not None:
            wire["criteria"] = {key: _stringify(value) for key, value in normalized.items()}
    elif isinstance(question, ChoiceQuestion):
        wire["criteria"] = {key: _stringify(value) for key, value in question.criteria.items()}
    elif isinstance(question, ScoreQuestion):
        wire["criteria"] = [_stringify(value) for value in question.criteria]
    return wire


def build_payload(
    model: str,
    state: DecisionState,
    questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
    session_id: str | None,
) -> dict[str, Any]:
    """Build the full ``POST /api/alpha/decisions`` request body.

    Plain-dict questions (the wire shape a library caller naturally writes) are
    validated into the typed models here, so they are not silently rejected.
    """
    typed: dict[str, DecisionQuestion] = {
        qid: _QUESTION_ADAPTER.validate_python(q) if isinstance(q, Mapping) else q for qid, q in questions.items()
    }
    payload: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": {question_id: question_to_wire(question) for question_id, question in typed.items()},
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def parse_response(body: Mapping[str, Any], *, backend: str, latency_ms: float) -> DecisionResult:
    """Parse a decoded JSON response body into a typed :class:`DecisionResult`."""
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise TypeError("response has no answers object")
    answers: dict[str, DecisionAnswer] = {}
    malformed: list[str] = []
    error_type = ""
    for question_id, raw in raw_answers.items():
        try:
            answers[str(question_id)] = _ANSWER_ADAPTER.validate_python(raw)
        except Exception as exc:  # trw:intentional one bad member is reported by id, not allowed to sink the siblings
            malformed.append(str(question_id))
            error_type = error_type or type(exc).__name__
    usage = body.get("usage") or {}
    return DecisionResult(
        model=str(body.get("model", "")),
        answers=answers,
        usage=dict(usage),
        backend=backend,
        latency_ms=latency_ms,
        malformed_ids=malformed,
        malformed_error_type=error_type,
    )


__all__ = ["build_payload", "normalize_noul_criteria", "parse_response", "question_to_wire"]
