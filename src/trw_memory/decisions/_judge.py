"""The ``DecisionJudge`` seam: a swappable, never-raising decision backend.

Every call site that wants a calibrated decision talks to this Protocol, never
to a concrete backend. The default is :class:`NullJudge`, which answers
nothing and costs nothing — a caller must always have a deterministic or prose
fallback for ``None`` per the Jev decision-backend design ("every Jev
call site needs a deterministic or local fallback").
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import JsonValue

from trw_memory.decisions._models import DecisionQuestion, DecisionResult

#: State sent to a judge: a string, a JSON object, or a JSON array — exactly
#: the three shapes the OpenRouter Decisions API accepts as ``state``.
DecisionState = str | dict[str, JsonValue] | list[JsonValue]


@runtime_checkable
class DecisionJudge(Protocol):
    """A backend that answers typed questions about a state, or abstains.

    Implementations MUST NOT raise: every error — timeout, transport failure,
    malformed response, disabled config — is reported as ``None``, never an
    exception. Callers branch on ``None`` exactly like an abstention; there is
    no separate error channel to handle.
    """

    def decide(
        self,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
        *,
        timeout_s: float = 10.0,
        session_id: str | None = None,
    ) -> DecisionResult | None: ...


class NullJudge:
    """The default judge: always abstains, never touches the network.

    This is what every caller gets until an operator explicitly opts in via
    :func:`trw_memory.decisions.judge_from_env`. It exists as a concrete class
    (not just "pass no judge") so a caller can hold a ``DecisionJudge`` typed
    reference unconditionally and never branch on "is a judge configured".
    """

    def decide(
        self,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
        *,
        timeout_s: float = 10.0,
        session_id: str | None = None,
    ) -> DecisionResult | None:
        return None


__all__ = ["DecisionJudge", "DecisionState", "NullJudge"]
