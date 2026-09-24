"""trw-jev decision seam: opt-in, calibrated yes/no, choice and score answers.

**Responsibility.** A narrow, swappable interface (:class:`DecisionJudge`) for
asking a typed question about a piece of state and getting back a probability
rather than prose. The default backend (:class:`NullJudge`) always abstains
and touches no network; :class:`JevHttpJudge` is the one opt-in backend, wired
to TypeSafe's Jev model over OpenRouter's Decisions API.

**Interface.** Construct a judge with :func:`judge_from_env` (enablement via
:func:`resolve_backend_enablement` — process env beats project scope beats
user scope beats off; ``TRW_JEV_BASE_URL``/``TRW_JEV_MODEL`` come from the
PROCESS env only; ``OPENROUTER_API_KEY`` may also come from a project
``.env``; off by default) or instantiate :class:`JevHttpJudge` directly for
explicit control. Describe a decision with :class:`NoulQuestion`
(yes/no), :class:`ChoiceQuestion` (pick one of N) or :class:`ScoreQuestion`
(ordered rubric); call ``judge.decide(state, questions)``.

**Invariants.** ``decide()`` NEVER raises — every failure mode (disabled,
missing key, timeout, rate limit exhausted, malformed response) returns a
:class:`DecisionFailure` naming its kind, which a caller treats as an abstention. The API key and
the raw state are never logged; only site, latency, outcome and cost are.

**Knobs.** ``TRW_JEV_ENABLED`` (bool, default off), ``OPENROUTER_API_KEY``
(required to enable), ``TRW_JEV_BASE_URL`` (default
``https://openrouter.ai/api/alpha/decisions``; must be ``https`` on an
allowlisted host, else the call abstains), ``TRW_JEV_MODEL`` (default
``~typesafe/jev-latest`` — the leading ``~`` is required; the plain slug
returns HTTP 400 "does not exist" on the live OpenRouter Decisions API).

This package has no dependency on ``trw_llm`` or ``trw_mcp`` (bottom-layer
rule per the Jev decision-backend design) — only ``httpx``
and ``pydantic``, both already required by ``trw-memory``.
"""

from __future__ import annotations

from trw_memory.decisions._enablement import resolve_backend_enablement
from trw_memory.decisions._env import judge_from_env, toolkit_from_env
from trw_memory.decisions._judge import DecisionJudge, DecisionState, NullJudge
from trw_memory.decisions._models import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionFailure,
    DecisionOutcome,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from trw_memory.decisions._redaction import default_redactor, redact_state, unique_key
from trw_memory.decisions.toolkit import Toolkit

__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAnswer",
    "DecisionFailure",
    "DecisionJudge",
    "DecisionOutcome",
    "DecisionQuestion",
    "DecisionResult",
    "DecisionState",
    "NoulAnswer",
    "NoulQuestion",
    "NullJudge",
    "ScoreAnswer",
    "ScoreQuestion",
    "Toolkit",
    "default_redactor",
    "judge_from_env",
    "redact_state",
    "resolve_backend_enablement",
    "toolkit_from_env",
    "unique_key",
]
