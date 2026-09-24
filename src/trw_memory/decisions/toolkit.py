"""Decision toolkit — ``ask(state, questions)`` is the API; everything else builds on it.

The wire protocol is one state plus a map of typed questions, all answered in one pass, and
that is the shape worth exposing: batching every question about a state into one request is
~10x cheaper per question than asking singly (measured 2026-09-19: $0.0000032/question at 100
per call vs $0.0000341 alone, latency roughly flat), and question types mix freely. So this
module makes the batched, mixed call the default path and gives callers small builders
(:func:`noul`, :func:`choice`, :func:`score`) to assemble it, rather than one round-trip per
helper.

Rules encoded here, each from a measurement or a verified defect (trw-jev research, 2026-09-19):

* **Every requested id gets an outcome** — a typed answer or a typed :class:`DecisionFailure`.
  A partial result never masquerades as a complete one (``AskResult.status``).
* **Whole-request redaction** — state, instructions, criteria and embedded item text all pass
  through the redactor; a secret-named key hides its value whatever its type.
* **Caller errors are raised, provider errors are returned** — 256 options or a misspelt noul
  criteria key is a bug to fix now (:class:`InvalidRequest`); a 429 is a failure to handle.
* **Conveniences build on ask**: ``classify``/``score_one`` were removed (test-only, PRD-CORE-295-FR01) —
  call :meth:`Toolkit.ask` with :func:`choice`/:func:`score` and read the id back with
  :meth:`AskResult.choice`/:meth:`AskResult.score`.
* **Item batching is a separate executor** (:meth:`Toolkit.batch_items`) with two schemas,
  because it changes the ANSWERS: items embedded in their own question stayed within mean
  max|dp| 0.06 of solo calls; items in one shared keyed state drifted 0.18 with 7/28 label
  flips. Rank order survived either (learning L-P1gA).
* **Never act on a close label** — :meth:`Policy.decide` reads probability mass and margin.
* **Thresholds are per-task inputs**, never baked in: absolute scores do not transfer.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import TypeAdapter, ValidationError

from trw_memory.decisions._judge import DecisionJudge, DecisionState
from trw_memory.decisions._models import OVER_CEILING_HINT, DecisionFailure, DecisionQuestion, DecisionResult
from trw_memory.decisions._policy import Policy, reliability
from trw_memory.decisions._redaction import default_redactor, redact_state, unique_key
from trw_memory.decisions._results import (
    DEAD_BAND,
    MAX_CHOICE_OPTIONS,
    MIN_MARGIN,
    AskResult,
    BatchResult,
    BatchSchema,
    ClassifyResult,
    InvalidCriteria,
    InvalidRequest,
    Outcome,
    RankedItem,
    RankResult,
    ScoreResult,
    _wrong_type,
)

__all__ = [
    "DEAD_BAND",
    "MAX_CHOICE_OPTIONS",
    "MIN_MARGIN",
    "AskResult",
    "BatchResult",
    "BatchSchema",
    "ClassifyResult",
    "DecisionFailure",
    "InvalidCriteria",
    "InvalidRequest",
    "Policy",
    "RankResult",
    "RankedItem",
    "ScoreResult",
    "Toolkit",
    "choice",
    "noul",
    "reliability",
    "score",
]


def noul(instructions: str, *, true: str, false: str) -> dict[str, Any]:
    """A yes/no question. Both sides DESCRIPTIVE: terse criteria measured AUC 0.737 vs 0.955."""
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


def choice(instructions: str, options: Mapping[str, str]) -> dict[str, Any]:
    """One of N (N <= 255). ``options`` maps each label to a descriptive rubric, never a bare word."""
    if len(options) > MAX_CHOICE_OPTIONS:
        raise InvalidRequest(f"{len(options)} options exceeds the {MAX_CHOICE_OPTIONS} server cap")
    if not options:
        raise InvalidRequest("choice needs at least one option")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: Sequence[str]) -> dict[str, Any]:
    """An ordered rubric, lowest to highest. Vary ONE attribute, monotonically, in ONE unit:
    the answer is an expected index over the levels, so arithmetic on it treats every step as
    equal (learning L-o0Or)."""
    if len(levels) < 2:
        raise InvalidRequest("score needs at least two ordered levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


_QUESTIONS_ADAPTER: TypeAdapter[dict[str, DecisionQuestion]] = TypeAdapter(dict[str, DecisionQuestion])


class Toolkit:
    """Capability-shaped wrapper over any :class:`DecisionJudge`."""

    def __init__(
        self,
        judge: DecisionJudge,
        *,
        redactor: Callable[[str], str] | None = default_redactor,
        timeout_s: float = 10.0,
        session_id: str | None = None,
    ) -> None:
        self._judge = judge
        self._redactor = redactor
        self._timeout_s = timeout_s
        self._session_id = session_id

    # -- redaction ------------------------------------------------------------

    @property
    def judge(self) -> DecisionJudge:
        """The judge this toolkit asks (after any wrappers), e.g. to label the backend in provenance."""
        return self._judge

    def _redact(self, value: Any) -> Any:
        return redact_state(value, self._redactor) if self._redactor is not None else value

    def _redact_text(self, text: str) -> str:
        return self._redactor(text) if self._redactor is not None else text

    def _redact_question(self, question: Mapping[str, Any]) -> dict[str, Any]:
        """Redact every value AND every key in a question: instructions and criteria may be arbitrary
        JSON per the wire models, and a choice label can carry an email or a token as easily as a
        value. Redaction never changes the number of options (collision-safe keys)."""
        out = dict(question)
        if "instructions" in out:
            out["instructions"] = self._redact(out["instructions"])
        if out.get("criteria") is not None:
            criteria = self._redact(out["criteria"])
            if isinstance(out["criteria"], (Mapping, list)) and len(criteria) != len(out["criteria"]):
                raise InvalidRequest("redaction changed the criteria count; refusing to ask a different question")
            out["criteria"] = criteria
        return out

    # -- the primitive --------------------------------------------------------

    def _typed_questions(
        self, questions: Mapping[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, DecisionQuestion]]:
        plain = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v)) for k, v in questions.items()}
        try:
            return plain, _QUESTIONS_ADAPTER.validate_python(plain)
        except ValidationError as exc:
            # The message may echo the offending input; scrub it like any other question prose.
            first = self._redact_text(str(exc.errors()[0].get("msg", "")))
            raise InvalidRequest(f"invalid questions: {exc.error_count()} error(s); first: {first}") from exc

    def ask(
        self,
        state: DecisionState,
        questions: Mapping[str, Mapping[str, Any] | DecisionQuestion],
        *,
        timeout_s: float | None = None,
        session_id: str | None = None,
    ) -> AskResult:
        """One call, every question about ``state``, an outcome for every id.

        Raises :class:`InvalidRequest` for what the caller must fix (unknown type, empty
        criteria, over-cap options, noul criteria not keyed true/false). Returns failures for
        what the caller must handle (auth, rate limit, timeout, provider, disabled).
        """
        if not questions:
            raise InvalidRequest("ask needs at least one question")
        plain, typed = self._typed_questions(questions)
        for qid, q in typed.items():
            if q.type == "choice" and len(q.criteria) > MAX_CHOICE_OPTIONS:
                raise InvalidRequest(f"{qid!r}: {len(q.criteria)} options exceeds the {MAX_CHOICE_OPTIONS} cap")
            if q.type == "noul" and q.criteria and "true" not in q.criteria and "false" not in q.criteria:
                raise InvalidCriteria(
                    f"{qid!r}: noul criteria must use the keys 'true' and 'false'; got {sorted(q.criteria)}"
                )
        # Question ids are JSON keys in the POST body, so they egress like state does (release-verify
        # N1). They are redacted on the wire and mapped back, so the caller's own ids stay usable.
        wire_id: dict[str, str] = {}
        for qid in plain:
            wire_id[qid] = unique_key(self._redact_text(qid), set(wire_id.values()))
        wire_questions = {wire_id[k]: self._redact_question(v) for k, v in plain.items()}
        started = time.monotonic()
        outcome = self._judge.decide(
            self._redact(state),
            wire_questions,
            timeout_s=self._timeout_s if timeout_s is None else timeout_s,
            session_id=self._session_id if session_id is None else session_id,
        )
        if isinstance(outcome, DecisionFailure):
            elapsed = round((time.monotonic() - started) * 1000, 1)
            return AskResult(dict.fromkeys(typed, outcome), backend="", latency_ms=elapsed)
        # Validate against the question as SENT: redaction may have renamed choice labels, and the
        # backend scores those, so the answer maps to the wire labels, not the caller's originals.
        return AskResult(
            {qid: _match_answer(wire_id[qid], wire_questions[wire_id[qid]], outcome) for qid in typed},
            model=outcome.model,
            backend=outcome.backend,
            latency_ms=outcome.latency_ms,
            usage=outcome.usage,
        )

    # -- item batching ----------------------------------------------------------

    def batch_items(
        self,
        items: Mapping[str, DecisionState],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        schema: BatchSchema = "embedded",
        chunk_size: int = 100,
        context: DecisionState | None = None,
    ) -> BatchResult:
        """Ask the same questions (any mix of types) about each of many items, batched.

        ``schema`` decides where each item's text goes, and it changes the answers (measured
        2026-09-19 on 28 items vs solo calls):

        * ``"embedded"`` (default) — the item rides in its own question's instructions; state is
          the shared ``context``. Mean max|dp| 0.06 from solo, 3/28 top-label flips. Use when
          you will read the values.
        * ``"keyed"`` — all items share one state dict and each question points at its key.
          Cheaper when several questions share an item (item text sent once), but mean 0.18
          drift and 7/28 flips: the shared state induces set-relative judgement. Fine for an
          ordering.

        Either way every question names its own item; without that, four findings of obviously
        different severity scored 0.59–0.61 flat.

        ``context`` is shared state every question may consult (a string or a JSON object). In
        the embedded schema it IS the request state; in the keyed schema it travels under
        ``_context``. Cost: the embedded schema repeats each item's text once per question, so
        N items × Q questions sends the item text Q times — budget against the ~56.8k-token
        request ceiling.
        """
        if schema not in ("embedded", "keyed"):
            raise InvalidRequest(f"schema must be 'embedded' or 'keyed', got {schema!r}")
        if chunk_size < 1:
            raise InvalidRequest(f"chunk_size must be >= 1, got {chunk_size}")
        if not questions:
            raise InvalidRequest("batch_items needs at least one question")
        self._typed_questions(questions)  # a shape error is the caller's, before any item request is built
        keys = list(items)
        qids = list(questions)
        per_item: dict[str, AskResult] = {}
        chunk_of: dict[str, int] = {}
        pending = [keys[start : start + chunk_size] for start in range(0, len(keys), chunk_size)]
        while pending:
            batch = pending.pop(0)
            state, wire, lookup = self._item_request(batch, items, questions, qids, schema, context)
            result = self.ask(state, wire)
            if len(batch) > 1 and _over_ceiling(result):
                # A 400 over the token ceiling costs nothing: split and retry the halves, in order,
                # so only an item that is too big on its own fails, and it fails as a request error.
                half = len(batch) // 2
                pending[:0] = [batch[:half], batch[half:]]
                continue
            grouped: dict[str, dict[str, Outcome]] = {key: {} for key in batch}
            for wid, (key, qid) in lookup.items():
                grouped[key][qid] = result.outcomes[wid]
            chunk_index = len(set(chunk_of.values()))
            for key in batch:
                chunk_of[key] = chunk_index
                per_item[key] = AskResult(
                    grouped[key],
                    model=result.model,
                    backend=result.backend,
                    latency_ms=result.latency_ms,
                    usage=result.usage,
                )
        return BatchResult({key: per_item[key] for key in keys}, chunk_of, schema)

    def _item_request(
        self,
        batch: list[str],
        items: Mapping[str, DecisionState],
        questions: Mapping[str, Mapping[str, Any]],
        qids: list[str],
        schema: BatchSchema,
        context: DecisionState | None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, tuple[str, str]]]:
        """One chunk's request: its state, its wire questions, and wire id -> (item, question)."""
        # Generated wire ids: caller keys and question ids are unrestricted strings, so any
        # composite of them can collide and hand one item another item's answer. The lookup is
        # explicit and the ids never leave this class.
        lookup: dict[str, tuple[str, str]] = {}
        wire: dict[str, dict[str, Any]] = {}
        state: dict[str, Any]
        if schema == "keyed":
            state = {"items": {key: items[key] for key in batch}}
            if context is not None:
                state["_context"] = context
            for i, key in enumerate(batch):
                for j, qid in enumerate(qids):
                    wid = f"i{i}q{j}"
                    lookup[wid] = (key, qid)
                    wire[wid] = {
                        **questions[qid],
                        "instructions": (
                            f"{questions[qid]['instructions']}\n\n"
                            f'Answer ONLY about the item under state.items["{key}"]; ignore the other items.'
                        ),
                    }
        else:
            state = (
                dict(context)
                if isinstance(context, dict)
                else {"context": context or "Each question carries its own item."}
            )
            for i, key in enumerate(batch):
                item_text = json.dumps(self._redact(items[key]), default=str)
                for j, qid in enumerate(qids):
                    wid = f"i{i}q{j}"
                    lookup[wid] = (key, qid)
                    wire[wid] = {
                        **questions[qid],
                        "instructions": (
                            f"{questions[qid]['instructions']}\n\nAnswer ONLY about the item inside "
                            f'the <item> tags.\n<item id="{i}">{item_text}</item>'
                        ),
                    }
        return state, wire, lookup

    def rank(
        self,
        items: Mapping[str, DecisionState],
        *,
        instructions: str,
        criteria: Mapping[str, str],
        chunk_size: int = 100,
        schema: BatchSchema = "keyed",
    ) -> RankResult:
        """Order items by P(true), best first, reporting any item that got no answer.

        Default schema is ``"keyed"``: an ordering survived it (Spearman 0.88–0.93 vs solo) and
        it is the cheaper request. Pass ``schema="embedded"`` if you will read the probabilities.
        """
        if not criteria or ("true" not in criteria and "false" not in criteria):
            raise InvalidCriteria("noul criteria must use the keys 'true' and 'false'")
        q = {"type": "noul", "instructions": instructions, "criteria": dict(criteria)}
        batch = self.batch_items(items, {"q": q}, schema=schema, chunk_size=chunk_size)
        scored = {key: batch.per_item[key].noul("q") for key in items}
        ranked = [RankedItem(k, v, items[k], batch.chunk_of.get(k, 0)) for k, v in scored.items() if v is not None]
        ranked.sort(key=lambda r: r.probability, reverse=True)
        return RankResult(ranked, [k for k, v in scored.items() if v is None])


def _over_ceiling(result: AskResult) -> bool:
    """True when the whole call was refused for exceeding the request token ceiling."""
    return result.status == "failed" and any(
        f.kind == "invalid_request" and OVER_CEILING_HINT in f.detail for f in result.failures.values()
    )


def _finite_unit(values: Any) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)


def _match_answer(question_id: str, question: Mapping[str, Any], result: DecisionResult) -> Outcome:
    """An answer counts only if it is the requested type AND possible for the requested question.

    A choice naming an option that was never offered, a distribution over unknown labels, or a
    score outside the rubric would otherwise flow into Policy.decide and be acted on.
    """
    expected_type = question["type"]
    if question_id in result.malformed_ids:
        return DecisionFailure(kind="malformed_response", detail=f"answer for {question_id!r} failed validation")
    answer = result.answers.get(question_id)
    if answer is None:
        return DecisionFailure(kind="malformed_response", detail=f"answer for {question_id!r} missing from response")
    if answer.type != expected_type:
        return _wrong_type(question_id, answer, expected_type)
    bad = DecisionFailure(
        kind="malformed_response", detail=f"answer for {question_id!r} is not possible for the question asked"
    )
    if answer.type == "choice":
        offered = set(question["criteria"])
        if answer.choice not in offered or not set(answer.probabilities) <= offered:
            return bad
        if not _finite_unit(answer.probabilities.values()):
            return bad
    elif answer.type == "score":
        top = len(question["criteria"]) - 1
        if not math.isfinite(answer.score) or not 0.0 <= answer.score <= top:
            return bad
        if not _finite_unit(answer.probabilities.values()):
            return bad
    elif answer.type == "noul" and not _finite_unit([answer.noul]):
        return bad
    return answer
