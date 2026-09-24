"""Assertion + anchor verification pass and the maintain-verify sweep.

PRD-CORE-231 FR02/FR03, moved into trw-memory by PRD-CORE-294 FR07(b) so the
daemon's ``memory_maintain`` and trw-mcp's ``maintain-verify`` run *the same*
computation and *the same* single persisted write. The assertion-evaluation
primitives stay in :mod:`trw_memory.lifecycle.verification`.

Seams:

* :func:`run_verification_pass` -- reads the filesystem (assertions + anchors)
  and returns what should be persisted for one entry.
* :func:`persist_verification_outcome` -- the single ``backend.update()`` that
  writes ``assertions``, ``verification_status`` and ``anchor_validity``
  together, plus the NFR02 ``verification_status_persist_drift`` self-check.
* :func:`run_maintain_verify` -- the bounded keyset sweep over every entry with
  assertions or anchors. With ``project_root=None`` nothing can be checked, so
  every entry is left untouched: an unknown result is never recorded as passed.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

import structlog

from trw_memory.namespaces.validation import DEFAULT_NAMESPACE

logger = structlog.get_logger(__name__)

#: Knob defaults, equal to trw-mcp's ``TRWConfig`` defaults so the daemon and the
#: framework host produce the same verdict for the same tree.
DEFAULT_ASSERTION_FAILURE_PENALTY = 0.15
DEFAULT_ASSERTION_STALE_THRESHOLD_DAYS = 30
DEFAULT_ANCHOR_VALIDITY_VERIFIED_FLOOR = 1.0
DEFAULT_MAINTAIN_VERIFY_BATCH_LIMIT = 1000

#: PRD-CORE-244-FR03 widened this from ``Literal["stale"] | None``. ``None`` no
#: longer doubles as "healthy": a pass that examined an entry and found it clean
#: records ``"verified"``, and ``VerificationOutcome.checked_at`` stamps WHEN.
VerificationStatus = Literal["verified", "stale"] | None


@dataclass(slots=True)
class VerificationOutcome:
    """Everything one entry's verification pass computed."""

    entry_id: str
    #: The namespace that owns ``entry_id``. Part of the row's identity under
    #: trw-memory schema 5 (PRD-CORE-245 FR03), so ``persist_verification_outcome``
    #: cannot write the verdict without it — and before it was carried, that
    #: write raised a TypeError the best-effort handler swallowed, which meant
    #: the FR02/FR03 verdict never landed and the NFR02 drift tripwire could
    #: never fire.
    namespace: str = DEFAULT_NAMESPACE
    updated_assertions: list[dict[str, object]] = field(default_factory=list)
    assertion_status: dict[str, object] = field(default_factory=dict)
    passing: int = 0
    failing: int = 0
    stale: int = 0
    penalty: float = 0.0
    verification_status: VerificationStatus = None
    #: PRD-CORE-244-FR03: ISO-8601 stamp of THIS pass, set only when the pass
    #: actually examined something. "" leaves the persisted value untouched, so
    #: an entry nothing could be checked on never acquires a false exam record.
    checked_at: str = ""
    anchor_validity: float | None = None
    #: False when the entry HAS assertions but none could actually be checked
    #: (e.g. project_root unresolvable => every result is ``passed=None``).
    #: Nothing is persisted in that case: the historical ``first_failed_at``
    #: values would otherwise convict an entry that was never re-examined.
    verifiable: bool = True

    def update_fields(self) -> dict[str, object]:
        """The exact kwargs for the single batched ``backend.update()`` call.

        ``assertions`` is handed over as validated ``Assertion`` MODELS, not a
        pre-serialized JSON string: ``update()`` reconstructs the entry to
        recompute its sync hash, and a raw string there makes that
        reconstruction raise — which is how the FR06 write-back was silently
        dying inside the best-effort handler.
        """
        from trw_memory.models.memory import Assertion

        fields: dict[str, object] = {
            "assertions": [Assertion.model_validate(a, strict=False) for a in self.updated_assertions],
            # Scalar, last-write-wins: passing ``None`` is how a previously
            # persisted 'stale' or 'verified' verdict is cleared (FR02 AC2).
            "verification_status": self.verification_status,
        }
        if self.anchor_validity is not None:
            fields["anchor_validity"] = self.anchor_validity
        # PRD-CORE-244-FR03: the exam stamp rides the SAME batched update, so a
        # positive verdict and the time it was reached are one write, not two.
        if self.checked_at:
            fields["verification_checked_at"] = self.checked_at
        return fields


def _assertion_result_detail(
    entry_id: str,
    index: int,
    assertion: Any,
    result: Any,
) -> dict[str, object]:
    """Normalize verification result payloads to the recall response contract."""
    detail = cast("dict[str, object]", result.model_dump())
    detail["id"] = f"{entry_id}:{index}"
    detail.setdefault("type", getattr(assertion, "type", ""))
    detail.setdefault("pattern", getattr(assertion, "pattern", ""))
    detail.setdefault("target", getattr(assertion, "target", ""))
    return detail


def _reverify_anchors(
    raw_anchors: list[object],
    project_root: Path | None,
    entry_id: str,
) -> float | None:
    """Recompute anchor validity against the CURRENT tree (FR03).

    Returns ``None`` when nothing could be computed (no anchors, no resolvable
    project root, or a computation error) so the caller leaves the persisted
    write-time score untouched rather than overwriting it with a guess.
    """
    if not raw_anchors or project_root is None:
        return None
    try:
        from trw_memory.lifecycle.anchor_validation import compute_anchor_validity

        anchors = [a for a in raw_anchors if isinstance(a, dict)]
        if not anchors:
            return None
        return compute_anchor_validity(
            cast("list[dict[str, object]]", anchors),
            str(project_root),
        )
    except Exception:  # trw-fail-silent-allow: fail-open, re-verification is best-effort
        logger.debug("anchor_revalidation_skipped", entry_id=entry_id, exc_info=True)
        return None


def _apply_verdict(outcome: VerificationOutcome, *, moment: datetime, anchor_floor: float) -> None:
    """Stamp the exam and record a positive verdict (PRD-CORE-244 FR03).

    "Examined" is evidence-based rather than a flag: at least one assertion
    produced a real pass/fail, or an anchor score was actually recomputed. An
    entry where neither happened is left completely untouched — stamping it
    would make "never checked" indistinguishable from "checked and learned
    nothing", which is the conflation this FR exists to remove.

    ``"verified"`` requires no failing assertion AND (when anchors were scored)
    a score at or above *anchor_floor*: a partially drifted anchor set is not a
    clean bill of health. An already-computed ``"stale"`` verdict wins.
    """
    examined = (outcome.passing + outcome.failing) > 0 or outcome.anchor_validity is not None
    if not examined:
        return
    outcome.checked_at = moment.isoformat()
    if outcome.verification_status == "stale":
        return
    anchors_clean = outcome.anchor_validity is None or outcome.anchor_validity >= anchor_floor
    if outcome.failing == 0 and outcome.stale == 0 and anchors_clean:
        outcome.verification_status = "verified"


def run_verification_pass(
    entry_id: str,
    raw_assertions: list[object],
    raw_anchors: list[object],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    assertion_failure_penalty: float,
    assertion_stale_threshold_days: int,
    anchor_validity_verified_floor: float,
    project_root: Path | None,
    now: datetime | None = None,
) -> VerificationOutcome:
    """Verify one entry's assertions and re-verify its anchors.

    Args:
        entry_id: The learning/memory id (used for result ids).
        namespace: The namespace that owns *entry_id*; carried onto the outcome
            so the persist step can address the row (PRD-CORE-245 FR03).
        raw_assertions: Serialized assertion dicts from the stored entry.
        raw_anchors: Serialized anchor dicts from the stored entry.
        assertion_failure_penalty: Penalty scale for failing assertions.
        assertion_stale_threshold_days: Days of continuous failure before ``stale``.
        anchor_validity_verified_floor: The lowest recomputed anchor score a
            ``"verified"`` verdict tolerates (FR03).
        project_root: Repo root for filesystem-scoped verification, or ``None``.
        now: Injectable clock for tests; defaults to ``datetime.now(utc)``.

    Returns:
        A :class:`VerificationOutcome`. Never raises for a per-entry failure —
        the caller keeps scanning the remaining entries.
    """
    from trw_memory.lifecycle.verification import verify_assertions
    from trw_memory.models.memory import Assertion

    moment = now or datetime.now(timezone.utc)
    stale_threshold = moment - timedelta(days=assertion_stale_threshold_days)
    outcome = VerificationOutcome(entry_id=entry_id, namespace=namespace)
    outcome.anchor_validity = _reverify_anchors(raw_anchors, project_root, entry_id)

    if not raw_assertions:
        # Anchors-only entry: unavailable anchors must not erase prior evidence.
        outcome.verifiable = outcome.anchor_validity is not None
        # The anchor recomputation IS the examination.
        _apply_verdict(outcome, moment=moment, anchor_floor=anchor_validity_verified_floor)
        return outcome

    assertions_list = [Assertion.model_validate(a, strict=False) for a in raw_assertions if isinstance(a, dict)]
    results = verify_assertions(assertions_list, project_root)

    outcome.passing = sum(1 for r in results if r.passed is True)
    outcome.failing = sum(1 for r in results if r.passed is False)
    outcome.stale = sum(1 for r in results if r.passed is None)
    outcome.verifiable = any(r.passed is not None for r in results)
    outcome.assertion_status = {
        "passing": outcome.passing,
        "failing": outcome.failing,
        "stale": outcome.stale,
        "details": [
            _assertion_result_detail(entry_id, index, assertion, result)
            for index, (assertion, result) in enumerate(zip(assertions_list, results, strict=False), start=1)
        ],
    }
    if outcome.failing > 0 and results:
        outcome.penalty = assertion_failure_penalty * (outcome.failing / len(results))

    # FR06: fold verification results back into the stored assertion payload.
    for assertion, result in zip(assertions_list, results, strict=False):
        # mode="json" is load-bearing: a plain model_dump() leaves datetimes as
        # objects, so json.dumps() raised TypeError and the whole persist was
        # swallowed by the best-effort handler for any assertion that already
        # carried a first_failed_at — i.e. exactly the stale candidates.
        a_dict = assertion.model_dump(mode="json")
        if result.passed is not None:
            a_dict["last_result"] = result.passed
            a_dict["last_verified_at"] = moment.isoformat()
            a_dict["last_evidence"] = result.evidence
        # FR08: track first_failed_at transitions.
        if result.passed is False:
            if assertion.first_failed_at is None:
                a_dict["first_failed_at"] = moment.isoformat()
        elif result.passed is True:
            a_dict["first_failed_at"] = None
        outcome.updated_assertions.append(a_dict)

    # FR08: every assertion failing for longer than the threshold => stale.
    # An UNVERIFIABLE result (``passed is None``) is not a failure — requiring
    # every assertion to have actually failed on THIS run stops an unresolvable
    # project root from convicting an entry on nothing but historical timestamps.
    all_failing_now = bool(results) and all(r.passed is False for r in results)
    all_persistently_failing = (
        all_failing_now
        and len(outcome.updated_assertions) > 0
        and all(
            a.get("first_failed_at") is not None and datetime.fromisoformat(str(a["first_failed_at"])) < stale_threshold
            for a in outcome.updated_assertions
        )
    )
    if all_persistently_failing:
        outcome.verification_status = "stale"
    _apply_verdict(outcome, moment=moment, anchor_floor=anchor_validity_verified_floor)
    return outcome


def persist_verification_outcome(backend: Any, outcome: VerificationOutcome) -> bool:
    """Write an outcome through in ONE ``backend.update()`` call (FR02/FR03).

    Returns ``True`` when the write landed. Emits the NFR02
    ``verification_status_persist_drift`` warning when the value the caller
    computed is not the value that came back from storage — that warning firing
    means the persistence wiring is broken, which is a P1 bug, not a design gap.

    An outcome whose assertions could not be checked at all is skipped (DEBUG,
    no exception) — the PRD-CORE-086-FR09 degradation contract.
    """
    if not outcome.verifiable:
        logger.debug("verification_pass_unverifiable_skipped", entry_id=outcome.entry_id)
        return False
    try:
        updated = backend.update(outcome.entry_id, namespace=outcome.namespace, **outcome.update_fields())
    except Exception:  # justified: persist is best-effort, recall must not fail
        logger.debug("assertion_result_persist_failed", entry_id=outcome.entry_id, exc_info=True)
        return False

    # A backend that returns no real entry (test double, YAML shim) gives
    # nothing to compare against, so only a returned entry is checked.
    persisted = getattr(updated, "verification_status", outcome.verification_status)
    if persisted != outcome.verification_status:
        logger.warning(
            "verification_status_persist_drift",
            entry_id=outcome.entry_id,
            computed=outcome.verification_status,
            persisted=persisted,
        )
    return True


@dataclass(frozen=True, slots=True)
class VerifySettings:
    """The sweep's knobs, as one value a store method or a daemon tool can carry."""

    assertion_failure_penalty: float = DEFAULT_ASSERTION_FAILURE_PENALTY
    assertion_stale_threshold_days: int = DEFAULT_ASSERTION_STALE_THRESHOLD_DAYS
    anchor_validity_verified_floor: float = DEFAULT_ANCHOR_VALIDITY_VERIFIED_FLOOR
    batch_limit: int = DEFAULT_MAINTAIN_VERIFY_BATCH_LIMIT

    def __post_init__(self) -> None:
        # A daemon caller supplies these as JSON, so types and ranges are checked here, before the sweep.
        for name, low, high in (("assertion_failure_penalty", 0.0, 1.0), ("anchor_validity_verified_floor", 0.0, 1.0)):
            _check(name, getattr(self, name), (int, float), low, high)
        _check("assertion_stale_threshold_days", self.assertion_stale_threshold_days, (int,), 1, None)
        _check("batch_limit", self.batch_limit, (int,), 1, 100_000)


def _check(name: str, value: object, kinds: tuple[type, ...], low: float, high: float | None) -> None:
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise TypeError(f"{name} must be {' or '.join(k.__name__ for k in kinds)}, not {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):  # an int is finite (and may overflow isfinite)
        raise ValueError(f"{name} must be finite, not {value}")
    if value < low or (high is not None and value > high):  # type: ignore[operator]
        raise ValueError(f"{name}={value} is outside [{low}, {high if high is not None else 'inf'}]")


@dataclass(slots=True)
class MaintainVerifySummary:
    """Result of one sweep — the NFR04 audit record, in typed form."""

    entries_processed: int = 0
    stale_transitions: int = 0
    cleared_transitions: int = 0
    persist_failures: int = 0
    #: Entries whose verification raised; any makes the sweep a failure.
    entry_failures: int = 0
    #: Prior "verified" verdicts cleared because nothing could re-check them.
    invalidated: int = 0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        """Plain mapping for CLI/JSON output."""
        return asdict(self)


def _serialized(items: list[Any]) -> list[object]:
    """Normalize stored pydantic models to the dict shape the pass consumes."""
    out: list[object] = []
    for item in items:
        dump = getattr(item, "model_dump", None)
        out.append(dump() if callable(dump) else item)
    return out


def run_maintain_verify(
    backend: Any,
    *,
    project_root: Path | None,
    namespace: str | None = None,
    assertion_failure_penalty: float = DEFAULT_ASSERTION_FAILURE_PENALTY,
    assertion_stale_threshold_days: int = DEFAULT_ASSERTION_STALE_THRESHOLD_DAYS,
    anchor_validity_verified_floor: float = DEFAULT_ANCHOR_VALIDITY_VERIFIED_FLOOR,
    batch_limit: int = DEFAULT_MAINTAIN_VERIFY_BATCH_LIMIT,
) -> MaintainVerifySummary:
    """Verify every active assertion/anchor entry and persist observed verdicts.

    Uses bounded keyset ``entries_with_assertions`` pages (no N+1 acquisition
    fetch). A per-entry failure is counted in ``entry_failures`` and the sweep
    continues, so one bad row cannot abort it but cannot hide either. A prior
    ``"verified"`` verdict that this sweep cannot re-check (e.g. no project
    root) is cleared: a positive verdict needs current evidence.

    Args:
        backend: Memory backend exposing ``entries_with_assertions`` + ``update``.
        project_root: Repo root for filesystem-scoped verification. ``None``
            means nothing can be checked: no entry is verified and prior
            ``"verified"`` verdicts are cleared.
        namespace: Optional namespace scope; ``None`` sweeps every namespace.
        assertion_failure_penalty: See :func:`run_verification_pass`.
        assertion_stale_threshold_days: See :func:`run_verification_pass`.
        anchor_validity_verified_floor: See :func:`run_verification_pass`.
        batch_limit: Page size for each keyset acquisition.

    Returns:
        A :class:`MaintainVerifySummary` describing the sweep.
    """
    started = time.monotonic()
    summary = MaintainVerifySummary()

    cursor: tuple[str, str] | None = None
    while batch_limit > 0:
        entries = backend.entries_with_assertions(
            namespace=namespace, limit=batch_limit, include_anchors=True, after=cursor
        )
        if not entries:
            break
        next_cursor = (str(entries[-1].namespace), str(entries[-1].id))
        if cursor is not None and next_cursor <= cursor:
            raise ValueError("Maintenance backend did not advance its keyset cursor")
        # Advance independently of verification/persistence success.
        cursor = next_cursor
        for entry in entries:
            entry_id = str(getattr(entry, "id", ""))
            prior = getattr(entry, "verification_status", None)
            try:
                if not getattr(entry, "assertions", None) and not getattr(entry, "anchors", None):
                    # Selected for a non-empty stored column yet nothing decoded:
                    # the row mapper degraded a malformed payload to [].
                    raise ValueError("stored assertions/anchors did not decode")
                outcome = run_verification_pass(
                    entry_id,
                    _serialized(list(getattr(entry, "assertions", []) or [])),
                    _serialized(list(getattr(entry, "anchors", []) or [])),
                    namespace=str(getattr(entry, "namespace", namespace)),
                    assertion_failure_penalty=assertion_failure_penalty,
                    assertion_stale_threshold_days=assertion_stale_threshold_days,
                    anchor_validity_verified_floor=anchor_validity_verified_floor,
                    project_root=project_root,
                )
                # Nothing could be checked: never convict on historical
                # timestamps, and never let an old "verified" stand unexamined.
                if not outcome.verifiable and prior == "verified":
                    backend.update(entry_id, namespace=outcome.namespace, verification_status=None)
                    summary.invalidated += 1
            except Exception:  # justified: sweep-resilience, counted and surfaced via entry_failures
                logger.warning("maintain_verify_entry_failed", entry_id=entry_id, exc_info=True)
                summary.entry_failures += 1
                continue

            summary.entries_processed += 1
            if not outcome.verifiable:
                continue
            if not persist_verification_outcome(backend, outcome):
                summary.persist_failures += 1
                continue
            if outcome.verification_status == "stale" and prior != "stale":
                summary.stale_transitions += 1
            elif prior == "stale" and outcome.verification_status != "stale":
                # PRD-CORE-244 FR03: clearing a stale verdict now usually lands on
                # "verified" rather than None, so keying this on ``is None`` stopped
                # counting the very transition it exists to report.
                summary.cleared_transitions += 1

        if len(entries) < batch_limit:
            break

    summary.duration_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "maintain_verify_sweep_complete",
        entries_processed=summary.entries_processed,
        stale_transitions=summary.stale_transitions,
        cleared_transitions=summary.cleared_transitions,
        persist_failures=summary.persist_failures,
        entry_failures=summary.entry_failures,
        invalidated=summary.invalidated,
        duration_ms=summary.duration_ms,
    )
    return summary


def assertion_health(backend: Any, *, namespace: str, stale_days: int) -> dict[str, int] | None:
    """PRD-CORE-086 FR07: *namespace*'s assertion counts from the cached verdicts; ``None`` when none carries one.

    An assertion is stale once its last verification is older than *stale_days*,
    the same knob the sweep reads (PRD-CORE-263-FR08). The scan is the backend's
    bounded active-rows page, so a large store never costs a full-table read.
    """
    entries = backend.entries_with_assertions(namespace=namespace)
    if not entries:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
    counts = dict.fromkeys(("passing", "failing", "stale", "unverifiable"), 0)
    for entry in entries:
        for a in entry.assertions:
            if a.last_verified_at is None or a.last_verified_at < cutoff:
                counts["stale"] += 1
            else:
                counts[{True: "passing", False: "failing"}.get(a.last_result, "unverifiable")] += 1
    return {**counts, "total": len(entries)}


__all__ = [
    "DEFAULT_ANCHOR_VALIDITY_VERIFIED_FLOOR",
    "DEFAULT_ASSERTION_FAILURE_PENALTY",
    "DEFAULT_ASSERTION_STALE_THRESHOLD_DAYS",
    "DEFAULT_MAINTAIN_VERIFY_BATCH_LIMIT",
    "MaintainVerifySummary",
    "VerificationOutcome",
    "VerificationStatus",
    "assertion_health",
    "persist_verification_outcome",
    "run_maintain_verify",
    "run_verification_pass",
]
