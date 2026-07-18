"""PRD-SEC-001 Item 5 — 14-day observe-mode calibration clock.

The clock is a single YAML sidecar at
``<trw_dir>/memory/security/observe_start.yaml`` recording when a given
PRD (default ``PRD-SEC-001``) entered observe mode and the promotion-
review date. Idempotent — re-invocations read the existing file without
overwriting the original ``started_at`` timestamp.

Wired into :func:`trw_memory.security.trust_scorer.score_intake` via a
module-level ``_clock_started_for`` set so the disk read happens at most
once per process.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from trw_memory.storage.persistence import lock_for_rmw, write_yaml

__all__ = ["ObserveClockState", "read_observe_clock", "start_observe_clock"]

_LOG = structlog.get_logger(__name__)
_CLOCK_FILENAME = "observe_start.yaml"


class ObserveClockState(BaseModel):
    """Snapshot of the observe-mode calibration clock for one PRD."""

    model_config = ConfigDict(strict=True)

    started_at: str
    phase: str
    promotion_review_at: str
    prd: str


def _clock_path(trw_dir: Path) -> Path:
    return trw_dir / "memory" / "security" / _CLOCK_FILENAME


def read_observe_clock(trw_dir: Path) -> ObserveClockState | None:
    """Return the persisted clock state for *trw_dir*, or ``None`` if unset."""
    path = _clock_path(trw_dir)
    if not path.exists():
        return None
    try:
        yaml = YAML(typ="safe")
        # Fail open on a corrupt sidecar: unreadable (OSError), non-UTF-8 from a
        # torn/partial write (UnicodeDecodeError), or malformed YAML (YAMLError)
        # yields ``None`` ("clock unset") rather than raising into the intake
        # path (``trust_scorer.score_intake`` calls ``start_observe_clock`` ->
        # ``read_observe_clock`` on every first-of-process write). Mirrors the
        # namespace-sidecar seams in ``security/_runtime_quarantine`` and
        # ``integrations/_backend``. ``UnicodeDecodeError`` subclasses
        # ``ValueError``, not ``OSError``, so it must be listed explicitly.
        data = yaml.load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, YAMLError):
        _LOG.warning("observe_clock.read_failed", path=str(path), exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ObserveClockState.model_validate(data)
    except Exception:  # pragma: no cover — malformed sidecar
        _LOG.warning("observe_clock.invalid_sidecar", path=str(path), exc_info=True)
        return None


def start_observe_clock(
    trw_dir: Path,
    *,
    prd: str = "PRD-SEC-001",
    window_days: int = 14,
) -> ObserveClockState:
    """Idempotently record the start of a *window_days* observe-mode clock.

    If the sidecar already exists, the existing state is returned UNMODIFIED
    (including its original ``started_at``). Otherwise a new sidecar is
    written with ``started_at=<now>`` and ``promotion_review_at=<now +
    window_days>`` and then returned.
    """
    path = _clock_path(trw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for_rmw(path):
        existing = read_observe_clock(trw_dir)
        if existing is not None:
            return existing

        now = datetime.now(timezone.utc)
        review_at = now + timedelta(days=window_days)
        state = ObserveClockState(
            started_at=now.isoformat(),
            phase="observe",
            promotion_review_at=review_at.isoformat(),
            prd=prd,
        )
        write_yaml(path, dict(sorted(state.model_dump(mode="json").items())))
    _LOG.info(
        "observe_clock.started",
        prd=prd,
        started_at=state.started_at,
        promotion_review_at=state.promotion_review_at,
        window_days=window_days,
    )
    return state
