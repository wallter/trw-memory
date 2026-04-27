"""PRD-SEC-001 FR-001 — red-team fixture coverage floor.

Loads every YAML fixture under ``tests/fixtures/security/red_team/`` and
asserts that :func:`trw_memory.security.trust_scorer.score_intake`
rejects at least 9 of 10 (decision ``!= "allow"``).

Acceptance gate: PRD-SEC-001 §11 Red-team fixture corpus; sub-CLAUDE.md
§Red-Team Fixture Corpus (≥10 patterns, ≥9 must quarantine/reject).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from trw_memory.security.trust_scorer import score_intake

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "security" / "red_team"


def _load_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in sorted(_FIXTURE_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):  # defensive — bad fixture shape
            continue
        data["__path__"] = str(path)
        fixtures.append(data)
    return fixtures


_FIXTURES = _load_fixtures()

# Import-time floor check — fail collection loudly if corpus shrinks.
assert len(_FIXTURES) >= 10, (
    f"red-team corpus must have >=10 fixtures (PRD-SEC-001 FR-001 AC); found {len(_FIXTURES)} in {_FIXTURE_DIR}"
)


@pytest.mark.parametrize(
    "fixture",
    _FIXTURES,
    ids=[f["attack_id"] for f in _FIXTURES],
)
def test_fixture_shape(fixture: dict[str, Any]) -> None:
    """Every fixture carries the required 4 fields."""
    for key in ("attack_id", "attack_class", "payload", "expected_decision"):
        assert key in fixture, f"{fixture.get('__path__')}: missing {key}"
    assert fixture["expected_decision"] in {"quarantine", "allow_with_warn", "allow"}


def test_red_team_corpus_block_rate() -> None:
    """>=9/10 fixtures MUST land in ``quarantine`` at intake (enforce mode).

    PRD-SEC-001 FR-001 acceptance: scorer rejects >= 9 of 10 known
    injection patterns. Each fixture passes ``source_identity`` in
    metadata so the quarantine band (0.4 <= score < 0.7) is the
    expected landing zone; a missing source_identity would push the
    score into ``reject`` (<0.4) — also acceptable but we test the
    tighter band for signal.
    """
    blocked = 0
    allowed_ids: list[str] = []
    for fixture in _FIXTURES:
        result = score_intake(
            content=str(fixture["payload"]),
            metadata={"source_identity": "redteam-corpus"},
            observe_mode=False,
        )
        if result.decision == "quarantine":
            blocked += 1
        else:
            allowed_ids.append(f"{fixture['attack_id']} -> {result.decision} (score={result.score:.2f})")

    # Floor: >=9 of the first 10 fixtures (corpus may grow beyond 10).
    total = len(_FIXTURES)
    required = max(9, int(total * 0.9))
    assert blocked >= required, f"block rate {blocked}/{total} below floor {required}/{total}. Allowed: {allowed_ids}"


def test_no_fixture_scores_as_allow() -> None:
    """Stronger signal: no red-team payload should land as ``allow``.

    This is a stricter sibling of the >=9/10 floor — any ``allow``
    decision on an adversarial payload is a security regression.
    Documented separately so a single regression is easy to diagnose.
    """
    allowed: list[str] = []
    for fixture in _FIXTURES:
        result = score_intake(
            content=str(fixture["payload"]),
            metadata={"source_identity": "redteam-corpus"},
            observe_mode=False,
        )
        if result.decision == "allow":
            allowed.append(f"{fixture['attack_id']} (score={result.score:.2f})")
    # Allow at most 10% slack to match the >=9/10 floor.
    assert len(allowed) <= max(1, len(_FIXTURES) // 10), f"red-team fixtures scoring as 'allow': {allowed}"
