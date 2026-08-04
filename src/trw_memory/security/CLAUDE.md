# trw_memory/security — Sub-CLAUDE.md

Scope-local guidance for the trw-memory security layer. Closes the
sub-CLAUDE.md row of PRD-SEC-001 exit criteria.

## What Lives Here

| Module | Purpose | PRD ref |
|---|---|---|
| `encryption.py` | AES-256-GCM field-level encrypt/decrypt | Pre-SEC-001 |
| `keys.py` | Master-key storage, retrieval, rotation | Pre-SEC-001 |
| `rbac.py` | Role-based access control, namespace permissions | Pre-SEC-001 |
| `audit.py` | SHA-256 hash-chained immutable audit log | Pre-SEC-001 |
| `pii.py` | PII detection (Shannon-entropy based) + egress anonymization (`strip_pii`). The store path blocks API keys and records detections; it does not rewrite stored text — see `_runtime_pii.REDACTED_PII_TYPES` | Pre-SEC-001 |
| `poisoning.py` | Anomaly detection for memory poisoning | SEC-001 FR-003 |
| `trust_scorer.py` | Intake trust-score computation (observe-mode v1) | SEC-001 FR-001 |
| `recall_filter.py` | Recall-time filter for quarantined entries | SEC-001 FR-004 |
| `canary.py` | Canary learning injection + verification | SEC-001 FR-007 |
| `provenance.py` | Ed25519 provenance hash-chain on writes | SEC-001 FR-002 |
| `runtime.py` | Security runtime composition (bundled facade) | SEC-001 |
| `write_gate.py` | `guarded_store` — the ONLY supported way to persist a caller-supplied entry from outside `security/`. Runs `prepare_entry_for_store` then `backend.store`, diverting a quarantine decision to the review store | SEC-001 |

## v1 Rollout: Observe-Mode Only

`trust_scorer.score_intake(...)` and `recall_filter.filter_recall_window(...)`
both run in **observe-mode** during Sprint 96. Live write/recall paths emit
their what-they-would-do decisions into the append-only security event stream
(`events-YYYY-MM-DD.jsonl`) via `security/telemetry_emit.py`, rather than
structlog-only logs. The 14-day calibration clock starts when the modules
are deployed; threshold-lock promotion is a Sprint 97 decision gate
(PRD-SEC-001 §8 Rollout Phase 1 → Phase 2).

Do NOT promote to enforce mode in this package without the Sprint 97
maintainer sign-off + operator kill-switch flag.

The switches are the three below. This section previously named a single flag,
`config.security.memory_poisoning_enforce`, which **has never existed in code** —
a repo-wide grep returns only this document. An operator who followed it to flip
enforce mode changed nothing, and an auditor who checked "is there a kill switch"
against the doc got a false yes. Corrected 2026-07-30; verify a named switch
against `models/_config_security.py` before trusting it here.

| Control | Field | Default |
|---|---|---|
| Intake trust scoring | `trust_scoring_mode` | `observe` (log only) |
| Anomaly quarantine | `poisoning_detection_mode` | `observe` (log only) |
| Recall filtering | `recall_filter_mode` | `redact` |

`enable_trust_scoring` and `poisoning_detection_enabled` are the hard off
switches; `enable_recall_filter` is the recall-side equivalent.

## Editing Rules

- **Any new surface that persists a caller-supplied `MemoryEntry` must call
  `write_gate.guarded_store`, never `backend.store` directly.**
  `tests/test_store_write_gate_totality.py` derives every `.store(...)` call
  site in the production tree and fails on a new one that is neither guarded,
  inside `security/`, nor in its documented exclusion set. Add to that exclusion
  set only for an internal re-write of an already-gated entry, and say why.
- All public API symbols are listed in `__init__.py` `__all__`. When
  adding a new symbol, re-export it there and update
  `docs/requirements-aare-f/prds/agentic-hpo/PRD-SEC-001-*.md` §5 FR
  references.
- **structlog: never use `event=` kwarg** — it's reserved. Use
  `action=` or a descriptive kwarg per trw-memory root CLAUDE.md.
- `AuditLog` hash chain is append-only — `verify_audit_chain()` must
  stay property-test-backed (any mutation breaks the chain). Do not
  add an "update" path.
- Ed25519 keys live under `get_master_key()` — never re-roll key
  material in a test without using a fixture key set, or the key file
  leaks to `~/.trw/`.

## Canary Mechanism (FR-007)

`seed_canaries()` inserts N canary learnings with deterministic content.
`verify_canaries()` re-reads them via the normal recall path and compares
content hash. If any canary comes back tampered, that's a trust-layer
breach signal — emit a `MemorySecurityEvent` row to the security event
stream with `payload.event_name="canary_hash_drift"` or `"canary_missing"`.

Canary seeding is **idempotent** and **in-memory-hash-pinned**. Pinning
must happen at seed time, not retrieval time, so a compromised storage
layer cannot simply replay valid canary content.

## Red-Team Fixture Corpus (Pending — ≥10 patterns)

Location: `trw-memory/tests/fixtures/security/red_team/` (to be created).
`trust_scorer` must reject ≥9/10 patterns. When adding a fixture:
1. Write the attack payload as a `.yaml` file
2. Add a test that asserts `score_intake(...).decision == "quarantine"`
3. Update `poisoning.py` `AnomalyType` enum if the attack class is new

Keep the fixture set ≥10 entries — `tests/test_red_team_coverage.py`
enforces the floor at import time.

## Provenance Chain (FR-002)

Every `trw_learn` write appends a `ProvenanceEntry` signed with the
project's Ed25519 key. `provenance_verify` walks the chain and returns
the first broken link. The chain is **per-namespace**, not global, so
namespace-scoped compromise doesn't cascade.

Do not store signatures inside the entry payload — they go into the
provenance chain file (`.trw/memory/provenance.jsonl`) keyed by
`entry_id`. Keeping them separate makes rotation cheap.

## Testing

```bash
cd trw-memory
../.venv/bin/python -m pytest tests/test_poisoning_*.py -v
../.venv/bin/python -m pytest tests/test_trust_scorer.py -v
../.venv/bin/python -m pytest tests/test_canary.py -v
../.venv/bin/python -m pytest tests/test_provenance.py -v
../.venv/bin/python -m pytest tests/test_recall_filter.py -v
../.venv/bin/python -m mypy --strict src/trw_memory/security/
```

## References

- PRD: `docs/requirements-aare-f/prds/agentic-hpo/PRD-SEC-001-memory-poisoning-defense.md`
- Sprint: `docs/requirements-aare-f/sprints/active/sprint-96-agentic-hpo-foundation.md`
- Governance: `docs/research/agentic-hpo/governance-mapping-2026.md`
