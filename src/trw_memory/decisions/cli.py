"""Manual CLI for the decision toolkit — the offline-usable half of the capability.

    python -m trw_memory.decisions.cli ask   --state-file s.json --questions-file q.json
    python -m trw_memory.decisions.cli batch --items-file items.json --questions-file q.json [--schema embedded|keyed]
    python -m trw_memory.decisions.cli rank  --items-file items.json --instructions "..." \\
        --true "worth a human's attention now" --false "can wait" [--schema keyed|embedded]
    python -m trw_memory.decisions.cli classify --state-file s.json --options-file o.json \\
        --instructions "Which queue handles this?" [--act-at 0.7]
    python -m trw_memory.decisions.cli calibrate --pairs-file p.json     # [[prob, true/false], ...]

``ask`` is the primitive: one state, any mix of noul/choice/score questions, one call. ``batch``
asks the same questions about many items. Every subcommand prints JSON on stdout. Exit status:
0 complete, 2 caller error (fix the request), 3 failed (no answer at all), 4 partial (read the
failures before trusting the rest). Enablement follows the process env / project scope
(``--dotenv``'s directory: its ``.trw/config.yaml`` or ``TRW_JEV_ENABLED`` in its ``.env``) / user
scope (``~/.trw/config.yaml``) precedence in
:func:`trw_memory.decisions._enablement.resolve_backend_enablement`; ``OPENROUTER_API_KEY`` comes
from the environment or the project ``.env``. With nothing configured it resolves to the null
judge and every outcome is a ``disabled`` failure without touching the network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from trw_memory.decisions._env import toolkit_from_env
from trw_memory.decisions._redaction import default_redactor
from trw_memory.decisions.toolkit import InvalidRequest, Policy, Toolkit, choice, reliability

_EXIT = {"complete": 0, "partial": 4, "failed": 3}


def _load(path: str | None) -> Any:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _emit(payload: dict[str, Any], code: int) -> int:
    json.dump(payload, sys.stdout, indent=1, default=str)
    sys.stdout.write("\n")
    return code


def _toolkit(args: argparse.Namespace) -> Toolkit:
    # The dotenv's directory doubles as the project root, so a CLI invocation from a project
    # checkout gets the same project-scope enablement layer the MCP tool does (2026-09-23).
    project_root = Path(args.dotenv).parent if args.dotenv else None
    return toolkit_from_env(
        os.environ,
        redactor=default_redactor,
        dotenv_path=args.dotenv,
        project_root=project_root,
        timeout_s=args.timeout,
        session_id=args.session_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trw-decisions", description=__doc__)
    parser.add_argument("--dotenv", default=".env", help="project .env to read credentials from")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--session-id", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="every question about one state, one call")
    p.add_argument("--state-file", required=True)
    p.add_argument("--questions-file", required=True)

    p = sub.add_parser("batch", help="the same questions about many items")
    p.add_argument("--items-file", required=True, help='JSON {"key": <state>, ...}')
    p.add_argument("--questions-file", required=True)
    p.add_argument("--schema", choices=["embedded", "keyed"], default="embedded")
    p.add_argument("--chunk-size", type=int, default=100)

    p = sub.add_parser("rank", help="order many items by one yes/no question")
    p.add_argument("--items-file", required=True)
    p.add_argument("--instructions", required=True)
    p.add_argument("--true", dest="true_desc", required=True)
    p.add_argument("--false", dest="false_desc", required=True)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--schema", choices=["keyed", "embedded"], default="keyed")

    p = sub.add_parser("classify", help="one of N options")
    p.add_argument("--state-file", required=True)
    p.add_argument("--options-file", required=True, help='JSON {"label": "descriptive rubric", ...}')
    p.add_argument("--instructions", required=True)
    p.add_argument("--act-at", type=float, default=None, help="fitted on YOUR held-out labels")

    p = sub.add_parser("calibrate", help="reliability curve, ECE and Brier from labeled pairs")
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--bins", type=int, default=10)

    args = parser.parse_args(argv)
    if args.cmd == "calibrate":
        pairs = [(float(p), bool(ok)) for p, ok in _load(args.pairs_file)]
        return _emit(reliability(pairs, bins=args.bins), 0)

    kit = _toolkit(args)
    try:
        if args.cmd == "ask":
            result = kit.ask(_load(args.state_file), _load(args.questions_file))
            return _emit(
                {
                    "status": result.status,
                    "outcomes": result.to_wire(),
                    "model": result.model,
                    "backend": result.backend,
                    "latency_ms": result.latency_ms,
                    "usage": dict(result.usage),
                },
                _EXIT[result.status],
            )
        if args.cmd == "batch":
            batch = kit.batch_items(
                _load(args.items_file), _load(args.questions_file), schema=args.schema, chunk_size=args.chunk_size
            )
            return _emit(
                {
                    "status": batch.status,
                    "schema": batch.schema,
                    "unanswered": batch.unanswered,
                    "items": {
                        k: {"chunk": batch.chunk_of[k], "status": r.status, "outcomes": r.to_wire()}
                        for k, r in batch.per_item.items()
                    },
                },
                _EXIT[batch.status],
            )
        if args.cmd == "rank":
            items = _load(args.items_file)
            ranked = kit.rank(
                items,
                instructions=args.instructions,
                criteria={"true": args.true_desc, "false": args.false_desc},
                chunk_size=args.chunk_size,
                schema=args.schema,
            )
            return _emit(
                {
                    "status": ranked.status,
                    "schema": args.schema,
                    "n_items": len(items),
                    "n_answered": len(ranked.ranked),
                    "unanswered": ranked.unanswered,
                    "ranked": [{"key": r.key, "probability": r.probability, "chunk": r.chunk} for r in ranked.ranked],
                },
                _EXIT[ranked.status],
            )
        # classify
        asked = kit.ask(_load(args.state_file), {"_c": choice(args.instructions, _load(args.options_file))})
        classified = asked.choice("_c")
        payload: dict[str, Any] = {
            "status": asked.status,
            "label": classified.label,
            "probabilities": dict(classified.probabilities),
            "confidence": classified.confidence,
            "margin": round(classified.margin, 4) if classified.probabilities else None,
        }
        if classified.failure is not None:
            payload["failure"] = classified.failure.model_dump()
        if args.act_at is not None and classified.answered:
            route = Policy(act_at=args.act_at).decide(classified)
            payload["route"] = route
            payload["decisive"] = route == "act"
        return _emit(payload, _EXIT[asked.status])
    except InvalidRequest as exc:
        return _emit({"status": "caller_error", "error": str(exc)}, 2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
