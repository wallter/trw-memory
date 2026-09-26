"""Score a judged LOCOMO run with trw-jev: a probability per answer, as a second rubric.

Why a separate script and not a `--judge-model` in batch_judge.py: Jev is not a chat model. It takes
a state plus typed questions and returns probabilities, over ``/api/alpha/decisions``. Per the
Jev starter guide this code follows three of its rules that matter here:

* **one item per call by default.** Batched items are not independent (mean max|Δp| 0.18 with a
  shared state), and we compare probabilities against a threshold, so each answer is scored alone.
  ``--batch-size N`` groups N items per call the way the starter permits for threshold use: each
  item's text lives in its own question's ``instructions`` (measured drift 0.06, 3/28 label flips)
  rather than in a shared state, at ~1.7x tokens. Chunks are sized by measured ``input_tokens``
  against the ~56.8k combined ceiling, and any item a chunk fails to answer is reported, never
  dropped. Validate a batch size against a solo run before trusting its absolute probabilities.
* **descriptive wording, not labels.** Terse criteria *and* terse instructions together measured AUC
  0.737 against 0.955 for descriptive prose; the starter varied both, so this is the combined effect.
* **the threshold is not fitted.** The acceptance rate is reported at the raw 0.5 cut, with confident
  positive (>= 0.53), uncertain (0.47-0.53) and confident negative (< 0.47) counted separately, and
  the rank-based agreement (AUC against the harness judge) beside it, because the starter found an
  in-sample AUC of 1.000 falling to 0.766 held out. The 0.5 rate is exploratory, not a verdict.

The reference answer is preprocessed exactly as the harness judge preprocesses it (category 3 keeps
the text before the semicolon), so both rubrics judge against the same reference. Output records the
model id the API reports, the rubric and input hashes, the batch mode and the cutoff, so a run is
reproducible and never silently overwritten by a different configuration.

Writes ``jev.json`` next to the judged run: {qid: {"p": float|None, "harness": "CORRECT"|...}}.

    python jev_judge.py --judged <bench>/results/locomo/predicted_trw-v6__gpt4omini --cutoff 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import batch_judge as bj

from trw_memory.decisions import ChoiceQuestion, DecisionResult, NoulQuestion, ScoreQuestion
from trw_memory.decisions._jev_http import JevHttpJudge

DEAD_BAND = 0.03  # JEV-STARTER: ±0.03 around any acted-on threshold

QUESTION = {
    "correct": NoulQuestion(
        instructions=(
            "A question was answered from someone's conversation history. Judge whether the generated "
            "answer conveys the same fact as the reference answer a human wrote for this question."
        ),
        criteria={
            "true": (
                "The generated answer states the same fact as the reference answer. Extra detail, different "
                "wording, or a fuller sentence are fine as long as the substance matches, and a more precise "
                "form of the same fact (a full date where the reference gives a month) counts as matching."
            ),
            "false": (
                "The generated answer states a different fact, names a different entity, date or place, omits "
                "part of a multi-part reference answer, hedges between alternatives without committing, or "
                "declines to answer."
            ),
        },
    )
}


# --diagnose adds two more questions ABOUT THE SAME ITEM. The starter's cheap direction: questions
# about one state are near-independent (25-q vs 40-q batch moved answers by 0.014, at repeat noise),
# unlike batching separate items. So each item costs one call whether it carries 1 question or 3.
COVERAGE = ScoreQuestion(
    instructions=(
        "How much of the reference answer does the generated answer cover? Judge coverage of the facts "
        "the reference states, not style or length."
    ),
    criteria=[
        "none of the reference answer's facts appear",
        "one part of a multi-part reference answer, or a related but different fact",
        "most of the reference answer's facts, missing something the reference states",
        "everything the reference answer states",
    ],
)
FAILURE = ChoiceQuestion(
    instructions="What best describes the relationship between the generated answer and the reference answer?",
    criteria={
        "match": "states the same fact, allowing for wording, extra detail or a more precise form",
        "partial": "states some of a multi-part reference answer but omits the rest",
        "wrong_entity": "names a different person, thing or place than the reference",
        "wrong_time": "gives a date, duration or ordering that disagrees with the reference",
        "hedged": "offers alternatives or qualifies without committing to an answer",
        "no_answer": "declines, says it does not know, or says the memories do not contain it",
    },
)


MODELS_SEEN: set[str] = set()


def _answers(res: object) -> dict[str, Any]:
    """The answers of a successful call; ``{}`` for a ``DecisionFailure``, so the item is reported unscored."""
    if not isinstance(res, DecisionResult):
        return {}
    MODELS_SEEN.add(res.model)
    return dict(res.answers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged", type=Path, required=True, help="a predicted_<run>__<tag> directory")
    ap.add_argument("--bench-dir", type=Path, default=Path("~/.cache/trw-bench/memory-benchmarks"),
                    help="pinned harness, for the same gold-answer preprocessing the harness judge uses")  # fmt: skip
    ap.add_argument("--expected", type=int, default=None, help="fail unless exactly this many items are scored")
    ap.add_argument("--cutoff", type=int, default=10)
    ap.add_argument("--qids", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1, help="items per call (1 = solo, the safe default)")
    ap.add_argument("--out-name", default="jev.json", help="filename to write inside --judged")
    ap.add_argument("--diagnose", action="store_true",
                    help="also ask coverage (score) and failure mode (choice) about each item")  # fmt: skip
    args = ap.parse_args(argv)
    label = f"top_{args.cutoff}"

    # The harness judge preprocesses gold answers (category 3: keep the text before the semicolon).
    # JEV must see the same reference or the two rubrics are judging different questions.
    harness_prompts = bj.load_harness(args.bench_dir.expanduser())[1] if args.bench_dir else None

    def reference(d: dict[str, Any]) -> str:
        gold = d.get("ground_truth_answer")
        if gold is None:
            raise SystemExit(f"{d.get('question_id')}: no ground_truth_answer; refusing to judge against 'None'")
        if harness_prompts is not None and d.get("category") is not None:
            return str(harness_prompts.preprocess_answer(d["category"], str(gold)))
        return str(gold)

    files = sorted(args.judged.glob("conv*_q*.json"))
    if args.qids:
        keep = {q.strip() for q in args.qids.read_text().splitlines() if q.strip()}
        files = [f for f in files if f.stem in keep]
    judge = JevHttpJudge(api_key=bj.api_key())

    def score(path: Path) -> tuple[str, dict[str, Any]]:
        d = json.loads(path.read_text())
        cr = d.get("cutoff_results", {}).get(label)
        if cr is None:
            return path.stem, {"p": None, "harness": None, "reason": "not judged at this cutoff"}
        state = {
            "question": d["question"],
            "reference_answer": reference(d),
            "generated_answer": cr["generated_answer"],
        }
        questions = dict(QUESTION)
        if args.diagnose:
            questions["coverage"] = COVERAGE
            questions["failure"] = FAILURE
        res = judge.decide(state, questions, timeout_s=30)
        answers = _answers(res)
        row: dict[str, Any] = {
            "p": getattr(answers.get("correct"), "noul", None),
            "harness": cr["judgment"],
            "category": d.get("category_name"),
        }
        if args.diagnose:
            cov, fail = answers.get("coverage"), answers.get("failure")
            row["coverage"] = getattr(cov, "score", None)
            # score legend/probabilities are keyed by STRING, zero-indexed (the starter's most common bug)
            row["coverage_legend"] = getattr(cov, "legend", None)
            row["failure"] = getattr(fail, "choice", None)
            row["failure_p"] = (getattr(fail, "probabilities", None) or {}).get(getattr(fail, "choice", ""))
        return path.stem, row

    def item_question(qid: str, state: dict[str, Any]) -> NoulQuestion:
        """The item's own text inside its question, so a batched answer names what it is about."""
        base = QUESTION["correct"]
        return NoulQuestion(
            instructions=(
                f"{base.instructions}\n\nItem {qid}.\nQuestion: {state['question']}\n"
                f"Reference answer: {state['reference_answer']}\nGenerated answer: {state['generated_answer']}"
            ),
            criteria=base.criteria,
        )

    def score_chunk(chunk: list[Path]) -> list[tuple[str, dict[str, Any]]]:
        states, questions = {}, {}
        out_rows = []
        for path in chunk:
            d = json.loads(path.read_text())
            cr = d.get("cutoff_results", {}).get(label)
            if cr is None:
                out_rows.append((path.stem, {"p": None, "harness": None, "reason": "not judged at this cutoff"}))
                continue
            states[path.stem] = {
                "question": d["question"],
                "reference_answer": reference(d),
                "generated_answer": cr["generated_answer"],
                "harness": cr["judgment"],
                "category": d.get("category_name"),
            }
            questions[path.stem] = item_question(path.stem, states[path.stem])
        if questions:
            res = judge.decide({"task": "judge each item named in its own question"}, questions, timeout_s=60)
            answers = _answers(res)
            for qid, st in states.items():
                ans = answers.get(qid)
                out_rows.append((qid, {"p": getattr(ans, "noul", None), "harness": st["harness"],
                                       "category": st["category"]}))  # fmt: skip
        return out_rows

    if args.batch_size > 1:
        chunks = [files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)]
        with ThreadPoolExecutor(args.workers) as pool:
            rows = {q: r for part in pool.map(score_chunk, chunks) for q, r in part}
    else:
        with ThreadPoolExecutor(args.workers) as pool:
            rows = dict(pool.map(score, files))

    scored = {q: r for q, r in rows.items() if r["p"] is not None}
    unscored = [q for q, r in rows.items() if r["p"] is None]
    if args.expected is not None and len(scored) != args.expected:
        print(f"expected {args.expected} scored items, got {len(scored)} (unscored: {unscored[:5]})", file=sys.stderr)
    at_half = sum(1 for r in scored.values() if r["p"] >= 0.5)
    confident_yes = sum(1 for r in scored.values() if r["p"] >= 0.5 + DEAD_BAND)
    confident_no = sum(1 for r in scored.values() if r["p"] <= 0.5 - DEAD_BAND)
    band = len(scored) - confident_yes - confident_no
    harness = sum(1 for r in scored.values() if r["harness"] == "CORRECT")
    # AUC is against the harness's binary verdict; an unparseable harness reply (ERROR) is neither
    # a positive nor a negative, so it is excluded rather than silently counted as wrong.
    pos = [r["p"] for r in scored.values() if r["harness"] == "CORRECT"]
    neg = [r["p"] for r in scored.values() if r["harness"] == "WRONG"]
    errors = sum(1 for r in scored.values() if r["harness"] not in ("CORRECT", "WRONG"))
    provenance = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_reported": sorted(MODELS_SEEN),
        "rubric_sha": hashlib.sha256(QUESTION["correct"].model_dump_json().encode()).hexdigest()[:16],
        "inputs_sha": hashlib.sha256(json.dumps(sorted(rows), sort_keys=True).encode()).hexdigest()[:16],
        "cutoff": label, "batch_size": args.batch_size, "dead_band": DEAD_BAND,
    }  # fmt: skip
    out = args.judged / args.out_name
    out.write_text(json.dumps({"provenance": provenance, "rows": rows}, indent=1))
    auc = (
        sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    )
    print(json.dumps({
        "judged_dir": str(args.judged), "cutoff": label, "scored": len(scored), "unscored": len(unscored),
        "harness_correct": harness, "harness_pct": round(100 * harness / max(len(scored), 1), 1),
        "jev_at_0.5": at_half, "jev_pct_at_0.5": round(100 * at_half / max(len(scored), 1), 1),
        "confident_yes_ge_0.53": confident_yes, "confident_no_le_0.47": confident_no, "uncertain_band": band,
        "harness_errors_excluded_from_auc": errors,
        "auc_vs_harness": round(auc, 3), "batch_size": args.batch_size, "provenance": provenance,
    }, indent=1))  # fmt: skip
    if unscored:
        print(f"UNSCORED (never dropped silently): {unscored[:10]}{'…' if len(unscored) > 10 else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
