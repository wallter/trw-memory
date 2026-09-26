"""Evidence-aware failure attribution and answer-key audit for a judged LOCOMO run.

``jev_judge.py`` asks whether an answer matches the reference. It never sees the
memories, so its ``no_answer`` label cannot say *why* the reader declined. This
tool closes that gap: for every question it shows trw-jev the same three things a
human adjudicator would need, and asks three questions about that one state.

  1. ``reader_sufficient`` -- do the memories the reader was actually given contain
     enough to state the reference answer? This is the measurement that decides
     where work belongs. A wrong answer over sufficient context is a READER
     failure; a wrong answer over insufficient context is a RETRIEVAL failure.
     Aggregate recall@k cannot make this call per item, because it scores
     annotated ``dia_id`` overlap rather than semantic sufficiency: a
     non-annotated turn may support the answer perfectly well, and an annotated
     turn may be useless without the turn before it.
  2. ``key_supported`` -- is the reference answer itself supported by the dialogue
     turns LOCOMO cites as its evidence? An independent audit reported that 6.4%
     of the LOCOMO answer key is wrong. Any claim we publish off this dataset
     inherits that noise floor, so we measure it ourselves rather than cite it.
  3. ``bottleneck`` -- a single categorical attribution, so the three signals can
     disagree visibly instead of being averaged into a number.

Questions about one state are near-independent, so all three ride in one call
(~$0.0001/item, ~$0.15 for the full 1,540). Items are never batched together:
separate items in one call are not independent (measured 1.9% label flips).

Usage::

    python jev_audit.py --judged <predicted_dir> [--cutoff 10] [--qids FILE]

Writes ``jev_audit.json`` inside ``--judged`` and prints the attribution table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import batch_judge as bj
import retrieval_eval as re_eval

from trw_memory.decisions import ChoiceQuestion, DecisionResult, NoulQuestion
from trw_memory.decisions._jev_http import JevHttpJudge

DEAD_BAND = 0.03  # JEV-STARTER: +-0.03 around any acted-on threshold

READER_SUFFICIENT = NoulQuestion(
    instructions=(
        "Below are the memories a system retrieved and showed to a reader, the question the reader "
        "was asked, and the reference answer a human wrote. Judge only the memories: do they contain "
        "enough information for a careful reader to state the reference answer? Ignore what the "
        "generated answer actually said."
    ),
    criteria={
        "true": (
            "The retrieved memories state, or let a careful reader derive by ordinary reasoning over "
            "what is shown (including simple date arithmetic from a stated timestamp), every fact the "
            "reference answer gives."
        ),
        "false": (
            "Some fact the reference answer gives is absent from the retrieved memories, or deriving it "
            "would need information that is not shown."
        ),
    },
)

KEY_SUPPORTED = NoulQuestion(
    instructions=(
        "Below is a question, the reference answer a human wrote for it, and the verbatim dialogue "
        "turns the dataset cites as the evidence for that reference answer. Judge whether those cited "
        "turns actually support the reference answer."
    ),
    criteria={
        "true": (
            "The cited turns state, or directly imply, the reference answer. Relative expressions are "
            "supported when the turn's own timestamp makes the arithmetic determinate."
        ),
        "false": (
            "The cited turns do not support the reference answer: they say something different, they "
            "are missing the fact entirely, the required arithmetic is indeterminate from what is "
            "cited, or the answer attributes the fact to the wrong speaker."
        ),
    },
)

BOTTLENECK = ChoiceQuestion(
    instructions=(
        "Given the question, the reference answer, the cited evidence turns, the memories the reader "
        "was shown and the answer the reader produced, what single thing best explains the outcome?"
    ),
    criteria={
        "answered_correctly": "the generated answer conveys the reference answer",
        "retrieval_missed": "the retrieved memories lack a fact the reference answer needs",
        "reader_missed": "the retrieved memories were sufficient, but the generated answer is wrong or incomplete",
        "reader_declined": "the retrieved memories were sufficient, but the reader declined or said it did not know",
        "reference_unsupported": "the cited evidence turns do not support the reference answer, so the item is unscoreable",
        "ambiguous": "the question or the reference answer admits more than one defensible answer",
    },
)


def evidence_index(dataset: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """``{conversation_idx: {dia_id: "Speaker: text"}}`` in the harness's own turn formatting."""
    out: dict[int, dict[str, str]] = {}
    for ci, entry in enumerate(dataset):
        turns: dict[str, str] = {}
        for _key, date, _dt, dia_id, text in re_eval.iter_turns(entry["conversation"]):
            if dia_id:
                turns[dia_id] = f"[{date}] {text}"
        out[ci] = turns
    return out


def report(rows: dict[str, dict[str, Any]], cutoff: int, name: str) -> None:
    """Render the attribution tables. Separate from scoring so a saved run can be
    re-read for free -- the calls are paid for once and the framing changes often."""
    vals = [r for r in rows.values() if r.get("bottleneck") is not None]

    def pct(n: int, d: int) -> str:
        return f"{100.0 * n / d:5.1f}%" if d else "    -"

    # Probabilities are calibrated, so a 0.5 cut is exploratory; the counts below are
    # what the numbers mean, and the mid-band is reported separately because an item
    # near 0.5 is one the rubric cannot decide, not one the system got half right.
    def band(key: str, rowset: list[dict[str, Any]]) -> tuple[int, int, int, int]:
        """(yes, undecided, no, missing). A missing probability is NOT a "no" -- an
        item whose question failed to return has no verdict, and folding it into the
        negative band would quietly overstate every negative rate."""
        vals = [r.get(key) for r in rowset]
        hi = sum(1 for v in vals if v is not None and v > 0.5 + DEAD_BAND)
        lo = sum(1 for v in vals if v is not None and v < 0.5 - DEAD_BAND)
        missing = sum(1 for v in vals if v is None)
        return hi, len(rowset) - hi - lo - missing, lo, missing

    print(f"\n== evidence audit | {len(vals)} items | cutoff top-{cutoff} | {name}")
    for key in ("reader_sufficient", "key_supported"):
        hi, mid, lo, missing = band(key, vals)
        note = f"  missing {pct(missing, len(vals))}" if missing else ""
        print(f"{key:20s} yes {pct(hi, len(vals))}  undecided {pct(mid, len(vals))}  no {pct(lo, len(vals))}{note}")

    print(f"\n{'bottleneck':24s}{'all':>9s}" + "".join(f"{c:>16s}" for c in ("harness-right", "harness-wrong")))
    # `judgment` is the harness's string verdict ("CORRECT"/"WRONG"), so it must be
    # compared, not tested for truthiness -- every non-empty string is truthy.
    right = [r for r in vals if r.get("harness") == "CORRECT"]
    wrong = [r for r in vals if r.get("harness") == "WRONG"]
    counts = Counter(r["bottleneck"] for r in vals)
    for choice in BOTTLENECK.criteria:
        cr_ = sum(1 for r in right if r["bottleneck"] == choice)
        cw = sum(1 for r in wrong if r["bottleneck"] == choice)
        print(f"{choice:24s}{pct(counts[choice], len(vals)):>9s}{pct(cr_, len(right)):>16s}{pct(cw, len(wrong)):>16s}")

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in vals:
        by_cat[r.get("category") or "?"].append(r)
    print(f"\n{'category':14s}{'sufficient':>12s}{'key ok':>10s}{'retr.miss':>11s}{'reader':>9s}{'n':>6s}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        rm = sum(1 for r in rs if r["bottleneck"] == "retrieval_missed")
        rd = sum(1 for r in rs if r["bottleneck"] in ("reader_missed", "reader_declined"))
        print(
            f"{cat:14s}{pct(band('reader_sufficient', rs)[0], len(rs)):>12s}"
            f"{pct(band('key_supported', rs)[0], len(rs)):>10s}"
            f"{pct(rm, len(rs)):>11s}{pct(rd, len(rs)):>9s}{len(rs):>6d}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged", type=Path, required=True, help="a predicted_<run>__<tag> directory")
    ap.add_argument("--bench-dir", type=Path, default=Path("~/.cache/trw-bench/memory-benchmarks"))
    ap.add_argument("--dataset", type=Path, default=None, help="defaults to the pinned harness's locomo10.json")
    ap.add_argument("--cutoff", type=int, default=10, help="how many memories the reader was shown")
    ap.add_argument("--qids", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items (a cheap probe)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-name", default="jev_audit.json")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the table from a previous run's JSON without paying for it again")  # fmt: skip
    args = ap.parse_args(argv)
    label = f"top_{args.cutoff}"

    if args.report_only:
        saved = json.loads((args.judged / args.out_name).read_text())
        report(saved["rows"], saved.get("cutoff", args.cutoff), args.judged.name)
        return 0

    bench = args.bench_dir.expanduser()
    dataset_path = args.dataset or bench / "datasets/locomo/locomo10.json"
    evidence_by_conv = evidence_index(re_eval.load_json(str(dataset_path)))
    harness_prompts = bj.load_harness(bench)[1]

    files = sorted(args.judged.glob("conv*_q*.json"))
    if args.qids:
        keep = {q.strip() for q in args.qids.read_text().splitlines() if q.strip()}
        files = [f for f in files if f.stem in keep]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no judged files under {args.judged}")
    judge = JevHttpJudge(api_key=bj.api_key())

    def audit(path: Path) -> tuple[str, dict[str, Any]]:
        d = json.loads(path.read_text())
        cr = (d.get("cutoff_results") or {}).get(label)
        if cr is None:
            return path.stem, {"reason": "not judged at this cutoff"}
        gold = d.get("ground_truth_answer")
        if gold is None:
            raise SystemExit(f"{path.stem}: no ground_truth_answer; refusing to audit against 'None'")
        reference = str(harness_prompts.preprocess_answer(d["category"], str(gold)))

        # The reader saw exactly the first `cutoff` search results, content only --
        # `detail` (the preceding turns carried at ingest) is used for ranking but is
        # NOT returned by the shim, so it must not appear here either.
        results = ((d.get("retrieval") or {}).get("search_results") or [])[: args.cutoff]
        shown = "\n".join(f"- [{r.get('created_at', '')}] {r.get('memory', '')}" for r in results)

        turns = evidence_by_conv.get(d["conversation_idx"], {})
        cited = "\n".join(turns.get(e, f"<missing turn {e}>") for e in (d.get("evidence") or []))

        # TWO calls, deliberately. Sufficiency must be judged from the retrieved
        # memories ALONE: putting it in one state beside the cited evidence turns
        # and the generated answer lets the judge import a fact from the citations
        # and then call the context sufficient. An instruction to ignore a field
        # in the state is not isolation. The attribution call keeps the full
        # picture, because naming the bottleneck genuinely needs all of it.
        suff_state = {
            "question": d["question"],
            "reference_answer": reference,
            "retrieved_memories_shown_to_reader": shown or "<none retrieved>",
            "todays_date": d.get("reference_date", ""),
        }
        suff = judge.decide(suff_state, {"reader_sufficient": READER_SUFFICIENT}, timeout_s=45)

        full_state = {
            **suff_state,
            "cited_evidence_turns": cited or "<none cited>",
            "generated_answer": cr["generated_answer"],
        }
        res = judge.decide(full_state, {"key_supported": KEY_SUPPORTED, "bottleneck": BOTTLENECK}, timeout_s=45)
        if not isinstance(res, DecisionResult):
            return path.stem, {"reason": f"jev attribution call failed: {getattr(res, 'kind', 'unknown')}"}
        a = res.answers
        bn = a.get("bottleneck")
        return path.stem, {
            "category": d.get("category_name"),
            "harness": cr["judgment"],
            "n_cited": len(d.get("evidence") or []),
            "reader_sufficient": getattr(
                (suff.answers if isinstance(suff, DecisionResult) else {}).get("reader_sufficient"), "noul", None
            ),
            "key_supported": getattr(a.get("key_supported"), "noul", None),
            "bottleneck": getattr(bn, "choice", None),
            "bottleneck_p": (getattr(bn, "probabilities", None) or {}).get(getattr(bn, "choice", "")),
            "model": res.model,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = dict(pool.map(audit, files))

    scored = {k: v for k, v in rows.items() if v.get("bottleneck") is not None}
    failed = len(rows) - len(scored)
    if failed > len(rows) * 0.01:
        print(f"WARNING: {failed}/{len(rows)} items failed to score", file=sys.stderr)

    out = args.judged / args.out_name
    out.write_text(
        json.dumps(
            {
                "cutoff": args.cutoff,
                "model": sorted({str(r["model"]) for r in rows.values() if r.get("model")}),
                "n": len(scored),
                "n_failed": failed,
                "rows": rows,
            },
            indent=1,
        )
    )

    report(rows, args.cutoff, args.judged.name)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
