"""Answer + judge saved LOCOMO predictions through OpenRouter (Batch API at 50% off, or sync).

Reuses the pinned harness's own prompt builders, cutoff slicing, category-3 preprocessing and
``ANSWER:`` parsing, so the protocol is the stock ``--evaluate-only`` path. Differences, all
deliberate and reported:

* an unparseable judge reply, or a label other than CORRECT/WRONG, is recorded as ``ERROR``
  (score 0, like the stock runner's WRONG) and counted. As in the stock client, only empty or
  malformed-JSON replies are retried (up to ``JUDGE_ATTEMPTS`` = 5 in total); a parsed reply
  with a bad label is not.
* a request whose outcome is unknown (transport failure) is never re-sent within a run, because
  it may already have been billed; the next run of the same command retries it.
* a failed answer request is never judged.

Two rounds per cutoff, because the judge prompt contains the generated answer:
answer -> judge. Between them a quality gate stops the run if more than 1% of answers are
empty or truncated.

Safety, because every request costs money:

* **identity** — the state dir holds a manifest (predictions digest, models, reasoning, rubric,
  memories mode, cutoff, prompt and dataset hashes). A rerun with a different configuration is
  refused; use a new ``--tag``.
* **one writer** — an exclusive lock on the state dir.
* **no double submission** — a batch POST is never retried; intent is persisted first, and an
  uncertain outcome stops the run until ``--adopt <batch_id>`` (found via GET /api/v1/batches)
  or ``--abandon-uncertain`` resolves it.
* **admission** — the run's estimated cost must fit ``--budget`` and the key's remaining limit.
* **STOP** — ``~/.cache/trw-bench/STOP`` (written by watch.py) blocks every submission.
* **no lost results** — sync responses are appended to the state as each arrives.

    python batch_judge.py --bench-dir ~/.cache/trw-bench/memory-benchmarks --pred predicted_trw-v6 \
        --tag gpt4omini --model openai/gpt-4o-mini --cutoff 10 --budget 2 --phase P3
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API = "https://openrouter.ai/api/v1"
ROOT = Path.home() / ".cache" / "trw-bench"
LEDGER = ROOT / "ledger.jsonl"
STOP_FILE = ROOT / "STOP"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
MAX_TOKENS = 4096  # the stock runner's default for both roles
JUDGE_ATTEMPTS = 5  # the stock LLMClient's max_retries
ANSWER_ATTEMPTS = 3  # --retry-bad-answers cap
PROTOCOL = 3  # bump when request construction or parsing changes

# $/M tokens (in, out). Batch is half of standard. Unknown models need --price.
PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "google/gemini-2.5-flash": (0.30, 2.50),
}

# C3's strict re-score. The harness rubric accepts one correct list item, extra detail, dates within
# 14 days and durations within 50%; this one asks whether the answer states the gold fact.
STRICT_JUDGE_PROMPT = """Grade a question-answering system against a gold answer.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Label CORRECT only if the generated answer states the same fact as the gold answer:
- every item of a list-valued gold answer must be present;
- a date must match the gold answer at the gold answer's own precision (day, month or year);
- a number or duration must match exactly unless the gold answer is itself approximate;
- extra details are fine only if none of them contradicts the gold answer;
- "I don't know", hedges between alternatives, or a different entity are WRONG.
Otherwise label WRONG.

Return JSON: {{"reasoning": "<one sentence>", "label": "CORRECT" or "WRONG"}}"""


# --------------------------------------------------------------------------- harness


def load_harness(bench_dir: Path) -> Any:
    """Import the pinned harness's locomo module (prompts, dataset helpers)."""
    # trw-memory/benchmarks is also a package named "benchmarks"; make the harness win.
    for name in [m for m in sys.modules if m == "benchmarks" or m.startswith("benchmarks.")]:
        del sys.modules[name]
    sys.path.insert(0, str(bench_dir))
    from benchmarks.locomo import prompts, run

    return run, prompts


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- requests + parsing


def chat_body(model: str, system: str, user: str, *, json_mode: bool, reasoning: str | None) -> dict[str, Any]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS}
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    else:
        body["temperature"] = 0
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def parse_answer(content: str | None) -> tuple[str, bool]:
    """Stock rule: strip, then keep the text after the last ``ANSWER:``. Returns (answer, had_marker)."""
    text = (content or "").strip()
    if "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[-1].strip(), True
    return text, False


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_judge(content: str | None) -> tuple[str, str]:
    """Stock rule (JSON, unwrap a lone ``final``), tolerating a code fence. Returns (CORRECT|WRONG|ERROR, reason)."""
    if not content:
        return "ERROR", ""
    text = content.strip()
    if m := _FENCE.match(text):
        text = m.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "final" in parsed and len(parsed) == 1:
            inner = parsed["final"]
            parsed = json.loads(inner) if isinstance(inner, str) else inner
    except (json.JSONDecodeError, TypeError, ValueError):
        return "ERROR", ""
    if not isinstance(parsed, dict):
        return "ERROR", ""
    label = str(parsed.get("label") or "").strip().upper()
    if label not in ("CORRECT", "WRONG"):
        return "ERROR", str(parsed.get("reasoning", ""))
    return label, str(parsed.get("reasoning", ""))


def judge_retryable(content: str | None) -> bool:
    """The stock client retries an empty reply or malformed JSON, and nothing else."""
    if not content or not content.strip():
        return True
    text = content.strip()
    if m := _FENCE.match(text):
        text = m.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "final" in parsed and len(parsed) == 1 and isinstance(parsed["final"], str):
            json.loads(parsed["final"])  # the stock client parses the nested string too
    except (json.JSONDecodeError, ValueError):
        return True
    return False


def estimate_cost(reqs: dict[str, dict[str, Any]], role: str, mode: str, price: tuple[float, float]) -> float:
    """Upper-leaning estimate: prompt chars/4 in; out 600 tokens per answer (x3 with reasoning), 150 per judge."""
    tin = sum(sum(len(m["content"]) for m in b["messages"]) for b in reqs.values()) / 4
    per_out = 600 if role == "answer" else 150
    tout = sum(per_out * (3 if "reasoning" in b else 1) for b in reqs.values())
    scale = 0.5 if mode == "batch" else 1.0
    return scale * (tin * price[0] + tout * price[1]) / 1e6


# --------------------------------------------------------------------------- state


class State:
    """Per-(pred, tag, cutoff) working directory: manifest, lock, batch ids, results."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.batches_path = root / "batches.json"
        self.results_path = root / "results.jsonl"
        self.manifest_path = root / "manifest.json"

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        # One lock per output directory: runs for different cutoffs write the same output files.
        with (self.root.parent / "lock").open("w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"another batch_judge run holds {self.root}/lock") from None
            yield

    def check_manifest(self, manifest: dict[str, Any]) -> None:
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            diff = {k: (old.get(k), v) for k, v in manifest.items() if old.get(k) != v}
            if diff:
                raise SystemExit(f"state {self.root} was built with a different configuration {diff}; use a new --tag")
        else:
            self.manifest_path.write_text(json.dumps(manifest, indent=1))

    def batches(self) -> list[dict[str, Any]]:
        return json.loads(self.batches_path.read_text()) if self.batches_path.exists() else []

    def save_batches(self, rows: list[dict[str, Any]]) -> None:
        tmp = self.batches_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(self.batches_path)

    def rows(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        out = []
        for line in self.results_path.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):  # a torn last line from a killed run
                out.append(json.loads(line))
        return out

    def results(self) -> dict[str, dict[str, Any]]:
        """Latest row per custom_id, but a row with content is never replaced by a later failure."""
        out: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            if row.get("content") is not None or row["custom_id"] not in out:
                out[row["custom_id"]] = row
        return out

    def latest(self) -> dict[str, dict[str, Any]]:
        """The most recent row per custom_id, success or not (what happened on the last attempt)."""
        return {row["custom_id"]: row for row in self.rows()}

    def attempts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows():
            counts[row["custom_id"]] = counts.get(row["custom_id"], 0) + 1
        return counts

    def add_results(self, rows: list[dict[str, Any]]) -> None:
        if self.results_path.exists() and self.results_path.stat().st_size:
            with self.results_path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b"\n"
            if torn:  # a killed run left half a line; end it so the next row stays readable
                with self.results_path.open("a") as fh:
                    fh.write("\n")
        with self.results_path.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def ledger(event: dict[str, Any], path: Path | None = None) -> None:
    path = path or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


# --------------------------------------------------------------------------- transport


def api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    candidates = [Path(os.environ["TRW_BENCH_ENV_FILE"])] if os.getenv("TRW_BENCH_ENV_FILE") else []
    for env_file in (*candidates, Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                m = re.match(r"\s*OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
                if m:
                    return m.group(1)
    raise SystemExit("OPENROUTER_API_KEY not set (env, TRW_BENCH_ENV_FILE or .env)")


def http(
    method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 120, retry: bool = True
) -> dict[str, Any]:
    """JSON over HTTPS. ``retry=False`` for any POST whose repetition could be billed twice."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if retry and exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:500]!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if retry and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path} -> {exc!r}") from exc
    raise RuntimeError("unreachable")


def key_info() -> dict[str, Any] | None:
    try:
        return http("GET", "/key", timeout=15, retry=False)["data"]
    except (
        SystemExit,
        Exception,
    ):  # trw-fail-silent-allow: None means usage unknown; watch.py treats a sustained unknown as a stop and refuses to start without a baseline
        return None


def key_usage() -> float | None:
    info = key_info()
    return float(info["usage"]) if info and info.get("usage") is not None else None


def result_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise inline batch results to {custom_id, content, finish_reason, usage, error}."""
    items = batch.get("results") or batch.get("output") or batch.get("data") or []
    rows = []
    for item in items:
        resp = item.get("response") or {}
        body = resp.get("body") if isinstance(resp, dict) else None
        err = item.get("error")
        content = finish = None
        if isinstance(body, dict) and body.get("choices"):
            content = body["choices"][0].get("message", {}).get("content")
            finish = body["choices"][0].get("finish_reason")
        if content is None and not err:
            err = {"status_code": resp.get("status_code") if isinstance(resp, dict) else None, "body": body}
        usage = body.get("usage") if isinstance(body, dict) else None
        rows.append(
            {
                "custom_id": item.get("custom_id"),
                "content": content,
                "finish_reason": finish,
                "usage": usage,
                "error": err,
            }
        )
    return rows


# --------------------------------------------------------------------------- rounds


def submit_batches(
    state: State,
    role: str,
    model: str,
    reqs: dict[str, dict[str, Any]],
    chunk: int,
    phase: str,
    submit: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    ids = sorted(reqs)
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        if STOP_FILE.exists():
            raise SystemExit(f"STOP file present ({STOP_FILE}): not submitting")
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "requests": [{"custom_id": cid, "body": reqs[cid]} for cid in part],
        }
        intent = {"id": None, "role": role, "n": len(part), "status": "submitting", "submitted": time.time(),
                  "first": part[0], "last": part[-1]}  # fmt: skip
        state.save_batches([*state.batches(), intent])
        try:
            resp = submit(payload)
            bid = resp.get("id") or (resp.get("data") or {}).get("id")
            if not bid:
                raise RuntimeError(f"no batch id in response: {str(resp)[:300]}")
        except BaseException as exc:
            # The POST may have been accepted; resubmitting could pay twice.
            ledger({"phase": phase, "event": "submit_uncertain", "role": role, "n": len(part), "error": str(exc)[:300]})
            raise SystemExit(
                f"batch submit outcome unknown ({exc}). Find it with GET /api/v1/batches, then rerun with "
                f"--adopt <batch_id>, or with --abandon-uncertain if it was not created."
            ) from exc
        rows = [r for r in state.batches() if r.get("status") != "submitting"]
        rows.append({**intent, "id": bid, "status": resp.get("status", "submitted")})
        state.save_batches(rows)
        ledger(
            {"phase": phase, "event": "batch_submitted", "batch_id": bid, "role": role, "model": model, "n": len(part)}
        )
        print(f"submitted {role} batch {bid} ({len(part)} requests)", flush=True)


def poll_batches(state: State, role: str, phase: str, interval: float, fetch: Callable[[str], dict[str, Any]]) -> None:
    while True:
        rows = state.batches()
        if stuck := [r for r in rows if r["status"] == "submitting"]:
            raise SystemExit(
                f"unreconciled submission in {state.batches_path}: {stuck}; use --adopt / --abandon-uncertain"
            )
        open_rows = [r for r in rows if r["role"] == role and r["status"] not in TERMINAL]
        if not open_rows:
            return
        for r in open_rows:
            b = fetch(r["id"])
            b = b["data"] if isinstance(b.get("data"), dict) else b
            r["status"] = b.get("status", r["status"])
            if r["status"] in TERMINAL:
                got = result_rows(b)
                state.add_results(got)
                usage = b.get("usage") or {}
                ledger({
                    "phase": phase, "event": "batch_done", "batch_id": r["id"], "role": role, "status": r["status"],
                    "n": r["n"], "returned": len(got), "errors": sum(1 for g in got if g["content"] is None),
                    "truncated": sum(1 for g in got if g.get("finish_reason") == "length"),
                    "usage": usage, "is_byok": usage.get("is_byok"),
                })  # fmt: skip
                print(
                    f"batch {r['id']} {r['status']}: {len(got)}/{r['n']} results, cost={usage.get('cost')}", flush=True
                )
            elif time.time() - r["submitted"] > 12 * 3600:
                print(f"WARN batch {r['id']} still {r['status']} after 12h", flush=True)
        state.save_batches(rows)
        if any(r["status"] not in TERMINAL for r in rows if r["role"] == role):
            time.sleep(interval)


async def run_sync(state: State, reqs: dict[str, dict[str, Any]], workers: int, phase: str, role: str) -> None:
    """Standard endpoint. Each response is persisted as it arrives; a POST is never retried (billing)."""
    sem = asyncio.Semaphore(workers)
    errors = 0

    async def one(cid: str) -> None:
        nonlocal errors
        async with sem:
            if STOP_FILE.exists():
                return
            try:
                body = await asyncio.to_thread(http, "POST", "/chat/completions", reqs[cid], 120, False)
                choice = body["choices"][0] if body.get("choices") else {}
                content = (choice.get("message") or {}).get("content")
                row = {"custom_id": cid, "content": content, "finish_reason": choice.get("finish_reason"),
                       "usage": body.get("usage"), "error": None if content is not None else body}  # fmt: skip
            except (SystemExit, Exception) as exc:
                row = {"custom_id": cid, "content": None, "finish_reason": None, "usage": None, "error": str(exc)[:300]}
            errors += row["content"] is None
            state.add_results([row])

    await asyncio.gather(*(one(c) for c in sorted(reqs)))
    if STOP_FILE.exists():
        raise SystemExit(f"STOP file present ({STOP_FILE}): stopped mid-round")
    ledger({"phase": phase, "event": "sync_done", "role": role, "n": len(reqs), "errors": errors})


# --------------------------------------------------------------------------- main


def build_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pred", required=True, help="predict dir name under results/locomo (read-only)")
    ap.add_argument("--tag", required=True, help="label for this configuration, e.g. gpt4omini")
    ap.add_argument("--model", required=True, help="answerer model slug")
    ap.add_argument("--judge-model", default=None, help="judge model slug (default: --model)")
    ap.add_argument("--reasoning", default=None, help="answerer reasoning effort (omits temperature)")
    ap.add_argument("--judge-reasoning", default=None, help="judge reasoning effort (omits temperature)")
    ap.add_argument("--rubric", choices=["harness", "strict"], default="harness", help="judge rubric (C3)")
    ap.add_argument("--cutoff", type=int, default=10)
    ap.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--max-questions", type=int, default=None, help="per conversation (canaries)")
    ap.add_argument("--qids", type=Path, default=None, help="file with one question id per line")
    ap.add_argument("--judge-only-from", default=None, help="judge the answers saved in this output dir")
    ap.add_argument("--memories", choices=["retrieved", "none", "oracle"], default="retrieved",
                    help="none: no-memory baseline; oracle: the gold evidence turns (diagnostics)")  # fmt: skip
    ap.add_argument("--mode", choices=["batch", "sync"], default="batch")
    ap.add_argument("--workers", type=int, default=8, help="sync mode concurrency")
    ap.add_argument("--chunk", type=int, default=500, help="requests per batch")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--budget", type=float, default=None, help="refuse a round whose estimate exceeds this ($)")
    ap.add_argument("--price", default=None, help="in,out $/M tokens for a model not in PRICES")
    ap.add_argument("--force-judge", action="store_true", help="judge despite a failed answer-quality gate")
    ap.add_argument("--retry-bad-answers", action="store_true",
                    help=f"re-ask empty or truncated answers (up to {ANSWER_ATTEMPTS} attempts each)")  # fmt: skip
    ap.add_argument("--adopt", default=None, help="attach this batch id to the uncertain submission, then resume")
    ap.add_argument("--abandon-uncertain", action="store_true", help="the uncertain submission was not created")
    ap.add_argument("--phase", default="adhoc")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.rubric == "strict" and not args.judge_only_from:
        ap.error("--rubric strict re-scores fixed answers: pass --judge-only-from")
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_args(argv)
    bench = args.bench_dir.expanduser()
    run, prompts = load_harness(bench)
    judge_model = args.judge_model or args.model
    res_root = bench / "results" / "locomo"
    pred_dir = res_root / args.pred
    suffix = ("" if args.memories == "retrieved" else f"__{args.memories}") + (
        "__strict" if args.rubric == "strict" else ""
    )
    out_dir = res_root / f"{args.pred}__{args.tag}{suffix}"
    label = f"top_{args.cutoff}"
    state = State(out_dir / "_batch" / label)

    dataset_path = bench / "datasets" / "locomo" / "locomo10.json"
    dataset = run.load_dataset(str(dataset_path))
    convs = [int(c) for c in args.conversations.split(",")]
    items = run.expected_locomo_question_items(dataset, convs, prompts.CATEGORIES_TO_EVALUATE, args.max_questions)
    if args.qids:
        keep = {q.strip() for q in args.qids.read_text().splitlines() if q.strip()}
        items = [it for it in items if it[0] in keep]
    ok, missing = run.locomo_predict_outputs_complete(str(pred_dir), items)
    if not ok:
        raise SystemExit(f"predictions incomplete in {pred_dir}: {len(missing)} missing, e.g. {missing[:5]}")

    preds = {qid: json.loads((pred_dir / f"{qid}.json").read_text()) for qid, *_ in items}
    qa_of = {qid: qa for qid, _, _, qa in items}
    conv_of = {qid: conv for qid, conv, _, _ in items}
    evidence = run.load_evidence_lookup(str(dataset_path)) if args.memories == "oracle" else {}

    def memories(qid: str) -> list[dict[str, Any]]:
        if args.memories == "none":
            return []
        if args.memories == "oracle":  # evidence text already carries "said on <session date>"
            refs = qa_of[qid].get("evidence", [])
            return [{"memory": evidence[(conv_of[qid], r)]} for r in refs if (conv_of[qid], r) in evidence]
        return list(preds[qid]["retrieval"]["search_results"])[: args.cutoff]

    manifest = {
        "protocol": PROTOCOL, "pred": args.pred, "cutoff": args.cutoff, "memories": args.memories,
        "model": args.model, "judge_model": judge_model, "reasoning": args.reasoning,
        "judge_reasoning": args.judge_reasoning, "rubric": args.rubric, "judge_only_from": args.judge_only_from,
        "inputs_sha": sha256(json.dumps({q: [preds[q]["question"], preds[q].get("reference_date"),
                                              preds[q].get("user_profile"), memories(q), qa_of[q].get("answer")]
                                          for q in sorted(preds)}, sort_keys=True, default=str).encode()),
        "source_answers_sha": sha256(json.dumps(
            {q: json.loads((res_root / args.judge_only_from / f"{q}.json").read_text())["cutoff_results"][label]
             .get("generated_answer") for q in sorted(preds)}, sort_keys=True).encode()) if args.judge_only_from else None,
        "prompts_sha": sha256((bench / "benchmarks" / "locomo" / "prompts.py").read_bytes()),
        "dataset_sha": sha256(dataset_path.read_bytes()),
    }  # fmt: skip

    with state.locked():
        state.check_manifest(manifest)
        return _run(args, state, run, prompts, preds, qa_of, memories, judge_model, pred_dir, out_dir, res_root, label)


def _run(args, state, run, prompts, preds, qa_of, memories, judge_model, pred_dir, out_dir, res_root, label) -> int:  # noqa: ANN001
    if args.adopt or args.abandon_uncertain:
        rows = state.batches()
        uncertain = [r for r in rows if r["status"] == "submitting"]
        if len(uncertain) != 1:
            raise SystemExit(f"expected exactly one uncertain submission, found {len(uncertain)}")
        rows.remove(uncertain[0])
        if args.adopt:
            rows.append({**uncertain[0], "id": args.adopt, "status": "in_progress"})
        state.save_batches(rows)
        ledger({"phase": args.phase, "event": "uncertain_resolved", "adopt": args.adopt})

    price_of: dict[str, tuple[float, float]] = dict(PRICES)
    if args.price:
        pin, pout = (float(x) for x in args.price.split(","))
        price_of[args.model] = price_of[judge_model] = (pin, pout)

    def send(role: str, model: str, reqs: dict[str, dict[str, Any]], mode: str | None = None) -> None:
        mode = mode or args.mode
        if not reqs:
            return
        if args.dry_run:
            path = state.root / f"dryrun_{role}.jsonl"
            path.write_text("".join(json.dumps({"custom_id": c, "body": reqs[c]}) + "\n" for c in sorted(reqs)))
            print(f"dry-run: {len(reqs)} {role} requests -> {path}")
            return
        if STOP_FILE.exists():
            raise SystemExit(f"STOP file present ({STOP_FILE}): not submitting")
        if model not in price_of:
            raise SystemExit(f"no price for {model}; pass --price in,out")
        estimate = estimate_cost(reqs, role, mode, price_of[model])
        info = key_info()
        remaining = info.get("limit_remaining") if info else None
        print(f"{role}: {len(reqs)} requests, estimated ${estimate:.2f} (key remaining {remaining})", flush=True)
        if args.budget is not None and estimate > args.budget:
            raise SystemExit(f"{role} round estimated ${estimate:.2f} exceeds --budget ${args.budget:.2f}")
        if remaining is not None and estimate * 1.5 > float(remaining):
            raise SystemExit(f"{role} round estimate ${estimate:.2f} x1.5 exceeds the key's remaining ${remaining}")
        before = info.get("usage") if info else None
        if mode == "sync":
            asyncio.run(run_sync(state, reqs, args.workers, args.phase, role))
        else:
            submit_batches(state, role, model, reqs, args.chunk, args.phase,
                           lambda p: http("POST", "/batches", p, retry=False))  # fmt: skip
            poll_batches(state, role, args.phase, args.poll, lambda bid: http("GET", f"/batches/{bid}"))
        after = key_usage()
        ledger({"phase": args.phase, "event": "round_spend", "role": role, "model": model, "mode": mode,
                "estimate": round(estimate, 4), "key_usage_before": before, "key_usage_after": after,
                "delta": (after - float(before)) if before is not None and after is not None else None})  # fmt: skip

    # Pick up batches a previous invocation left in flight before building new work.
    if not args.dry_run and args.mode == "batch":
        for role in ("answer", "judge"):
            poll_batches(state, role, args.phase, args.poll, lambda bid: http("GET", f"/batches/{bid}"))

    # ---- answer round
    answers: dict[str, dict[str, Any]] = {}
    answerer = args.model
    if args.judge_only_from:
        src = res_root / args.judge_only_from
        for qid in preds:
            cr = json.loads((src / f"{qid}.json").read_text())["cutoff_results"][label]
            answerer = cr.get("answerer_model", answerer)
            answers[qid] = {"answer": cr["generated_answer"], "had_marker": cr.get("had_answer_marker", True),
                            "empty": not cr["generated_answer"], "truncated": cr.get("answer_truncated", False)}  # fmt: skip
    else:
        done, tries = state.results(), state.attempts()
        reqs = {}
        for qid, d in preds.items():
            cid = f"{qid}:answer"
            row = done.get(cid, {})
            bad = row.get("content") is not None and (
                not parse_answer(row["content"])[0] or row.get("finish_reason") == "length"
            )
            if row.get("content") is None or (args.retry_bad_answers and bad and tries.get(cid, 0) < ANSWER_ATTEMPTS):
                gen = prompts.get_answer_generation_prompt(
                    d["question"],
                    memories(qid),
                    reference_date=d.get("reference_date"),
                    user_profile=d.get("user_profile"),
                )
                reqs[cid] = chat_body(args.model, "", gen, json_mode=False, reasoning=args.reasoning)
        send("answer", args.model, reqs)
        if args.dry_run:
            print("dry-run stops after the answer round (judge prompts need answers)")
            return 0
        done = state.results()
        for qid in preds:
            row = done.get(f"{qid}:answer")
            # A failed answer request is retried on the next run, never judged.
            if row is None or row.get("content") is None:
                continue
            ans, marker = parse_answer(row["content"])
            answers[qid] = {"answer": ans, "had_marker": marker, "empty": not ans,
                            "truncated": row.get("finish_reason") == "length"}  # fmt: skip
        bad = sum(a["empty"] or a["truncated"] for a in answers.values()) + (len(preds) - len(answers))
        if bad > 0.01 * len(preds) and not args.force_judge:
            ledger({"phase": args.phase, "event": "answer_gate_failed", "bad": bad, "n": len(preds)})
            raise SystemExit(f"answer quality gate: {bad}/{len(preds)} missing, empty or truncated; "
                             f"inspect, then rerun (retries failures) or pass --force-judge")  # fmt: skip

    # ---- judge round, retrying only malformed replies, as the stock client does
    def judge_id(qid: str) -> str:  # keyed to the answer text, so a re-asked answer is judged afresh
        return f"{qid}:judge:{sha256(answers[qid]['answer'].encode())[:12]}"

    sent: set[str] = set()
    for attempt in range(JUDGE_ATTEMPTS):
        done, tries, latest = state.results(), state.attempts(), state.latest()
        reqs = {}
        for qid, a in answers.items():
            cid = judge_id(qid)
            row = done.get(cid)
            if cid in sent and latest.get(cid, {}).get("content") is None:
                continue  # its last attempt in THIS run failed: outcome unknown, maybe billed; next run retries
            if (
                row is not None
                and row.get("content") is not None
                and (not judge_retryable(row["content"]) or tries.get(cid, 0) >= JUDGE_ATTEMPTS)
            ):
                continue
            qa = qa_of[qid]
            gold = prompts.preprocess_answer(qa["category"], str(qa["answer"]))
            if args.rubric == "strict":
                jp = STRICT_JUDGE_PROMPT.format(question=preds[qid]["question"], gold=gold, generated=a["answer"])
            else:
                jp = prompts.get_judge_prompt(qa["category"], preds[qid]["question"], gold, a["answer"])
            reqs[cid] = chat_body(judge_model, prompts.JUDGE_SYSTEM_PROMPT, jp, json_mode=True,
                                  reasoning=args.judge_reasoning)  # fmt: skip
        if not reqs:
            break
        # A handful of residual malformed replies is re-asked on the standard endpoint rather than
        # waiting another batch window.
        send("judge", judge_model, reqs, "sync" if attempt and len(reqs) <= 20 else None)
        sent |= set(reqs)
    done = state.results()

    # ---- write outputs in the runner's format, merging into any other cutoffs already judged
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(("answered", "empty_answer", "no_answer_marker", "answer_truncated", "judged",
                            "judge_error", "judge_truncated", "correct"), 0)  # fmt: skip
    for qid, d in preds.items():
        a = answers.get(qid)
        jrow = done.get(judge_id(qid)) if a is not None else None
        if a is None or jrow is None or jrow.get("content") is None:
            continue  # unfinished; reported as missing and retried by the next run
        verdict, reason = parse_judge(jrow["content"])
        counts["answered"] += 1
        counts["empty_answer"] += a["empty"]
        counts["no_answer_marker"] += not a["had_marker"]
        counts["answer_truncated"] += a["truncated"]
        counts["judged"] += 1
        counts["judge_error"] += verdict == "ERROR"
        counts["judge_truncated"] += jrow.get("finish_reason") == "length"
        counts["correct"] += verdict == "CORRECT"
        path = out_dir / f"{qid}.json"
        out = json.loads(path.read_text()) if path.exists() else dict(d)
        out.setdefault("cutoff_results", {})[label] = {
            "judgment": verdict, "score": 1.0 if verdict == "CORRECT" else 0.0, "generated_answer": a["answer"],
            "memories_evaluated": len(memories(qid)), "reason": reason, "had_answer_marker": a["had_marker"],
            "answer_truncated": a["truncated"], "answerer_model": answerer, "judge_model": judge_model,
            "memories_mode": args.memories, "rubric": args.rubric,
        }  # fmt: skip
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(path)
    counts = {"questions": len(preds), **counts, "missing": len(preds) - counts["judged"]}
    ledger({"phase": args.phase, "event": "summary", "pred": args.pred, "tag": args.tag, "cutoff": label, **counts})
    print(json.dumps({"out": str(out_dir), "cutoff": label, **counts}))
    bad = counts["judge_error"] + counts["judge_truncated"] + counts["answer_truncated"]
    if counts["judged"] and bad / counts["judged"] > 0.01:
        print(f"GATE FAILED: {bad}/{counts['judged']} judge errors or truncations (> 1%)", file=sys.stderr)
        return 2  # stop the runbook before any further spend
    return 1 if counts["missing"] else 0  # rerun the same command to retry what is missing


if __name__ == "__main__":
    raise SystemExit(main())
