"""Monitor a hosted-judge benchmark phase and stop it on a tripwire.

Run it under the harness ``Monitor`` tool (or a terminal). It prints one line per state change,
so every line is worth reading::

    python watch.py --phase P1b --budget 6 --pred predicted_mem0-v6 --expected 154

Checks every ``--interval`` seconds:

* spend since the watcher started, from ``GET /api/v1/key`` (limit accounting, or usage +
  byok_usage): warn at 75% of --budget, stop at
  100%; stop if usage has been unreadable for ``--usage-outage`` seconds (fail closed)
* progress: new predictions OR new mem0 writes OR checkpoint updates; warn after 10 min idle,
  stop after 30 (ingest writes no prediction until a conversation is fully stored)
* prediction quality once 50 exist: empty ``search_results``, results without ``created_at``
  (warn > 0.5%, stop > 2%)
* ingestion checkpoints: ``total_chunks_failed`` (warn > 0, stop > 1% of chunks)
* mem0 writes (the shim's ``add_log.jsonl``; default: every ``~/.cache/trw-bench/state/*/``):
  failed writes (stop > 1%, at least 3), zero-memory extractions after 100 writes
  (warn > 25%, stop > 60%; small talk legitimately yields nothing)
* shim/run logs: Tracebacks and HTTP 429/5xx lines (warn on any, stop above 50 per check)
* ledger rows for this phase: batch not ``completed``, fewer results than requests, > 1% errors
  or truncations, uncertain submissions, failed answer gates, summaries with
  judge errors or truncations > 1% (missing questions only warn: rerun to retry them)
* load average (warn > 8; stop above ``--max-load`` when given, for local phases)

A hard stop writes ``~/.cache/trw-bench/STOP`` (batch_judge.py and phases.sh refuse to start
paid work while it exists) and sends SIGTERM to each recorded process group in
``~/.cache/trw-bench/<phase>.pids``. In-flight batches cannot be cancelled and finish on their
own. A heartbeat goes to ``~/.cache/trw-bench/watch.<phase>.heartbeat``. Exit 3 = hard stop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import batch_judge

ROOT = Path.home() / ".cache" / "trw-bench"
STOP_FILE = ROOT / "STOP"
LEDGER = ROOT / "ledger.jsonl"
LOG_BAD = re.compile(r"Traceback|\b(?:HTTP/\S+ |status[_ ]code[=: ]*)(?:429|5\d\d)\b|Too Many Requests")


def say(level: str, msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {level} {msg}", flush=True)


class Tail:
    """Incremental reader for a growing text file; only whole lines are consumed."""

    def __init__(self, path: Path, from_end: bool) -> None:
        self.path = path
        self.offset = path.stat().st_size if from_end and path.exists() else 0

    def lines(self) -> list[str]:
        if not self.path.exists() or self.path.stat().st_size <= self.offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1  # leave a torn last line for the next read
        self.offset += end
        return data[:end].decode(errors="replace").splitlines()


def jsonl(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # trw-fail-silent-allow: a line torn by a concurrent writer; Tail re-reads nothing, the writer's next rows stay readable
            continue
    return out


def predict_stats(d: Path) -> dict[str, Any]:
    files = list(d.glob("conv*_q*.json"))
    empty = no_date = total = 0
    for p in files:
        try:
            res = json.loads(p.read_text()).get("retrieval", {}).get("search_results", [])
        except (
            json.JSONDecodeError,
            OSError,
        ):  # trw-fail-silent-allow: a prediction file mid-write; it is counted on the next poll, and verify checks completeness
            continue
        empty += not res
        total += len(res)
        no_date += sum(1 for r in res if not r.get("created_at"))
    failed = chunks = 0
    checkpoints = list(d.glob("_ingestion_*.json")) + list(d.glob("_progress*.json"))
    for p in d.glob("_ingestion_*.json"):
        try:
            c = json.loads(p.read_text())
        except (
            json.JSONDecodeError,
            OSError,
        ):  # trw-fail-silent-allow: a checkpoint mid-write; re-read next poll, and verify fails a missing or failed checkpoint
            continue
        failed += int(c.get("total_chunks_failed", 0))
        chunks += int(c.get("total_chunks_processed", 0)) + int(c.get("total_chunks_failed", 0))
    newest = max((p.stat().st_mtime for p in files + checkpoints), default=0.0)
    return {"files": len(files), "empty": empty, "no_date": no_date, "results": total,
            "failed_chunks": failed, "chunks": chunks, "newest": newest}  # fmt: skip


def stop_processes(phase: str) -> None:
    pid_file = ROOT / f"{phase}.pids"
    if not pid_file.exists():
        return
    for token in pid_file.read_text().split():
        try:
            pid = int(token)
            # phases.sh starts each job as its own process-group leader; never signal a shared group.
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
                say("STOP", f"SIGTERM to process group {pid}")
            else:
                os.kill(pid, signal.SIGTERM)
                say("STOP", f"SIGTERM to pid {pid} (not a group leader)")
        except (ProcessLookupError, ValueError, PermissionError) as exc:
            say("WARN", f"could not signal {token}: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--budget", type=float, required=True, help="phase stop-at, dollars")
    ap.add_argument("--bench-dir", type=Path, default=ROOT / "memory-benchmarks")
    ap.add_argument("--pred", action="append", default=[], help="predict dir name(s) under results/locomo")
    ap.add_argument("--expected", type=int, default=1540, help="predictions each --pred should reach")
    ap.add_argument("--add-log", type=Path, action="append", default=None, help="shim add_log.jsonl path(s)")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--stall-warn", type=float, default=600)
    ap.add_argument("--stall-stop", type=float, default=1800)
    ap.add_argument("--usage-outage", type=float, default=600)
    ap.add_argument("--max-load", type=float, default=None, help="hard-stop load average (local phases)")
    ap.add_argument("--once", action="store_true", help="one check, then exit (tests, spot checks)")
    ap.add_argument("--from-start", action="store_true", help="read the whole ledger (audit a finished phase)")
    args = ap.parse_args(argv)

    ROOT.mkdir(parents=True, exist_ok=True)
    bench = args.bench_dir.expanduser()
    heartbeat = ROOT / f"watch.{args.phase}.heartbeat"
    heartbeat.unlink(missing_ok=True)  # not ready until every baseline below is captured
    started = time.time()
    # A budget needs a baseline: never publish readiness without one (each lookup: 15 s, no retry).
    start_usage = batch_judge.key_usage()
    for _ in range(4):
        if start_usage is not None:
            break
        time.sleep(15)
        start_usage = batch_judge.key_usage()
    if start_usage is None:
        say("STOP", "cannot read key usage for a spend baseline; not starting (no heartbeat written)")
        return 4
    usage_ok_at = time.time()
    say("INFO", f"phase {args.phase}: budget ${args.budget:.2f}, key usage at start "
                f"{'unknown' if start_usage is None else f'${start_usage:.4f}'}")  # fmt: skip
    add_logs = args.add_log or []
    tails: dict[Path, Tail] = {}
    ledger_tail = Tail(LEDGER, from_end=not args.from_start)
    last: dict[str, str] = {}
    adds: dict[str, int] = {"n": 0, "failed": 0, "zero": 0, "duplicate": 0}
    logs_at_start = {*bench.glob("logs/*.log"), *ROOT.glob("logs-*.txt")}
    last_error = ""
    add_seen_at = 0.0
    heartbeat.write_text(str(time.time()))  # ready: phases.sh may start paid work from here on

    def report(key: str, level: str, msg: str) -> None:
        if last.get(key) != f"{level} {msg}":
            last[key] = f"{level} {msg}"
            say(level, msg)

    while True:
        stops: list[str] = []

        # -- spend (fail closed on a long outage)
        usage = batch_judge.key_usage()
        if start_usage is None and usage is not None:
            start_usage = usage
        if usage is None or start_usage is None:
            if time.time() - usage_ok_at > args.usage_outage:
                stops.append(f"key usage unreadable for {(time.time() - usage_ok_at) / 60:.0f} min")
            else:
                report("spend", "WARN", "key usage unreadable")
        else:
            usage_ok_at = time.time()
            spend = usage - start_usage
            if spend >= args.budget:
                stops.append(f"spend ${spend:.2f} >= budget ${args.budget:.2f}")
            elif spend >= 0.75 * args.budget:
                report("spend", "WARN", f"spend ${spend:.2f} is {100 * spend / args.budget:.0f}% of budget")
            else:
                report("spend", "OK", f"spend ${spend:.2f} / ${args.budget:.2f}")

        # -- mem0 writes (incremental)
        for path in add_logs or list(ROOT.glob("state/*/add_log.jsonl")):
            tail = tails.setdefault(path, Tail(path, from_end=False))
            for row in jsonl(tail.lines()):
                if float(row.get("ts", 0)) < started:
                    continue  # an earlier run's writes say nothing about this phase
                if row.get("duplicate"):
                    adds["duplicate"] += 1  # the runner retried a slow write; the shim did not pay twice
                    continue
                adds["n"] += 1
                adds["failed"] += bool(row.get("error"))
                adds["zero"] += row.get("results") == 0
                add_seen_at = max(add_seen_at, float(row.get("ts", 0)))
                if row.get("error"):
                    last_error = str(row["error"])
        if adds["n"]:
            n = adds["n"]
            report(
                "adds",
                "OK",
                f"mem0 writes {n}: failed {adds['failed']}, zero-yield {100 * adds['zero'] / n:.0f}%, "
                f"client retries absorbed {adds['duplicate']}",
            )
            if adds["failed"] >= 3 and adds["failed"] / n > 0.01:
                stops.append(f"mem0 writes failing: {adds['failed']}/{n}, last: {last_error}")
            if n >= 100 and adds["zero"] / n > 0.60:
                stops.append(f"mem0 extracting nothing on {100 * adds['zero'] / n:.0f}% of writes")
            elif n >= 100 and adds["zero"] / n > 0.25:
                report("adds.zero", "WARN", f"mem0 zero-yield writes {100 * adds['zero'] / n:.0f}%")

        # -- predictions: progress, quality, checkpoints
        for name in args.pred:
            st = predict_stats(bench / "results" / "locomo" / name)
            report(f"{name}.progress", "OK", f"{name}: {st['files']}/{args.expected} predictions")
            idle = time.time() - max(st["newest"], add_seen_at, started)
            if st["files"] < args.expected:
                if idle > args.stall_stop:
                    stops.append(f"{name}: no progress for {idle / 60:.0f} min")
                elif idle > args.stall_warn:
                    report(f"{name}.stall", "WARN", f"{name}: no progress for {idle / 60:.0f} min")
            if st["files"] >= 50:
                e, nd = st["empty"] / st["files"], st["no_date"] / max(st["results"], 1)
                if e > 0.02 or nd > 0.02:
                    stops.append(f"{name}: empty retrievals {100 * e:.1f}%, undated results {100 * nd:.1f}%")
                elif e > 0.005 or nd > 0.005:
                    report(f"{name}.quality", "WARN", f"{name}: empty {100 * e:.1f}%, undated {100 * nd:.1f}%")
            if st["failed_chunks"]:
                frac = st["failed_chunks"] / max(st["chunks"], 1)
                if frac > 0.01:
                    stops.append(f"{name}: {st['failed_chunks']} failed ingest chunks ({100 * frac:.1f}%)")
                else:
                    report(f"{name}.chunks", "WARN", f"{name}: {st['failed_chunks']} failed ingest chunks")

        # -- logs
        for log in [*bench.glob("logs/*.log"), *ROOT.glob("logs-*.txt")]:
            # logs that existed at start: only new lines; logs created since: from their first line
            tail = tails.setdefault(log, Tail(log, from_end=log in logs_at_start))
            bad = [line for line in tail.lines() if LOG_BAD.search(line)]
            if len(bad) > 50:
                stops.append(f"{log.name}: {len(bad)} new error lines, last: {bad[-1][:160]}")
            elif bad:
                say("WARN", f"{log.name}: {len(bad)} new error line(s), last: {bad[-1][:160]}")

        # -- ledger rows for this phase
        for r in jsonl(ledger_tail.lines()):
            if r.get("phase") != args.phase:
                continue
            ev = r.get("event")
            if ev == "batch_done":
                n = max(int(r.get("n", 0)), 1)
                msg = (f"batch {r.get('batch_id')} {r.get('status')}: {r.get('returned')}/{r.get('n')} back, "
                       f"errors {r.get('errors')}, truncated {r.get('truncated')}, "
                       f"cost {(r.get('usage') or {}).get('cost')}")  # fmt: skip
                bad_n = int(r.get("errors") or 0) + int(r.get("truncated") or 0)
                # at least 3 bad rows, so one error in a 40-request remainder chunk is not a stop
                bad_batch = (r.get("status") != "completed" or int(r.get("returned", 0)) < int(r.get("n", 0))
                             or (bad_n >= 3 and bad_n / n > 0.01))  # fmt: skip
                if r.get("is_byok"):
                    # Provider-billed: usage.cost is not the real charge. Spend is tracked from the key's
                    # own accounting (usage + byok_usage) instead, so this is a note, not a stop.
                    report(f"byok.{r.get('batch_id')}", "WARN", f"batch {r.get('batch_id')} billed via BYOK; "
                           f"cost comes from the key's spend, not usage.cost")  # fmt: skip
                if bad_batch:
                    stops.append(msg)
                else:
                    say("OK", msg)
            elif ev == "summary":
                n = max(int(r.get("questions", 0)), 1)
                msg = (f"{r.get('pred')} [{r.get('tag')} {r.get('cutoff')}]: {r.get('correct')}/{r.get('judged')} "
                       f"correct, judge errors {r.get('judge_error')}, truncated "
                       f"{r.get('answer_truncated')}/{r.get('judge_truncated')}, missing {r.get('missing')}")  # fmt: skip
                errs = sum(int(r.get(k) or 0) for k in ("judge_error", "answer_truncated", "judge_truncated"))
                if errs / n > 0.01:
                    stops.append(msg)
                elif r.get("missing"):
                    say("WARN", msg + " (rerun the same command to retry)")
                else:
                    say("OK", msg)
            elif ev in ("submit_uncertain", "answer_gate_failed"):
                stops.append(f"{ev}: {json.dumps({k: v for k, v in r.items() if k not in ('ts', 'phase')})}")

        load = os.getloadavg()[0]
        if args.max_load is not None and load > args.max_load:
            stops.append(f"load average {load:.1f}")
        elif load > 8:
            report("load", "WARN", f"load average {load:.1f}")

        heartbeat.write_text(str(time.time()))
        if stops:
            STOP_FILE.write_text(json.dumps({"phase": args.phase, "reasons": stops, "ts": time.time()}))
            for s in stops:
                say("STOP", s)
            stop_processes(args.phase)
            return 3
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
