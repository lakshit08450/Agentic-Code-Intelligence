"""Stage 2 executions on validation (plan 2a.2 and the G2 inputs). All runs go through the sandbox (R4).

    # sandbox throughput on N runs (stage-1 top-k pairs of the first validation queries)
    python -m codeintel.eval.ceiling bench --model gte-modernbert-base --runs 500 --workers 12
    # gold solution on its own parsed samples -> results/gold_outcomes.json
    python -m codeintel.eval.ceiling gold --workers 12
    # every (query, top-k doc) pair for all validation queries; resumable via cache/exec
    python -m codeintel.eval.ceiling topk --model gte-modernbert-base --k 100 --workers 12
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.devset import load_devset
from codeintel.stage2.sampleio import parse
from codeintel.stage2.sandbox import SANDBOX_VERSION, ExecCache, run_program
from codeintel.stage2.verifier import samples_for, verify

log = logging.getLogger("ceiling")
RESULTS = REPO_ROOT / "results"


def _check_sandbox_ready() -> None:
    """Refuse to run APPS code unless the hostile-program tests pass (plan 11.3)."""
    import pytest

    rc = pytest.main(["-q", "-p", "no:cacheprovider", str(REPO_ROOT / "tests" / "test_sandbox.py")])
    if rc != 0:
        raise SystemExit("tests/test_sandbox.py failed: not running any APPS code")


def bench(model: str, runs: int, workers: int) -> None:
    dev = load_devset("val")
    pos = np.load(topk_path(model))["pos"]
    jobs = []
    for qi in range(len(dev.query_ids)):
        s = parse(dev.query_texts[qi])
        if s.kind != "stdin":
            continue
        for p in pos[qi, :10]:
            for inp, _ in s.pairs:
                jobs.append((dev.corpus_texts[p], inp))
        if len(jobs) >= runs:
            break
    jobs = jobs[:runs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda j: run_program(*j), jobs))  # uncached on purpose
    dt = time.perf_counter() - t0
    # re-run every TIMEOUT alone: those that no longer time out were caused by parallel load
    t1 = time.perf_counter()
    timeouts = [i for i, r in enumerate(res) if r.status == "TIMEOUT"]
    serial = {i: run_program(*jobs[i]) for i in timeouts}
    rerun_s = time.perf_counter() - t1
    spurious = sum(1 for r in serial.values() if r.status != "TIMEOUT")
    c = collections.Counter(r.status + ":" + r.reason for r in res)
    walls = np.array([r.wall_s for r in res])
    out = {
        "runs": len(res), "workers": workers, "seconds": round(dt, 2), "runs_per_s": round(len(res) / dt, 2),
        "timeouts_parallel": len(timeouts), "timeouts_spurious": spurious, "serial_rerun_s": round(rerun_s, 2),
        "effective_runs_per_s": round(len(res) / (dt + rerun_s), 2),
        "wall_p50": float(np.percentile(walls, 50)), "wall_p95": float(np.percentile(walls, 95)),
        "status_counts": dict(c.most_common()), "sandbox_version": SANDBOX_VERSION,
    }
    log.info("bench: %s", out)
    (RESULTS / f"sandbox_bench_w{workers}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def gold(workers: int) -> None:
    dev = load_devset("val")
    cache = ExecCache()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(lambda qg: verify(qg[0], qg[1], cache), zip(dev.query_texts, dev.gold_texts)))
    dt = time.perf_counter() - t0
    per_query = {}
    for q, qt, v in zip(dev.query_ids, dev.query_texts, verdicts):
        per_query[q] = {"outcome": v.outcome, "reason": v.reason, "n_samples": len(samples_for(qt).pairs)}
    dist = collections.Counter(v.outcome for v in verdicts)
    reasons = collections.Counter(f"{v.outcome}:{v.reason}" for v in verdicts)
    n_stdin = sum(1 for qt in dev.query_texts if samples_for(qt).kind == "stdin")
    summary = {
        "n_queries": len(verdicts), "seconds": round(dt, 1), "sandbox_version": SANDBOX_VERSION,
        "outcomes": dict(dist.most_common()), "reasons": dict(reasons.most_common()),
        "pass_rate_all": dist["PASS"] / len(verdicts),
        "pass_rate_among_stdin_parsed": dist["PASS"] / max(1, n_stdin), "n_stdin_parsed": n_stdin,
    }
    (RESULTS / "gold_outcomes.json").write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=1), encoding="utf-8")
    log.info("gold: %s", summary)


def topk(model: str, k: int, workers: int) -> None:
    dev = load_devset("val")
    pos = np.load(topk_path(model))["pos"][:, :k]
    cache = ExecCache()
    pairs = [
        (qi, int(p))
        for qi in range(len(dev.query_ids))
        if samples_for(dev.query_texts[qi]).kind == "stdin"
        for p in pos[qi]
    ]
    log.info("topk: %d pairs (%d stdin queries x k=%d), exec cache has %d runs", len(pairs), len(pairs) // k, k, len(cache))
    t0 = time.perf_counter()
    done, runs = 0, 0
    counts = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(verify, dev.query_texts[qi], dev.corpus_texts[p], cache) for qi, p in pairs]
        for f in as_completed(futs):
            v = f.result()
            done += 1
            runs += v.runs
            counts[v.outcome] += 1
            if done % 2000 == 0 or done == len(pairs):
                el = time.perf_counter() - t0
                log.info(
                    "topk progress %d/%d pairs, %d new runs, %.1f runs/s, eta %.0f min, %s",
                    done, len(pairs), runs, runs / el, (len(pairs) - done) * el / done / 60, dict(counts),
                )
    log.info("topk finished in %.1f min; exec cache has %d runs", (time.perf_counter() - t0) / 60, len(cache))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "gold", "topk"])
    ap.add_argument("--model", default="gte-modernbert-base")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    _check_sandbox_ready()
    {"bench": lambda: bench(a.model, a.runs, a.workers), "gold": lambda: gold(a.workers),
     "topk": lambda: topk(a.model, a.k, a.workers)}[a.mode]()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
