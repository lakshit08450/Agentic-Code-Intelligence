"""Stage 2 grid on validation (plan 11.5) with the G3 gate. Never executes code (exec cache only).

Scope: validation queries whose statement has stdin samples (the test split is 97.9% such statements).
Weighting: each query is weighted by test_share(family) / validation_share(family) over the format
families of the stdin subset (label-free statement-text families, results/gap_analysis.json item 3),
so Codeforces/AtCoder dominate as they do on test.

    python -m codeintel.eval.tune_stage2 --model qwen3-embedding-0.6b --k-for-test 20
Writes results/stage2_grid.csv, results/stage2_tuning.json, configs/stage2_final.yaml.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
from dataclasses import asdict, replace

import numpy as np
import yaml

from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.bootstrap import paired_bootstrap
from codeintel.eval.devset import load_devset
from codeintel.eval.families import TEST_MIX, family
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.stage2.reranker import Stage2Cfg, stage2_score
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import samples_for, verify

GRID = {
    "k": [10, 20, 50],
    "w_pass": [0.1, 0.3, 1.0],
    "w_wrong": [0.0, 0.05, 0.1],
    "w_err": [0.0, 0.05],
    "trivial_pass_factor": [1.0, 0.5, 0.1],
}


def weighted_bootstrap(a, b, w, n=1000, seed=13):
    a, b, w = map(np.asarray, (a, b, w))
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    delta = float((d * w).sum() / w.sum())
    return {"delta": delta, "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-embedding-0.6b")
    ap.add_argument("--k-for-test", type=int, default=20, help="k the test run can afford; the frozen config uses it")
    a = ap.parse_args()

    dev = load_devset("val")
    z = np.load(topk_path(a.model))
    pos, cos = z["pos"][:, :50], z["scores"][:, :50]
    sel = [qi for qi in range(len(dev.query_ids)) if samples_for(dev.query_texts[qi]).kind == "stdin"]
    qids = [dev.query_ids[qi] for qi in sel]
    qrels = {q: dev.qrels[q] for q in qids}
    fams = [family(dev.query_texts[qi]) for qi in sel]
    val_share = collections.Counter(fams)
    w = np.array([TEST_MIX.get(f, 0.0) / (val_share[f] / len(fams)) for f in fams])
    w = w / w.mean()

    cache = ExecCache()
    outcomes = np.empty((len(sel), 50), dtype=object)
    for r, qi in enumerate(sel):
        for c in range(50):
            outcomes[r, c] = verify(dev.query_texts[qi], dev.corpus_texts[pos[qi, c]], cache, cache_only=True).outcome
    missing = int((outcomes == "MISSING").sum())
    if missing:
        raise SystemExit(f"{missing} pairs not in the exec cache; run `ceiling topk` first")

    def evaluate(cfg: Stage2Cfg):
        k = cfg.k
        sc = np.array([[stage2_score(cos[qi, c], outcomes[r, c], dev.query_texts[qi], cfg) for c in range(k)] for r, qi in enumerate(sel)])
        pq = per_query(run_from_topk(qids, dev.corpus_ids, pos[sel, :k], sc), qrels, (1, 10))
        return pq

    base = per_query(run_from_topk(qids, dev.corpus_ids, pos[sel], cos[sel]), qrels, (1, 10))
    wmean = lambda v: float((v * w).sum() / w.sum())  # noqa: E731
    rows, best_by_k = [], {}
    for vals in itertools.product(*GRID.values()):
        cfg = Stage2Cfg(**dict(zip(GRID.keys(), vals)), multi_answer_neutral=True)
        pq = evaluate(cfg)
        row = dict(zip(GRID.keys(), vals)) | {
            "w_ndcg_at_10": wmean(pq["ndcg_at_10"]), "w_mrr_at_10": wmean(pq["mrr_at_10"]),
            "ndcg_at_10": float(pq["ndcg_at_10"].mean()), "mrr_at_10": float(pq["mrr_at_10"].mean()),
        }
        rows.append(row)
        if cfg.k not in best_by_k or row["w_ndcg_at_10"] > best_by_k[cfg.k][0]["w_ndcg_at_10"]:
            best_by_k[cfg.k] = (row, cfg, pq)

    with open(REPO_ROOT / "results" / "stage2_grid.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    report = {
        "n_queries": len(sel), "families_validation_stdin": dict(val_share), "weights": "test_share/val_share, mean 1",
        "stage1": {"w_ndcg_at_10": wmean(base["ndcg_at_10"]), "w_mrr_at_10": wmean(base["mrr_at_10"]),
                   "ndcg_at_10": float(base["ndcg_at_10"].mean()), "mrr_at_10": float(base["mrr_at_10"].mean())},
        "best_by_k": {},
    }
    for k, (row, cfg, pq) in sorted(best_by_k.items()):
        g3 = weighted_bootstrap(base["ndcg_at_10"], pq["ndcg_at_10"], w)
        g3_mrr = weighted_bootstrap(base["mrr_at_10"], pq["mrr_at_10"], w)
        abl = evaluate(replace(cfg, multi_answer_neutral=False))
        report["best_by_k"][k] = {
            "config": asdict(cfg), "metrics": row, "G3_ndcg_at_10": g3, "G3_mrr_at_10": g3_mrr,
            "unweighted_G3_ndcg_at_10": paired_bootstrap(base["ndcg_at_10"], pq["ndcg_at_10"]),
            "ablation_fix3_off_w_ndcg_at_10": wmean(abl["ndcg_at_10"]),
            "ablation_fix4_off_w_ndcg_at_10": wmean(evaluate(replace(cfg, trivial_pass_factor=1.0))["ndcg_at_10"]),
        }
    chosen = report["best_by_k"][a.k_for_test]
    report["chosen_for_test"] = {"k": a.k_for_test, "passes_G3": chosen["G3_ndcg_at_10"]["excludes_zero"] and chosen["G3_ndcg_at_10"]["delta"] > 0}
    (REPO_ROOT / "results" / "stage2_tuning.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if report["chosen_for_test"]["passes_G3"]:
        frozen = chosen["config"] | {"stage1_config": "configs/stage1_final.yaml"}
        (REPO_ROOT / "configs" / "stage2_final.yaml").write_text(
            "# FROZEN by tune_stage2.py (validation stdin subset, test-mix weighted, G3 passed). Do not edit.\n"
            + yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
