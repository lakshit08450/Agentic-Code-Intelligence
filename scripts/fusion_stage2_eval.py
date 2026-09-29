"""Fusion step 4 (validation only): Stage 2 with the frozen weights on fusion candidates vs Qwen3 candidates,
at k=50 (decision) and k=20 (reference). Stdin subset, test-mix weighted, paired bootstrap. Exec cache only.
Writes results/fusion_stage2.json."""

from __future__ import annotations

import collections
import json
from dataclasses import replace

import numpy as np

from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.devset import load_devset
from codeintel.eval.families import TEST_MIX, family
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.eval.tune_stage2 import weighted_bootstrap
from codeintel.stage2.reranker import load_stage2_cfg, stage2_score
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import samples_for, verify

dev = load_devset("val")
sel = [qi for qi in range(len(dev.query_ids)) if samples_for(dev.query_texts[qi]).kind == "stdin"]
qids = [dev.query_ids[qi] for qi in sel]
qrels = {q: dev.qrels[q] for q in qids}
fams = [family(dev.query_texts[qi]) for qi in sel]
share = collections.Counter(fams)
w = np.array([TEST_MIX.get(f, 0.0) / (share[f] / len(fams)) for f in fams])
w /= w.mean()
cache = ExecCache()
frozen = load_stage2_cfg("configs/stage2_final.yaml")


def stage2(model: str, k: int) -> dict:
    z = np.load(topk_path(model))
    pos, cos = z["pos"][:, :k], z["scores"][:, :k]
    cfg = replace(frozen, k=k)
    sc = np.zeros((len(sel), k))
    missing = 0
    for r, qi in enumerate(sel):
        for c in range(k):
            v = verify(dev.query_texts[qi], dev.corpus_texts[pos[qi, c]], cache, cache_only=True).outcome
            missing += v == "MISSING"
            sc[r, c] = stage2_score(cos[qi, c], v, dev.query_texts[qi], cfg)
    if missing:
        raise SystemExit(f"{model} k={k}: {missing} pairs not cached")
    return per_query(run_from_topk(qids, dev.corpus_ids, pos[sel], sc), qrels, (1, 10))


wm = lambda v: float((v * w).sum() / w.sum())  # noqa: E731
report = {"n_stdin": len(sel), "frozen_weights": {k: v for k, v in vars(frozen).items() if k not in ("k", "workers", "stage1_config")}}
for k in (50, 20):
    q, f = stage2("qwen3-embedding-0.6b", k), stage2("fusion-qwen3-gemma-w0.3", k)
    report[f"k{k}"] = {
        "qwen3_stage2": {"w_ndcg_at_10": wm(q["ndcg_at_10"]), "w_mrr_at_10": wm(q["mrr_at_10"]), "w_recall_at_1": wm(q["recall_at_1"])},
        "fusion_stage2": {"w_ndcg_at_10": wm(f["ndcg_at_10"]), "w_mrr_at_10": wm(f["mrr_at_10"]), "w_recall_at_1": wm(f["recall_at_1"])},
        "paired_bootstrap_fusion_minus_qwen3_ndcg_at_10": weighted_bootstrap(q["ndcg_at_10"], f["ndcg_at_10"], w),
        "paired_bootstrap_fusion_minus_qwen3_mrr_at_10": weighted_bootstrap(q["mrr_at_10"], f["mrr_at_10"], w),
        "queries_improved": int((f["ndcg_at_10"] > q["ndcg_at_10"]).sum()),
        "queries_worse": int((f["ndcg_at_10"] < q["ndcg_at_10"]).sum()),
    }
g = report["k50"]["paired_bootstrap_fusion_minus_qwen3_ndcg_at_10"]
report["condition_a_pass_k50"] = bool(g["excludes_zero"] and g["delta"] > 0)
g20 = report["k20"]["paired_bootstrap_fusion_minus_qwen3_ndcg_at_10"]
report["condition_a_pass_k20"] = bool(g20["excludes_zero"] and g20["delta"] > 0)
(REPO_ROOT / "results" / "fusion_stage2.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
