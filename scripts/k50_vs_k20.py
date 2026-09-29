"""Paired bootstrap of Stage 2 k=50 vs k=20 at the frozen weights, validation stdin subset,
test-mix weighted (same scope and weights as tune_stage2). Exec cache only; never runs code.
Writes results/k50_vs_k20.json."""

from __future__ import annotations

import collections
import json
from dataclasses import replace

import numpy as np

from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.bootstrap import paired_bootstrap
from codeintel.eval.devset import load_devset
from codeintel.eval.families import TEST_MIX, family
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.eval.tune_stage2 import weighted_bootstrap
from codeintel.stage2.reranker import load_stage2_cfg, stage2_score
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import samples_for, verify

dev = load_devset("val")
z = np.load(topk_path("qwen3-embedding-0.6b"))
pos, cos = z["pos"][:, :50], z["scores"][:, :50]
sel = [qi for qi in range(len(dev.query_ids)) if samples_for(dev.query_texts[qi]).kind == "stdin"]
qids = [dev.query_ids[qi] for qi in sel]
qrels = {q: dev.qrels[q] for q in qids}
fams = [family(dev.query_texts[qi]) for qi in sel]
share = collections.Counter(fams)
w = np.array([TEST_MIX.get(f, 0.0) / (share[f] / len(fams)) for f in fams])
w = w / w.mean()
cache = ExecCache()
out = np.array([[verify(dev.query_texts[qi], dev.corpus_texts[pos[qi, c]], cache, cache_only=True).outcome for c in range(50)] for qi in sel])
assert not (out == "MISSING").any()

frozen = load_stage2_cfg("configs/stage2_final.yaml")
res = {}
for k in (20, 50):
    cfg = replace(frozen, k=k)
    sc = np.array([[stage2_score(cos[qi, c], out[r, c], dev.query_texts[qi], cfg) for c in range(k)] for r, qi in enumerate(sel)])
    res[k] = per_query(run_from_topk(qids, dev.corpus_ids, pos[sel, :k], sc), qrels, (10,))
report = {
    "frozen_weights": {k: v for k, v in vars(frozen).items() if k not in ("k", "workers", "stage1_config")},
    "n_queries": len(sel),
    "w_ndcg_at_10": {k: float((res[k]["ndcg_at_10"] * w).sum() / w.sum()) for k in res},
    "paired_weighted_bootstrap_k50_minus_k20_ndcg_at_10": weighted_bootstrap(res[20]["ndcg_at_10"], res[50]["ndcg_at_10"], w),
    "paired_weighted_bootstrap_k50_minus_k20_mrr_at_10": weighted_bootstrap(res[20]["mrr_at_10"], res[50]["mrr_at_10"], w),
    "paired_unweighted_bootstrap_k50_minus_k20_ndcg_at_10": paired_bootstrap(res[20]["ndcg_at_10"], res[50]["ndcg_at_10"]),
    "queries_changed": int((res[50]["ndcg_at_10"] != res[20]["ndcg_at_10"]).sum()),
    "queries_improved": int((res[50]["ndcg_at_10"] > res[20]["ndcg_at_10"]).sum()),
    "queries_worse": int((res[50]["ndcg_at_10"] < res[20]["ndcg_at_10"]).sum()),
}
(REPO_ROOT / "results" / "k50_vs_k20.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
