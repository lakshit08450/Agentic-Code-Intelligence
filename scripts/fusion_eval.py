"""Fusion step 3 (validation only): EmbeddingGemma alone and Qwen3+Gemma fusion for w in {0.3..0.7}.

Reads top-100 candidates written by `codeintel.eval.bakeoff metrics` for each config. Reports
NDCG@10, MRR@10, recall@10/20/50 on all 1,000 validation queries and on the stdin subset weighted
to the test mix. Picks w by weighted recall@50 on the stdin subset (Stage 2 depth), recall@20 as
tie-break, with the G3 paired bootstrap against Qwen3 alone. Writes results/fusion_stage1.json.
"""

from __future__ import annotations

import collections
import json

import numpy as np

from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.devset import load_devset
from codeintel.eval.families import TEST_MIX, family
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.eval.tune_stage2 import weighted_bootstrap
from codeintel.stage2.verifier import samples_for

NAMES = ["qwen3-embedding-0.6b", "embeddinggemma-300m"] + [f"fusion-qwen3-gemma-w{w}" for w in (0.3, 0.4, 0.5, 0.6, 0.7)]
K = (10, 20, 50)

dev = load_devset("val")
stdin = np.array([samples_for(t).kind == "stdin" for t in dev.query_texts])
fams = [family(t) for t, s in zip(dev.query_texts, stdin) if s]
share = collections.Counter(fams)
w_stdin = np.array([TEST_MIX.get(f, 0.0) / (share[f] / len(fams)) for f in fams])
w_stdin /= w_stdin.mean()

pq = {}
for name in NAMES:
    z = np.load(topk_path(name))
    pq[name] = per_query(run_from_topk(dev.query_ids, dev.corpus_ids, z["pos"], z["scores"]), dev.qrels, K)

report = {"n_all": len(dev.query_ids), "n_stdin": int(stdin.sum()), "models": {}}
for name in NAMES:
    m = pq[name]
    wm = lambda v: float((v[stdin] * w_stdin).sum() / w_stdin.sum())  # noqa: E731
    report["models"][name] = {
        "all": {k: float(m[k].mean()) for k in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_20", "recall_at_50")},
        "stdin_weighted": {k: wm(m[k]) for k in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_20", "recall_at_50")},
    }
fusions = [n for n in NAMES if n.startswith("fusion")]
best = max(fusions, key=lambda n: (report["models"][n]["stdin_weighted"]["recall_at_50"], report["models"][n]["stdin_weighted"]["recall_at_20"]))
base = pq["qwen3-embedding-0.6b"]
g3 = {m: weighted_bootstrap(base[m][stdin], pq[best][m][stdin], w_stdin) for m in ("recall_at_50", "recall_at_20", "ndcg_at_10")}
report["chosen"] = best
report["G3_vs_qwen3_stdin_weighted"] = g3
report["G3_pass"] = bool(g3["recall_at_50"]["excludes_zero"] and g3["recall_at_50"]["delta"] > 0)
(REPO_ROOT / "results" / "fusion_stage1.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
for n in NAMES:
    a, s = report["models"][n]["all"], report["models"][n]["stdin_weighted"]
    print(f"{n:28s} all: ndcg10 {a['ndcg_at_10']:.4f} mrr10 {a['mrr_at_10']:.4f} r10 {a['recall_at_10']:.3f} r20 {a['recall_at_20']:.3f} r50 {a['recall_at_50']:.3f} | "
          f"stdin-w: ndcg10 {s['ndcg_at_10']:.4f} r10 {s['recall_at_10']:.3f} r20 {s['recall_at_20']:.3f} r50 {s['recall_at_50']:.3f}")
print("chosen:", best, "G3:", json.dumps(g3), "pass:", report["G3_pass"])
