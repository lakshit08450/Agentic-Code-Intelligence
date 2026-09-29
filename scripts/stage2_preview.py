"""Diagnostic only (not a tuning run): Stage 2 re-rank preview on validation stdin queries with ONE fixed,
untuned rule (score = cosine + 1.0*[PASS]), from the exec cache only (never runs code).
Writes results/stage2_preview.json. The real grid + G3 gate is tune_stage2.py (Phase 2b)."""
import collections, json
import numpy as np
from codeintel.common.config import REPO_ROOT
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.bootstrap import paired_bootstrap
from codeintel.eval.devset import load_devset
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import samples_for
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gap", REPO_ROOT / "scripts" / "gap_analysis.py"); gap = importlib.util.module_from_spec(spec); spec.loader.exec_module(gap)

dev = load_devset("val")
z = np.load(topk_path("qwen3-embedding-0.6b")); pos, cos = z["pos"][:, :50], z["scores"][:, :50]
cache = ExecCache(); idx = {d: i for i, d in enumerate(dev.corpus_ids)}
sel = [qi for qi in range(len(dev.query_ids)) if samples_for(dev.query_texts[qi]).kind == "stdin"]
qids = [dev.query_ids[qi] for qi in sel]
new_scores = cos[sel].copy()
fam_gold = collections.defaultdict(collections.Counter)
for r, qi in enumerate(sel):
    for c in range(50):
        v = gap.cached_verdict(dev.query_texts[qi], dev.corpus_texts[pos[qi, c]], cache)
        new_scores[r, c] += 1.0 * (v == "PASS")
g = json.load(open(REPO_ROOT / "results" / "gold_outcomes_v3_newparser.json", encoding="utf-8"))["per_query"]
for qi, q in enumerate(dev.query_ids):
    fam_gold[gap.family(dev.query_texts[qi])][g[q]["outcome"]] += 1
qrels = {q: dev.qrels[q] for q in qids}
s1 = per_query(run_from_topk(qids, dev.corpus_ids, pos[sel], cos[sel]), qrels, (1, 10, 50))
s2 = per_query(run_from_topk(qids, dev.corpus_ids, pos[sel], new_scores), qrels, (1, 10, 50))
out = {
    "n_stdin_queries": len(sel), "rule": "cosine + 1.0*PASS (fixed, untuned; diagnostic)",
    "stage1": {m: float(s1[m].mean()) for m in ("ndcg_at_10", "mrr_at_10", "recall_at_1")},
    "stage2_preview": {m: float(s2[m].mean()) for m in ("ndcg_at_10", "mrr_at_10", "recall_at_1")},
    "bootstrap_ndcg_at_10": paired_bootstrap(s1["ndcg_at_10"], s2["ndcg_at_10"]),
    "gold_outcome_by_family": {f: dict(c) for f, c in fam_gold.items()},
}
(REPO_ROOT / "results" / "stage2_preview.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
