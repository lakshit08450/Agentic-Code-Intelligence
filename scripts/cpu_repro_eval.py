"""Evaluate CPU reproducibility from the caches only (no encoding): for each pipeline whose CPU fp32
vectors are complete in cache/embeddings_cpu_fp32, compare top-10 agreement, NDCG@10 and recall@50 on
the 200 validation queries (seed 13) against the GPU-cache version. Incomplete pipelines are reported
as such. Writes results/cpu_repro.json (same fields as scripts/cpu_repro.py)."""

from __future__ import annotations

import copy
import json

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.stage1.search import top_k

PIPELINES = {"qwen3_only": "configs/stage1_final.yaml", "fusion_w0.3": "configs/fusion/qwen3_gemma_w0.3.yaml"}

dev = load_devset("val")
sel = np.sort(np.random.default_rng(13).choice(len(dev.query_ids), 200, replace=False))
qids = [dev.query_ids[i] for i in sel]
qtexts = [dev.query_texts[i] for i in sel]
qrels = {q: dev.qrels[q] for q in qids}
out = {"n_queries": 200, "corpus_docs": len(dev.corpus_texts), "pipelines": {}}
for name, path in PIPELINES.items():
    sides = {}
    try:
        for side, cache_dir in (("gpu", None), ("cpu", "cache/embeddings_cpu_fp32")):
            cfg = copy.deepcopy(load_cfg(path))
            cfg.on_cache_miss = "error"
            if cache_dir:
                cfg.cache_dir = cache_dir
            enc = PrePostPipelineEncoder(cfg)
            sides[side] = top_k(enc.embed(qtexts, True), enc.embed(dev.corpus_texts, False), 50)
    except RuntimeError as e:
        out["pipelines"][name] = {"status": "incomplete", "detail": str(e)[:200]}
        continue
    (gp, gs), (cp, cs) = sides["gpu"], sides["cpu"]
    mg = per_query(run_from_topk(qids, dev.corpus_ids, gp, gs), qrels, (10, 50))
    mc = per_query(run_from_topk(qids, dev.corpus_ids, cp, cs), qrels, (10, 50))
    out["pipelines"][name] = {
        "status": "complete",
        "top10_identical_order_share": float(np.mean([np.array_equal(gp[i, :10], cp[i, :10]) for i in range(len(qids))])),
        "top10_same_set_share": float(np.mean([set(gp[i, :10]) == set(cp[i, :10]) for i in range(len(qids))])),
        "top1_same_share": float(np.mean(gp[:, 0] == cp[:, 0])),
        "ndcg_at_10": {"gpu": float(mg["ndcg_at_10"].mean()), "cpu": float(mc["ndcg_at_10"].mean()),
                       "diff": float(mc["ndcg_at_10"].mean() - mg["ndcg_at_10"].mean())},
        "recall_at_50": {"gpu": float(mg["recall_at_50"].mean()), "cpu": float(mc["recall_at_50"].mean()),
                         "diff": float(mc["recall_at_50"].mean() - mg["recall_at_50"].mean())},
        "max_abs_score_diff_top50": float(np.abs(gs - cs).max()),
    }
(REPO_ROOT / "results" / "cpu_repro.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
