"""CPU reproducibility of Stage 1 (P's request, 30 Sep): re-encode the full corpus and 200 validation
queries on CPU in fp32 (separate cache dir, the GPU cache is not touched) for Qwen3 and EmbeddingGemma,
then compare against the GPU-cache rankings for the Qwen3-only and the fusion (w=0.3) pipelines:
% of queries with identical top-10, NDCG@10 and recall@50 differences. Writes results/cpu_repro.json.
"""

from __future__ import annotations

import copy
import json
import logging
import time

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset
from codeintel.eval.metrics import per_query, run_from_topk
from codeintel.stage1.encoder import PrePostPipelineEncoder, l2
from codeintel.stage1.search import top_k

log = logging.getLogger("cpu_repro")
CPU_CACHE = "cache/embeddings_cpu_fp32"
MODELS = {"qwen3": "configs/stage1_final.yaml", "gemma": "configs/bakeoff/embeddinggemma_300m.yaml"}


def main() -> None:
    dev = load_devset("val")
    sel = np.sort(np.random.default_rng(13).choice(len(dev.query_ids), 200, replace=False))
    qids = [dev.query_ids[i] for i in sel]
    qtexts = [dev.query_texts[i] for i in sel]
    qrels = {q: dev.qrels[q] for q in qids}
    vecs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {"gpu": {}, "cpu": {}}
    timing = {}
    for name, path in MODELS.items():
        gcfg = load_cfg(path)
        gcfg.on_cache_miss = "error"
        g = PrePostPipelineEncoder(gcfg)
        vecs["gpu"][name] = (g.embed(qtexts, True), g.embed(dev.corpus_texts, False))
        ccfg = copy.deepcopy(load_cfg(path))
        ccfg.cache_dir, ccfg.device = CPU_CACHE, "cpu"
        c = PrePostPipelineEncoder(ccfg)
        t0 = time.perf_counter()
        cd = c.embed(dev.corpus_texts, False)
        t1 = time.perf_counter()
        cq = c.embed(qtexts, True)
        t2 = time.perf_counter()
        vecs["cpu"][name] = (cq, cd)
        timing[name] = {"corpus_s": round(t1 - t0, 1), "queries_s": round(t2 - t1, 1)}
        log.info("%s CPU fp32 encode: %s", name, timing[name])

    def pipeline(side: str, weights: dict[str, float]):
        q = l2(np.concatenate([np.sqrt(w) * vecs[side][m][0] for m, w in weights.items()], axis=1))
        d = l2(np.concatenate([np.sqrt(w) * vecs[side][m][1] for m, w in weights.items()], axis=1))
        return top_k(q, d, 50)

    out = {"n_queries": 200, "cpu_encode_timing": timing, "pipelines": {}}
    for pname, weights in (("qwen3_only", {"qwen3": 1.0}), ("fusion_w0.3", {"qwen3": 0.3, "gemma": 0.7})):
        (gp, gs), (cp, cs) = pipeline("gpu", weights), pipeline("cpu", weights)
        mg = per_query(run_from_topk(qids, dev.corpus_ids, gp, gs), qrels, (10, 50))
        mc = per_query(run_from_topk(qids, dev.corpus_ids, cp, cs), qrels, (10, 50))
        same10 = float(np.mean([np.array_equal(gp[i, :10], cp[i, :10]) for i in range(len(qids))]))
        set10 = float(np.mean([set(gp[i, :10]) == set(cp[i, :10]) for i in range(len(qids))]))
        top1 = float(np.mean(gp[:, 0] == cp[:, 0]))
        out["pipelines"][pname] = {
            "top10_identical_order_share": same10, "top10_same_set_share": set10, "top1_same_share": top1,
            "ndcg_at_10": {"gpu": float(mg["ndcg_at_10"].mean()), "cpu": float(mc["ndcg_at_10"].mean()),
                           "diff": float(mc["ndcg_at_10"].mean() - mg["ndcg_at_10"].mean())},
            "recall_at_50": {"gpu": float(mg["recall_at_50"].mean()), "cpu": float(mc["recall_at_50"].mean()),
                             "diff": float(mc["recall_at_50"].mean() - mg["recall_at_50"].mean())},
            "max_abs_score_diff_top50": float(np.abs(gs - cs).max()),
        }
    (REPO_ROOT / "results" / "cpu_repro.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
