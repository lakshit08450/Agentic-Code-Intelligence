"""CPU equivalence check (plan Section 6 / Phase 1.4).

Encode 200 documents and 50 validation queries on CPU (fp32) and compare with the GPU-filled
cache: max |cos_cpu(q, d) - cos_gpu(q, d)| must be < 1e-3 and the top-10 order of the 200 docs
must match for all 50 queries. Writes results/cpu_check_<name>.json.

    .venv/Scripts/python scripts/cpu_check.py --config configs/bakeoff/gte_modernbert.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset
from codeintel.stage1.encoder import PrePostPipelineEncoder, SentenceTransformerBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--queries", type=int, default=50)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    cfg.on_cache_miss = "error"
    enc = PrePostPipelineEncoder(cfg)
    dev = load_devset("val")
    rng = np.random.default_rng(13)
    docs = [dev.corpus_texts[i] for i in rng.choice(len(dev.corpus_texts), a.docs, replace=False)]
    queries = [dev.query_texts[i] for i in rng.choice(len(dev.query_texts), a.queries, replace=False)]

    gq, gd = enc.embed(queries, True), enc.embed(docs, False)  # GPU vectors from the cache
    cpu = SentenceTransformerBackend(cfg.models[0], "cpu", cfg.batch_size, fp16_on_cuda=False, attn_budget=cfg.attn_budget)
    t0 = time.perf_counter()
    cq = cpu.encode(enc.model_texts(queries, True, 0))
    cd = cpu.encode(enc.model_texts(docs, False, 0))
    dt = time.perf_counter() - t0

    s_gpu, s_cpu = gq @ gd.T, cq @ cd.T
    top_gpu = np.argsort(-s_gpu, axis=1, kind="stable")[:, :10]
    top_cpu = np.argsort(-s_cpu, axis=1, kind="stable")[:, :10]
    mismatched = [i for i in range(len(queries)) if not np.array_equal(top_gpu[i], top_cpu[i])]
    res = {
        "config": a.config, "model": cfg.models[0].id, "n_docs": len(docs), "n_queries": len(queries),
        "max_abs_cos_diff": float(np.abs(s_gpu - s_cpu).max()),
        "mean_abs_cos_diff": float(np.abs(s_gpu - s_cpu).mean()),
        "min_self_cos_query": float((gq * cq).sum(1).min()), "min_self_cos_doc": float((gd * cd).sum(1).min()),
        "top10_mismatched_queries": len(mismatched),
        "top10_set_mismatched_queries": sum(set(top_gpu[i]) != set(top_cpu[i]) for i in range(len(queries))),
        "cpu_encode_seconds": round(dt, 1),
    }
    res["pass"] = res["max_abs_cos_diff"] < 1e-3 and res["top10_mismatched_queries"] == 0
    out = REPO_ROOT / "results" / f"cpu_check_{cfg.name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
