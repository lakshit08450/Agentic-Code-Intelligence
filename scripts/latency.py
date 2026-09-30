"""CPU query latency (plan Section 13), for the submission pipeline (fusion w=0.3) and Qwen3-only.

Stage 1 per query: CPU fp32 query encode with no cache + cosine search over the full corpus (corpus
vectors from the CPU fp32 cache written by scripts/cpu_repro.py). Stage 2 per query: fresh sandbox
executions of the top-k candidates (no exec cache), 12 workers, frozen k=50 weights. N validation
stdin queries (seed 13). Run with the CPU otherwise idle. Writes results/latency_<pipeline>.json.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.stage1.search import top_k
from codeintel.stage2.reranker import load_stage2_cfg, stage2_score
from codeintel.stage2.verifier import samples_for, verify

PIPELINES = {"fusion_w0.3": "configs/fusion/qwen3_gemma_w0.3.yaml", "qwen3_only": "configs/stage1_final.yaml"}
CPU_CACHE = "cache/embeddings_cpu_fp32"


def main(names: list[str], n: int = 20, k: int = 50) -> None:
    dev = load_devset("val")
    stdin_idx = [i for i, t in enumerate(dev.query_texts) if samples_for(t).kind == "stdin"]
    sel = sorted(np.random.default_rng(13).choice(stdin_idx, n, replace=False).tolist())
    s2 = load_stage2_cfg("configs/stage2_final_k50.yaml")
    for name in names:
        cfg_path = PIPELINES[name]
        dcfg = copy.deepcopy(load_cfg(cfg_path))
        dcfg.cache_dir, dcfg.device, dcfg.on_cache_miss = CPU_CACHE, "cpu", "error"
        docs = PrePostPipelineEncoder(dcfg).embed(dev.corpus_texts, is_query=False)
        tmp = Path(tempfile.mkdtemp(prefix="lat_", dir=REPO_ROOT / "cache"))
        qcfg = copy.deepcopy(load_cfg(cfg_path))
        qcfg.cache_dir, qcfg.device = str(tmp), "cpu"
        enc = PrePostPipelineEncoder(qcfg)
        enc.embed(["warm-up query"], is_query=True)  # model load excluded from latency
        s1, s2t, res = [], [], []
        for i in sel:
            q = dev.query_texts[i]
            t0 = time.perf_counter()
            qv = enc.embed([q], is_query=True)
            pos, cos = top_k(qv, docs, k)
            t1 = time.perf_counter()
            cands = [dev.corpus_texts[p] for p in pos[0]]
            with ThreadPoolExecutor(max_workers=s2.workers) as ex:
                verdicts = list(ex.map(lambda d: verify(q, d, None), cands))
            scores = [stage2_score(float(c), v.outcome, q, s2) for c, v in zip(cos[0], verdicts)]
            t2 = time.perf_counter()
            s1.append(t1 - t0)
            s2t.append(t2 - t1)
            res.append(int(np.argmax(scores)))
        out = {
            "pipeline": name, "config": cfg_path, "n_queries": n, "k": k, "device": "cpu (fp32)",
            "stage1_ms": {"p50": 1000 * float(np.percentile(s1, 50)), "p95": 1000 * float(np.percentile(s1, 95))},
            "stage2_ms": {"p50": 1000 * float(np.percentile(s2t, 50)), "p95": 1000 * float(np.percentile(s2t, 95))},
            "total_ms": {"p50": 1000 * float(np.percentile(np.add(s1, s2t), 50)), "p95": 1000 * float(np.percentile(np.add(s1, s2t), 95))},
            "stage2_executions_per_s": n * k / sum(s2t),
        }
        (REPO_ROOT / "results" / f"latency_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(PIPELINES))
