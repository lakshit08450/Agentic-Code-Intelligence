"""Fill the CPU fp32 embedding cache (cache/embeddings_cpu_fp32) for one config: full corpus + the 200
validation queries used by cpu_repro.py (seed 13). Lets two models encode in parallel processes.

    .venv/Scripts/python scripts/encode_cpu_fp32.py configs/bakeoff/embeddinggemma_300m.yaml [--threads 4]
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from codeintel.common.config import load_cfg
from codeintel.eval.devset import load_devset
from codeintel.stage1.encoder import PrePostPipelineEncoder

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--threads", type=int, default=None)
a = ap.parse_args()
if a.threads:
    import torch

    torch.set_num_threads(a.threads)
dev = load_devset("val")
sel = np.sort(np.random.default_rng(13).choice(len(dev.query_ids), 200, replace=False))
cfg = copy.deepcopy(load_cfg(a.config))
cfg.cache_dir, cfg.device = "cache/embeddings_cpu_fp32", "cpu"
enc = PrePostPipelineEncoder(cfg)
t0 = time.perf_counter()
enc.fill_cache([dev.query_texts[i] for i in sel], True, 0, progress=True)
enc.fill_cache(dev.corpus_texts, False, 0, progress=True)
print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)
