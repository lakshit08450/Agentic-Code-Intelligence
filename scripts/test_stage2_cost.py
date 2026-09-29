"""Label-free cost estimate for Stage 2 on test: Qwen3 top-k candidates for test query TEXTS (queries not
in train qrels), from the embedding cache; counts pairs needing execution. Never runs code, no qrels."""
import json
import numpy as np
from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_apps_texts, load_train_qrels
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.stage1.search import search
from codeintel.stage2.sampleio import parse
from codeintel.stage2.verifier import program_mode

cfg = load_cfg("configs/bakeoff/qwen3_0p6b.yaml"); cfg.on_cache_miss = "error"
enc = PrePostPipelineEncoder(cfg)
corpus_ids, corpus_texts, qt = load_apps_texts()
train = set(load_train_qrels())
texts = [t for q, t in qt.items() if q not in train]
pos, _ = search(enc, texts, corpus_texts, k=50)
modes = [program_mode(t) for t in corpus_texts]
samples = [parse(t) for t in texts]
out = {"n_test_queries": len(texts)}
for k in (10, 20, 50):
    pairs = runs = 0
    for qi, s in enumerate(samples):
        if s.kind != "stdin":
            continue
        for p in pos[qi, :k]:
            if modes[p] == "stdin":
                pairs += 1
    out[f"k{k}"] = {"pairs_to_execute": pairs, "share_of_all_pairs": pairs / (len(texts) * k),
                    "hours_at_4_runs_per_s": pairs * 1.01 / 4 / 3600}
(REPO_ROOT / "results" / "test_stage2_cost.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
