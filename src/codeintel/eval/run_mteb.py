"""Run MTEB AppsRetrieval (test split) for a frozen Stage 1 config and write the MTEB JSON.

This is the only place the test qrels are loaded, inside mteb.evaluate (R3). Test runs are
allowed only for Phase 0 G1 / Output A v0 and Phase 4, or with P's approval (R2).

    .venv/Scripts/python -m codeintel.eval.run_mteb --config configs/bakeoff/gte_modernbert.yaml \
        --out results/appsretrieval_results_A.json --prediction-folder results/predictions/stage1
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import mteb

from codeintel.common import proc
from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.stage1.encoder import PrePostPipelineEncoder

log = logging.getLogger("run_mteb")


def _hardware() -> dict:
    import psutil
    import torch

    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "mteb": mteb.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="device for cache misses (default: config, i.e. cpu)")
    ap.add_argument("--out", required=True, help="path of the MTEB result JSON")
    ap.add_argument("--prediction-folder", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--require-cache", action="store_true", help="fail on any cache miss")
    args = ap.parse_args()

    cfg = load_cfg(args.config, device=args.device)
    if args.require_cache:
        cfg.on_cache_miss = "error"
    enc = PrePostPipelineEncoder(cfg)
    task = mteb.get_task("AppsRetrieval")

    t0 = time.perf_counter()
    result = mteb.evaluate(
        enc,
        task,
        cache=None,  # never read or write ~/.cache/mteb
        overwrite_strategy="always",
        encode_kwargs={"batch_size": args.batch_size},
        prediction_folder=args.prediction_folder,
        co2_tracker=False,
        num_proc=1,  # DataLoader num_workers=0: no worker processes (plan Section 8)
    )
    elapsed = time.perf_counter() - t0

    task_result = result.task_results[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    task_result.to_disk(out)

    scores = task_result.scores["test"][0]
    commit = proc.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    meta = {
        "config": cfg.source_path,
        "config_name": cfg.name,
        "git_commit": commit,
        "device_for_misses": cfg.device,
        "cache_stats": enc.stats,
        "elapsed_s": round(elapsed, 1),
        "hardware": _hardware(),
        "headline": {k: scores[k] for k in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_100") if k in scores},
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["headline"] | {"cache_stats": enc.stats, "elapsed_s": meta["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
