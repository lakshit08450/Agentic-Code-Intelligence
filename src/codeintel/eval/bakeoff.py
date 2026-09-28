"""Stage 1 bake-off on the validation split (plan 1.3).

    # metrics from the embedding cache (GPU-filled), plus top-100 saved for Stage 2
    python -m codeintel.eval.bakeoff metrics --configs configs/bakeoff/*.yaml
    # CPU throughput on 100 docs / 100 queries (run with the CPU otherwise idle)
    python -m codeintel.eval.bakeoff throughput --configs configs/bakeoff/*.yaml

Writes results/bakeoff.csv (one row per model, both parts merged) and
cache/stage1/<name>_val_top100.npz (positions into the corpus + cosine scores).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset
from codeintel.eval.metrics import per_query, run_from_topk, summarize
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.stage1.search import search

log = logging.getLogger("bakeoff")
CSV = REPO_ROOT / "results" / "bakeoff.csv"
PERQ = REPO_ROOT / "results" / "bakeoff_perquery.json"
TOPK_DIR = REPO_ROOT / "cache" / "stage1"
K_VALUES = (1, 10, 20, 50, 100)


def _read_rows() -> dict[str, dict]:
    if not CSV.exists():
        return {}
    with open(CSV, encoding="utf-8", newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def _write_rows(rows: dict[str, dict]) -> None:
    cols: list[str] = []
    for r in rows.values():
        cols += [c for c in r if c not in cols]
    CSV.parent.mkdir(exist_ok=True)
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows.values())


def topk_path(name: str, split: str = "val", k: int = 100) -> Path:
    return TOPK_DIR / f"{name}_{split}_top{k}.npz"


def mteb_crosscheck(run: dict, qrels: dict, ours: dict[str, float]) -> dict[str, float]:
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    r = calculate_retrieval_scores(run, qrels, [10, 100])
    theirs = {"ndcg_at_10": r.ndcg["NDCG@10"], "mrr_at_10": r.mrr["MRR@10"], "recall_at_100": r.recall["Recall@100"]}
    return {m: abs(ours[m] - v) for m, v in theirs.items()}


def metrics(configs: list[str], crosscheck: bool) -> None:
    dev = load_devset("val")
    rows = _read_rows()
    perq = json.loads(PERQ.read_text(encoding="utf-8")) if PERQ.exists() else {}
    for i, cfg_path in enumerate(configs):
        cfg = load_cfg(cfg_path)
        cfg.on_cache_miss = "error"
        enc = PrePostPipelineEncoder(cfg)
        pos, scores = search(enc, dev.query_texts, dev.corpus_texts, k=max(K_VALUES))
        TOPK_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(topk_path(cfg.name), pos=pos, scores=scores.astype(np.float32))
        run = run_from_topk(dev.query_ids, dev.corpus_ids, pos, scores)
        pq = per_query(run, dev.qrels, K_VALUES)
        s = summarize(pq)
        row = rows.get(cfg.name, {}) | {"name": cfg.name, "model": cfg.models[0].id, "config": cfg_path}
        row |= {m: f"{v:.5f}" for m, v in s.items() if m.split("_at_")[0] in {"ndcg", "mrr", "recall"}}
        rows[cfg.name] = row
        perq[cfg.name] = {m: v.round(6).tolist() for m, v in pq.items()}
        if crosscheck and i == 0:
            diffs = mteb_crosscheck(run, dev.qrels, s)
            log.info("mteb cross-check (abs diff) for %s: %s", cfg.name, diffs)
            row["mteb_max_abs_diff"] = f"{max(diffs.values()):.2e}"
        log.info("%s: %s | cache %s", cfg.name, {m: round(v, 4) for m, v in s.items()}, enc.stats)
    _write_rows(rows)
    PERQ.write_text(json.dumps(perq), encoding="utf-8")


def throughput(configs: list[str], n: int, threads: int | None) -> None:
    import torch

    if threads:
        torch.set_num_threads(threads)
    dev = load_devset("val")
    rng = np.random.default_rng(13)
    docs = [dev.corpus_texts[i] for i in rng.choice(len(dev.corpus_texts), n, replace=False)]
    queries = [dev.query_texts[i] for i in rng.choice(len(dev.query_texts), n, replace=False)]
    rows = _read_rows()
    for cfg_path in configs:
        cfg = load_cfg(cfg_path)  # device: cpu from config
        enc = PrePostPipelineEncoder(cfg)
        be = enc._backend(0)
        be.encode(enc.model_texts(docs[:4], False, 0))  # warm-up
        res = {}
        for label, texts, is_q in (("docs", docs, False), ("queries", queries, True)):
            t0 = time.perf_counter()
            be.encode(enc.model_texts(texts, is_q, 0))
            res[f"cpu_{label}_per_s"] = n / (time.perf_counter() - t0)
        log.info("%s CPU (%d threads): %s", cfg.name, torch.get_num_threads(), {k: round(v, 2) for k, v in res.items()})
        row = rows.get(cfg.name, {"name": cfg.name, "model": cfg.models[0].id, "config": cfg_path})
        row |= {k: f"{v:.2f}" for k, v in res.items()} | {"cpu_threads": str(torch.get_num_threads())}
        rows[cfg.name] = row
        del enc, be
    _write_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["metrics", "throughput"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "metrics":
        metrics(a.configs, crosscheck=not a.no_crosscheck)
    else:
        throughput(a.configs, a.n, a.threads)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
