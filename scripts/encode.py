"""Encode the AppsRetrieval corpus and all queries for a Stage 1 config into cache/embeddings/.

Loads only the `corpus` and `queries` configs of CoIR-Retrieval/apps at the revision mteb
pins, never qrels (R3). Texts are prepared exactly as mteb prepares them, so the final
mteb run is all cache hits. Resumable: the cache is written in chunks and skipped on rerun.

    .venv/Scripts/python scripts/encode.py --config configs/bakeoff/gte_modernbert.yaml --device cuda
"""

from __future__ import annotations

import argparse
import logging
import time

import mteb
from datasets import load_dataset
from mteb._create_dataloaders import _corpus_to_dict  # same doc-text rule mteb applies

from codeintel.common.config import load_cfg
from codeintel.stage1.encoder import PrePostPipelineEncoder

log = logging.getLogger("encode")


def load_texts(target: str) -> list[str]:
    ds_meta = mteb.get_task("AppsRetrieval").metadata.dataset
    ds = load_dataset(ds_meta["path"], target, revision=ds_meta["revision"])
    split = next(iter(ds.values()))
    if target == "corpus":
        # only title and text are passed; the helper requires an "id" key, left blank (R1)
        return [_corpus_to_dict({"id": "", "title": r.get("title") or "", "text": r["text"]})["text"] for r in split]
    return list(split["text"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cuda|cpu (overrides config)")
    ap.add_argument("--targets", default="corpus,queries")
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N texts per target (smoke test)")
    ap.add_argument("--chunk", type=int, default=512, help="texts per cache chunk file")
    args = ap.parse_args()

    cfg = load_cfg(args.config, device=args.device)
    enc = PrePostPipelineEncoder(cfg)
    for target in args.targets.split(","):
        texts = load_texts(target)
        if args.limit:
            texts = texts[: args.limit]
        is_query = target == "queries"
        for i, m in enumerate(cfg.models):
            t0 = time.perf_counter()
            n = enc.fill_cache(texts, is_query, i, chunk=args.chunk, progress=True)
            dt = time.perf_counter() - t0
            rate = f"{n / dt:.1f} texts/s" if n else "all cached"
            log.info(
                "%s | %s | %d texts, %d newly embedded in %.1fs (%s) on %s | cache now %d",
                target, m.id, len(texts), n, dt, rate, cfg.device, len(enc.stores[i]),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
