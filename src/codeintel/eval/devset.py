"""Validation protocol (plan Section 10.1, R2/R3).

Train qrels (5,000 pairs, local `data/apps/data/train-*.parquet`) are split into train 4,000 /
validation 1,000 query IDs with seed 13. The validation corpus is the full 8,765-document
corpus. Texts come from the `corpus` and `queries` configs at the revision mteb pins, prepared
exactly as mteb prepares them. Test qrels are never opened here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from codeintel.common.config import REPO_ROOT
from codeintel.common.hashing import sha1_text

SEED = 13
N_VAL = 1000
TRAIN_QRELS = REPO_ROOT / "data" / "apps" / "data" / "train-00000-of-00001.parquet"
SPLIT_FILE = REPO_ROOT / "results" / "devset_split.json"


@dataclass
class DevSet:
    query_ids: list[str]          # validation query IDs (dictionary keys only, R1)
    query_texts: list[str]        # aligned with query_ids
    corpus_ids: list[str]         # full corpus
    corpus_texts: list[str]       # aligned with corpus_ids, mteb doc-text rule applied
    qrels: dict[str, dict[str, int]]
    gold_texts: list[str]         # gold document text per validation query (Stage 2 ceiling)


@lru_cache(maxsize=1)
def load_apps_texts() -> tuple[list[str], list[str], dict[str, str]]:
    import mteb
    from datasets import load_dataset
    from mteb._create_dataloaders import _corpus_to_dict

    meta = mteb.get_task("AppsRetrieval").metadata.dataset
    corpus = next(iter(load_dataset(meta["path"], "corpus", revision=meta["revision"]).values()))
    queries = next(iter(load_dataset(meta["path"], "queries", revision=meta["revision"]).values()))
    corpus_ids = [str(x) for x in corpus["_id"]]
    corpus_texts = [
        _corpus_to_dict({"id": "", "title": t or "", "text": x})["text"] for t, x in zip(corpus["title"], corpus["text"])
    ]
    query_text = {str(i): t for i, t in zip(queries["_id"], queries["text"])}
    return corpus_ids, corpus_texts, query_text


def load_train_qrels() -> dict[str, dict[str, int]]:
    df = pq.read_table(TRAIN_QRELS).to_pandas()
    qrels: dict[str, dict[str, int]] = {}
    for q, d, s in zip(df["query-id"].astype(str), df["corpus-id"].astype(str), df["score"].astype(int)):
        qrels.setdefault(q, {})[d] = s
    return qrels


def split_qrels(qrels: dict[str, dict[str, int]], seed: int = SEED, n_val: int = N_VAL):
    qids = sorted(qrels)
    perm = np.random.default_rng(seed).permutation(len(qids))
    val = sorted(qids[i] for i in perm[:n_val])
    train = sorted(qids[i] for i in perm[n_val:])
    return {q: qrels[q] for q in train}, {q: qrels[q] for q in val}


def load_devset(split: str = "val") -> DevSet:
    train, val = split_qrels(load_train_qrels())
    part = val if split == "val" else train
    corpus_ids, corpus_texts, query_text = load_apps_texts()
    pos = {d: i for i, d in enumerate(corpus_ids)}
    qids = sorted(part)
    gold = [corpus_texts[pos[next(d for d, s in part[q].items() if s > 0)]] for q in qids]
    return DevSet(qids, [query_text[q] for q in qids], corpus_ids, corpus_texts, part, gold)


def write_split_record() -> dict:
    train, val = split_qrels(load_train_qrels())
    rec = {
        "seed": SEED,
        "n_train": len(train),
        "n_val": len(val),
        "val_query_ids_sha1": sha1_text(",".join(sorted(val))),
        "train_query_ids_sha1": sha1_text(",".join(sorted(train))),
        "val_query_ids": sorted(val),
    }
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


if __name__ == "__main__":
    r = write_split_record()
    print({k: v for k, v in r.items() if k != "val_query_ids"})
