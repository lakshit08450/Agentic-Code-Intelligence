"""Versioned retrieval CLI (Path B: P1 and Bonus).

    # build / update an index (incremental: only new versions are added)
    python -m codeintel.versioned.cli index --store stores/demo --git C:/path/repo --revs v1 v2 v3
    python -m codeintel.versioned.cli index --store stores/demo --git C:/path/repo --range v1..main
    python -m codeintel.versioned.cli index --store stores/snap --snapshots snaps/v1 snaps/v2
    python -m codeintel.versioned.cli index --store stores/js --jsonl v1.jsonl v2.jsonl
    # search one version (P1) or all versions in a range (Bonus)
    python -m codeintel.versioned.cli search --store stores/demo --rev v2 "reverse a linked list" [--verify]
    python -m codeintel.versioned.cli search --store stores/demo --range v1..v3 "query text"
    python -m codeintel.versioned.cli versions --store stores/demo

Defaults to CPU and configs/stage1_final.yaml (R5). Output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.versioned.ingest import git_first_parent, git_versions, jsonl_versions, snapshot_versions
from codeintel.versioned.search import search_range, search_rev
from codeintel.versioned.store import Store


def open_store(path: str, config: str, device: str | None) -> Store:
    return Store(path, PrePostPipelineEncoder(config, device=device))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="codeintel.versioned.cli")
    ap.add_argument("--config", default="configs/stage1_final.yaml")
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("--store", required=True)
    src = ix.add_mutually_exclusive_group(required=True)
    src.add_argument("--git")
    src.add_argument("--snapshots", nargs="+")
    src.add_argument("--jsonl", nargs="+")
    ix.add_argument("--revs", nargs="+", help="git revisions, oldest first")
    ix.add_argument("--range", dest="rev_range", help="git A..B along first-parent (A itself included)")
    se = sub.add_parser("search")
    se.add_argument("--store", required=True)
    g = se.add_mutually_exclusive_group(required=True)
    g.add_argument("--rev")
    g.add_argument("--range", dest="rng")
    se.add_argument("query")
    se.add_argument("--k", type=int, default=10)
    se.add_argument("--verify", action="store_true", help="Stage 2 re-rank when the query has sample I/O")
    ve = sub.add_parser("versions")
    ve.add_argument("--store", required=True)
    a = ap.parse_args(argv)

    store = open_store(a.store, a.config, a.device)
    if a.cmd == "index":
        if a.git:
            revs = a.revs or []
            if a.rev_range:
                start, _, end = a.rev_range.partition("..")
                revs = [start] + git_first_parent(a.git, a.rev_range)
            versions = git_versions(a.git, revs)
        elif a.snapshots:
            versions = snapshot_versions(a.snapshots)
        else:
            versions = jsonl_versions(a.jsonl)
        have = {label for _, label, _ in store.versions()}
        out = [store.add_version(v) for v in versions if v.label not in have]
        print(json.dumps(out, indent=1))
    elif a.cmd == "search":
        if a.rev is not None:
            res = search_rev(store, a.query, a.rev, a.k, a.verify)
        else:
            lo, _, hi = a.rng.partition("..")
            res = search_range(store, a.query, lo, hi, a.k, a.verify)
        print(json.dumps(res, indent=1))
    else:
        print(json.dumps([{"ordinal": o, "label": l, "source": s} for o, l, s in store.versions()], indent=1))


if __name__ == "__main__":
    sys.exit(main())
