"""Path B metrics (plan Section 12.5), CPU only, scaled down for time (P, 30 Sep): 5 versions of a
validation mini-corpus = gold solutions of 200 validation queries (seed 13, same as cpu_repro) + 800
distractors from the train split, as JSONL units with stable ids (ground-truth unit identity).

Edits per version (seeded): 10% whitespace/comment edits, 5% variable renames, 5% small code edits
(numeric constant changes), +2% new units, -2% deleted units (gold units are never deleted).

Reports, per pipeline (fusion w=0.3 = submission, and Qwen3-only):
- single-version (v4) NDCG@10 / MRR@10;
- all versions v0..v4 without collapse (every revision ranked; duplicates occupy slots) vs with collapse
  (one result per unit) at unit level;
- incremental vs full rebuild time and embeddings reused (%), from an empty embedding cache;
- a real git demo: this repo's src/ over its last 4 commits (incremental times, reuse).
Query vectors come from the CPU fp32 cache written by scripts/cpu_repro.py.
Writes results/versioning_<pipeline>.json.
"""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from codeintel.common import proc
from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_devset, load_train_qrels, split_qrels
from codeintel.eval.metrics import per_query
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.versioned.ingest import Version, git_first_parent, git_versions
from codeintel.versioned.store import Store

PIPELINES = {"fusion_w0.3": "configs/fusion/qwen3_gemma_w0.3.yaml", "qwen3_only": "configs/stage1_final.yaml"}
CPU_CACHE = "cache/embeddings_cpu_fp32"
N_Q, N_DISTRACT = 200, 800


def rename_var(text: str, rng: random.Random) -> str:
    names = sorted(set(re.findall(r"^\s*([a-z_][a-z0-9_]{0,15})\s*=", text, re.M)) - {"_"})
    if not names:
        return text + "\n"
    old = rng.choice(names)
    return re.sub(rf"\b{re.escape(old)}\b", old + "_v", text)


def small_edit(text: str, rng: random.Random) -> str:
    nums = list(re.finditer(r"\b\d+\b", text))
    if not nums:
        return text + "\npass\n"
    m = rng.choice(nums)
    return text[: m.start()] + str(int(m.group()) + 1) + text[m.end():]


def ws_comment(text: str, rng: random.Random) -> str:
    return ("# refactored\n" + text) if rng.random() < 0.5 else text.replace("\n", "\n\n", 1)


def build_versions(dev) -> tuple[list[Version], list[str], dict[str, str], list[str]]:
    rng = np.random.default_rng(13)
    sel = np.sort(rng.choice(len(dev.query_ids), N_Q, replace=False))
    qids = [dev.query_ids[i] for i in sel]
    idx = {d: i for i, d in enumerate(dev.corpus_ids)}
    gold = {q: next(iter(dev.qrels[q])) for q in qids}
    train, _ = split_qrels(load_train_qrels())
    train_docs = sorted({d for rel in train.values() for d in rel} - set(gold.values()))
    r = random.Random(13)
    distract = r.sample(train_docs, N_DISTRACT + 200)
    pool, spare = distract[:N_DISTRACT], distract[N_DISTRACT:]
    units = {f"u_{d}": dev.corpus_texts[idx[d]] for d in list(gold.values()) + pool}
    gold_unit = {q: f"u_{d}" for q, d in gold.items()}
    protected = set(gold_unit.values())
    versions = [Version("v0", "sim", {k: (k + ".py", t) for k, t in units.items()})]
    for v in range(1, 5):
        rv = random.Random(100 + v)
        keys = sorted(units)
        for k in rv.sample(keys, len(keys) // 10):
            units[k] = ws_comment(units[k], rv)
        for k in rv.sample(keys, len(keys) // 20):
            units[k] = rename_var(units[k], rv)
        for k in rv.sample(keys, len(keys) // 20):
            units[k] = small_edit(units[k], rv)
        for k in rv.sample([k for k in keys if k not in protected], len(keys) // 50):
            del units[k]
        for _ in range(len(keys) // 50):
            if spare:
                d = spare.pop()
                units[f"u_{d}"] = dev.corpus_texts[idx[d]]
        versions.append(Version(f"v{v}", "sim", {k: (k + ".py", t) for k, t in units.items()}))
    return versions, qids, gold_unit, [dev.query_texts[i] for i in sel]


_BACKENDS: dict = {}


def fresh_encoder(cfg_path: str, tmp: Path, tag: str) -> PrePostPipelineEncoder:
    """Empty embedding cache per store (honest timings), but ONE loaded copy of each model (RAM)."""
    from codeintel.stage1.encoder import SentenceTransformerBackend

    cfg = copy.deepcopy(load_cfg(cfg_path))
    cfg.cache_dir, cfg.device = str(tmp / f"emb_{tag}"), "cpu"
    for m in cfg.models:
        if m.id not in _BACKENDS:
            _BACKENDS[m.id] = SentenceTransformerBackend(m, "cpu", cfg.batch_size, fp16_on_cuda=False, attn_budget=cfg.attn_budget)
    return PrePostPipelineEncoder(cfg, backends=[_BACKENDS[m.id] for m in cfg.models])


def unit_metrics(store: Store, qv: np.ndarray, qids, gold_unit, rows, collapse: bool) -> dict:
    """rows from store.alive_at / alive_in_range; unit key via units table."""
    ukey = dict(store.db.execute("SELECT unit_id, key FROM units"))
    vec = store.vectors[np.array([r[5] for r in rows])]
    scores = qv @ vec.T
    run = {}
    for i, q in enumerate(qids):
        order = np.argsort(-scores[i], kind="stable")
        seen: set[str] = set()
        ranked: list[str] = []
        for j in order:
            u = ukey[rows[j][1]]
            if u in seen:
                if collapse:
                    continue
                ranked.append(f"{u}#dup{len(ranked)}")  # a repeated unit occupies a slot, never relevant
            else:
                seen.add(u)
                ranked.append(u)
            if len(ranked) == 10:
                break
        run[q] = {u: float(10 - n) for n, u in enumerate(ranked)}
    pq = per_query(run, {q: {gold_unit[q]: 1} for q in qids}, (10,))
    return {"ndcg_at_10": float(pq["ndcg_at_10"].mean()), "mrr_at_10": float(pq["mrr_at_10"].mean())}


def run_pipeline(name: str, cfg_path: str, versions, qids, gold_unit, qtexts) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"sim_{name}_", dir=REPO_ROOT / "cache"))
    try:
        qcfg = copy.deepcopy(load_cfg(cfg_path))
        qcfg.cache_dir, qcfg.device = CPU_CACHE, "cpu"
        qv = PrePostPipelineEncoder(qcfg).embed(qtexts, is_query=True)  # all cached (cpu_repro): no model load
        inc = Store(tmp / "inc", fresh_encoder(cfg_path, tmp, "inc"))
        steps = [inc.add_version(v) for v in versions]
        t0 = time.perf_counter()
        full = Store(tmp / "full", fresh_encoder(cfg_path, tmp, "full"))
        full_stats = full.add_version(versions[-1])
        full_s = time.perf_counter() - t0
        last = inc.ordinal("v4")
        res = {
            "pipeline": name, "config": cfg_path, "units_v0": len(versions[0].items), "units_v4": len(versions[-1].items),
            "single_version_v4": unit_metrics(inc, qv, qids, gold_unit, inc.alive_at(last), collapse=True),
            "all_versions_no_collapse": unit_metrics(inc, qv, qids, gold_unit, inc.alive_in_range(0, last), collapse=False),
            "all_versions_collapse": unit_metrics(inc, qv, qids, gold_unit, inc.alive_in_range(0, last), collapse=True),
            "incremental_steps": [{k: s[k] for k in ("label", "units", "unchanged", "modified", "added", "removed", "embedded", "reused", "reused_share", "ms")} for s in steps],
            "incremental_update_ms_v1_to_v4": [s["ms"]["total"] for s in steps[1:]],
            "full_rebuild_v4_s": round(full_s, 2), "full_rebuild_embedded": full_stats["embedded"],
        }
        # real git demo: this repo's src/ over the last 4 first-parent commits
        revs = git_first_parent(REPO_ROOT, "HEAD")[-4:]
        g = Store(tmp / "git", fresh_encoder(cfg_path, tmp, "git"))
        res["git_demo"] = [
            {k: s[k] for k in ("label", "units", "unchanged", "modified", "added", "removed", "embedded", "reused", "ms")}
            for s in (g.add_version(v) for v in git_versions(REPO_ROOT, revs))
        ]
        return res
    finally:
        for st in ("inc", "full", "g"):
            if st in locals():
                locals()[st].db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    dev = load_devset("val")
    versions, qids, gold_unit, qtexts = build_versions(dev)
    names = sys.argv[1:] or list(PIPELINES)
    for name in names:
        res = run_pipeline(name, PIPELINES[name], versions, qids, gold_unit, qtexts)
        (REPO_ROOT / "results" / f"versioning_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in res if k not in ("incremental_steps", "git_demo")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
