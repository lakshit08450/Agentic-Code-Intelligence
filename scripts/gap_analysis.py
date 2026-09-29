"""Gap diagnosis -> results/gap_analysis.json (P's request, Tue 29 Sep morning).

Test labels are never opened. Test-side numbers come only from (a) mteb's own saved result JSON
and (b) test query TEXTS, identified as the queries not in the train qrels (label-free).

    .venv/Scripts/python scripts/gap_analysis.py
"""

from __future__ import annotations

import collections
import json
import random
import re

import numpy as np

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.bakeoff import topk_path
from codeintel.eval.devset import load_apps_texts, load_devset, load_train_qrels
from codeintel.stage2.callspec import parse_calls
from codeintel.stage2.compare import outputs_match
from codeintel.stage2.sampleio import parse
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import program_mode, samples_for, verify

OUT = REPO_ROOT / "results" / "gap_analysis.json"
MODEL = "qwen3-embedding-0.6b"
CFG = "configs/bakeoff/qwen3_0p6b.yaml"


# ---------------- 1. Stage 1 loss buckets ----------------
def loss_buckets(dev) -> dict:
    t = json.loads((REPO_ROOT / "results" / "test_qwen3_zeroshot.json").read_text(encoding="utf-8"))["scores"]["test"][0]
    r = {k: t[f"recall_at_{k}"] for k in (1, 3, 5, 10, 20, 100, 1000)}
    test = {
        "source": "mteb recall_at_k from results/test_qwen3_zeroshot.json (one relevant doc per query, CoIR); no qrels opened",
        "recall_at_k": r,
        "rank_buckets": {
            "1": r[1], "2-10": r[10] - r[1], "11-20": r[20] - r[10], "21-100": r[100] - r[20],
            "101-1000": r[1000] - r[100], ">1000": 1 - r[1000],
        },
        "note": "mteb k_values are 1,3,5,10,20,100,1000, so the 11-50 / 51-100 split is not available on test",
    }
    pos = np.load(topk_path(MODEL))["pos"]
    gold_pos = {d: i for i, d in enumerate(dev.corpus_ids)}
    ranks = []
    for qi, q in enumerate(dev.query_ids):
        g = gold_pos[next(iter(dev.qrels[q]))]
        hit = np.where(pos[qi] == g)[0]
        ranks.append(int(hit[0]) + 1 if len(hit) else 10**6)
    ranks = np.array(ranks)
    val = {
        "recall_at_k": {k: float((ranks <= k).mean()) for k in (1, 10, 20, 50, 100)},
        "rank_buckets": {
            "1": float((ranks == 1).mean()), "2-10": float(((ranks >= 2) & (ranks <= 10)).mean()),
            "11-50": float(((ranks >= 11) & (ranks <= 50)).mean()), "51-100": float(((ranks >= 51) & (ranks <= 100)).mean()),
            ">100": float((ranks > 100).mean()),
        },
    }
    return {"test": test, "validation": val}


# ---------------- 2. token lengths ----------------
def token_lengths(val_texts: list[str], test_texts: list[str]) -> dict:
    from transformers import AutoTokenizer

    cfg = load_cfg(CFG)
    m = cfg.models[0]
    tok = AutoTokenizer.from_pretrained(m.id, revision=m.revision)
    out = {"model": m.id, "max_seq_length": m.max_seq_length, "query_prompt_tokens": len(tok(m.query_prompt)["input_ids"])}
    for name, texts in (("validation", val_texts), ("test", test_texts)):
        lens = np.array([len(ids) for ids in tok([m.query_prompt + t for t in texts])["input_ids"]])
        out[name] = {
            "n": len(lens), "mean": float(lens.mean()),
            "p50": float(np.percentile(lens, 50)), "p90": float(np.percentile(lens, 90)),
            "p99": float(np.percentile(lens, 99)), "max": int(lens.max()),
            "share_over_512": float((lens > 512).mean()), "share_over_1024": float((lens > 1024).mean()),
            "share_over_2048": float((lens > 2048).mean()),
            "share_truncated_at_max_seq_length": float((lens > m.max_seq_length).mean()),
        }
    return out


# ---------------- 3. format families ----------------
_F = [
    ("atcoder_stdin", re.compile(r"Input is given from Standard Input|^-----Sample Input(?: \d+)?-----\s*$", re.M)),
    ("codeforces_stdin", re.compile(r"^-----Examples?-----\s*$[\s\S]*^Input\s*$", re.M)),
    ("codechef_stdin", re.compile(r"-----Sample Input:?-----|-----Example Input-----|-----Input:-----|Subtask|Example case|-----EXAMPLE-----|-----Example-----", re.M)),
    ("kattis_stdin", re.compile(r"Sample Input 1")),
    ("hackerrank_stdin", re.compile(r"=====")),
    ("leetcode_call", re.compile(r"^\s*Input:\s*(?:[A-Za-z_]\w*\s*=|\[|\")|^Example 1:\s*$|class Solution", re.M)),
    ("codewars_call", re.compile(r"```|^#+ |\bkata\b|^\s*[A-Za-z_]\w*\([^)]*\)\s*(?:==|=>|->|=|#|//)", re.M | re.I)),
]


def family(text: str) -> str:
    for name, rx in _F:
        if rx.search(text):
            return name
    return "other_stdin" if re.search(r"-----Input-----|^Input\s*$", text, re.M) else "no_samples_or_unknown"


def format_mix(val_texts: list[str], test_texts: list[str]) -> dict:
    out = {}
    for name, texts in (("validation", val_texts), ("test", test_texts)):
        c = collections.Counter(family(t) for t in texts)
        out[name] = {k: {"n": v, "share": v / len(texts)} for k, v in sorted(c.items())}
        pk = collections.Counter(parse(t).kind for t in texts)
        out[name + "_stdin_parser_kinds"] = dict(pk)
    return out


# ---------------- 4. call-based parser misses ----------------
def call_misses(dev, n: int = 40) -> dict:
    miss = [q for q, g in zip(dev.query_texts, dev.gold_texts) if program_mode(g) == "call" and parse_calls(q).kind == "none"]
    random.Random(13).shuffle(miss)
    sample = miss[:n]

    def bucket(t: str) -> str:
        if re.search(r"TreeNode|ListNode|\broot\s*=|\bhead\s*=|linked list|binary tree", t, re.I):
            return "tree_linked_list"
        if re.search(r"in[- ]place|modify .* in|do not return|don't return", t, re.I):
            return "in_place"
        if re.search(r"```(?:javascript|js|csharp|c#|java|ruby|haskell|cpp|c\+\+)|\bnew \w+\(|;\s*//|\{[\d ,]+\}|\bvar |\bfunction\b", t, re.I):
            return "other_language_syntax"
        if re.search(r"\b[a-z_]\w*\s*=\s*[\[\d\"']", t) and re.search(r"result should be|should return|returns?", t, re.I):
            return "keyword_args"
        if re.search(r"(?i)should return|for example|e\.g\.|returns? `?\w|the result|output should", t):
            return "prose_examples"
        return "other"

    c = collections.Counter(bucket(t) for t in sample)
    return {"unparsed_call_based_validation": len(miss), "sampled": len(sample), "buckets": dict(c.most_common())}


# ---------------- 5. gold failures ----------------
_MULTI = re.compile(r"(?i)any of them|multiple (?:possible |valid |correct )?(?:answers|solutions)|any (?:valid|correct|such|possible)|if there are (?:several|many|multiple)|print any|output any")
_REMOVED = re.compile(r"fractions\.gcd|from fractions import gcd|time\.clock\(|string\.maketrans|collections\.(?:Mapping|Iterable|Callable)\b|\.has_key\(|asyncio\.async|base64\.encodestring")
_LINUX = re.compile(r"^\s*(?:import|from)\s+(?:resource|signal|fcntl|termios|posix|pwd|grp)\b", re.M)


def gold_failures(dev) -> dict:
    g = json.loads((REPO_ROOT / "results" / "gold_outcomes_v3_newparser.json").read_text(encoding="utf-8"))["per_query"]
    cache = ExecCache()
    causes: dict[str, list[str]] = collections.defaultdict(list)
    for qi, q in enumerate(dev.query_ids):
        v = g[q]
        if v["outcome"] == "PASS":
            continue
        stmt, prog = dev.query_texts[qi], dev.gold_texts[qi]
        if v["reason"] == "call_program":
            cause = "call_based_harness_not_built"
        elif v["outcome"] == "UNKNOWN":
            cause = f"unknown_{v['reason']}"
        else:
            errs = []
            for inp, _ in samples_for(stmt).pairs:
                r = cache.get(cache.key(prog, inp))
                if r is not None:
                    errs.append(r.stderr_tail)
            err = "\n".join(errs)
            if _REMOVED.search(prog) or re.search(r"has no attribute '(?:gcd|clock|maketrans)'", err):
                cause = "version_removed_api"
            elif _LINUX.search(prog):
                cause = "linux_only_import"
            elif "RecursionError" in err or "stack overflow" in err.lower() or (v["outcome"] == "ERROR" and "exit_-107" in v["reason"]):
                cause = "recursion_stack"
            elif v["reason"] == "guard":
                cause = "sandbox_guard_false_positive"
            elif v["outcome"] == "WRONG" and _MULTI.search(stmt):
                cause = "multiple_valid_answers"
            elif v["outcome"] == "WRONG" and any(
                re.search(r"(?i)explanation|note|\bthe\b.*\bis\b", o) for _, o in samples_for(stmt).pairs
            ):
                cause = "parser_contamination"
            else:
                cause = "genuine_mismatch_or_label_noise"
        causes[cause].append(q)
    static = {
        "gold_with_version_removed_api": sum(bool(_REMOVED.search(p)) for p in dev.gold_texts),
        "gold_with_linux_only_import": sum(bool(_LINUX.search(p)) for p in dev.gold_texts),
    }
    return {
        "counts": {k: len(v) for k, v in sorted(causes.items(), key=lambda kv: -len(kv[1]))},
        "example_query_ids": {k: v[0] for k, v in causes.items()},
        "static_scan_all_1000_gold": static,
    }


# ---------------- 6. distractor false passes ----------------
_TRIVIAL = re.compile(r"^(?:-?\d|YES|NO|Yes|No|yes|no|-1|0|1|True|False|IMPOSSIBLE|Impossible|possible|Possible)$")


def cached_verdict(stmt: str, prog: str, cache: ExecCache) -> str | None:
    """Stage 2 outcome from the exec cache only (never runs code); None if a needed run is missing."""
    s = samples_for(stmt)
    if s.kind != "stdin" or program_mode(prog) != "stdin":
        return "UNKNOWN"
    for inp, exp in s.pairs:
        r = cache.get(cache.key(prog, inp))
        if r is None:
            return None
        if r.status != "OK":
            return r.status
        if not outputs_match(r.stdout, exp):
            return "WRONG"
    return "PASS"


def false_passes(dev, k: int = 50) -> dict:
    pos = np.load(topk_path(MODEL))["pos"][:, :k]
    cache = ExecCache()
    idx = {d: i for i, d in enumerate(dev.corpus_ids)}
    stats = collections.Counter()
    for qi, q in enumerate(dev.query_ids):
        stmt = dev.query_texts[qi]
        s = samples_for(stmt)
        if s.kind != "stdin":
            continue
        gold_id = next(iter(dev.qrels[q]))
        gold_text = dev.corpus_texts[idx[gold_id]]
        verdicts = {int(p): cached_verdict(stmt, dev.corpus_texts[p], cache) for p in pos[qi]}
        if any(v is None for v in verdicts.values()):
            stats["queries_with_missing_runs"] += 1
            continue
        key = "trivial_output" if all(_TRIVIAL.match(o.strip()) for _, o in s.pairs) else "nontrivial_output"
        other = [p for p, v in verdicts.items() if v == "PASS" and dev.corpus_ids[p] != gold_id]
        other_nodup = [p for p in other if dev.corpus_texts[p] != gold_text]
        gold_pass = verdicts.get(idx[gold_id]) == "PASS"
        stats[f"{key}:queries"] += 1
        stats[f"{key}:gold_in_top{k}"] += idx[gold_id] in verdicts
        stats[f"{key}:gold_passes"] += gold_pass
        stats[f"{key}:any_nongold_pass"] += bool(other)
        stats[f"{key}:any_nongold_pass_excl_dup"] += bool(other_nodup)
        stats[f"{key}:gold_pass_and_nongold_pass"] += gold_pass and bool(other_nodup)
        stats[f"{key}:nongold_passes_total"] += len(other_nodup)
    out = {"queries_with_missing_runs": stats["queries_with_missing_runs"]}
    for key in ("trivial_output", "nontrivial_output"):
        n = max(1, stats[f"{key}:queries"])
        out[key] = {
            "queries": stats[f"{key}:queries"],
            "share_gold_in_top50": stats[f"{key}:gold_in_top{k}"] / n,
            "share_gold_passes": stats[f"{key}:gold_passes"] / n,
            "share_any_nongold_pass": stats[f"{key}:any_nongold_pass"] / n,
            "share_any_nongold_pass_excl_exact_duplicates": stats[f"{key}:any_nongold_pass_excl_dup"] / n,
            "share_gold_pass_and_a_nongold_pass": stats[f"{key}:gold_pass_and_nongold_pass"] / n,
            "mean_nongold_passes_per_query": stats[f"{key}:nongold_passes_total"] / n,
        }
    return out


# ---------------- 7. throughput ----------------
def throughput(dev, k: int = 50) -> dict:
    pos = np.load(topk_path(MODEL))["pos"][:, :k]
    cache = ExecCache()
    c = collections.Counter()
    runs_per_pair = []
    for qi in range(len(dev.query_ids)):
        stmt = dev.query_texts[qi]
        s = samples_for(stmt)
        ce = parse_calls(stmt) if s.kind != "stdin" else None
        for p in pos[qi]:
            prog = dev.corpus_texts[p]
            mode = program_mode(prog)
            if s.kind == "stdin" and mode == "stdin":
                c["run_stdin"] += 1
                n = 0
                for inp, exp in s.pairs:
                    r = cache.get(cache.key(prog, inp))
                    if r is None:
                        break
                    n += 1
                    if r.status != "OK" or not outputs_match(r.stdout, exp):
                        break
                runs_per_pair.append((n, len(s.pairs)))
            elif s.kind == "stdin" and mode == "call":
                c["skip_mode_mismatch_stdin_statement_call_program"] += 1
            elif ce is not None and ce.kind != "none":
                if mode == "stdin":
                    c["skip_mode_mismatch_call_statement_stdin_program"] += 1
                elif ce.func and "class Solution" not in prog and ce.func.replace("_", "").lower() not in {
                    n.replace("_", "").lower() for n in re.findall(r"^\s*def\s+(\w+)", prog, re.M)
                }:
                    c["skip_function_name_mismatch"] += 1
                else:
                    c["run_call"] += 1
            else:
                c["skip_no_samples_or_examples"] += 1
    total = sum(c.values())
    executed = [n for n, _ in runs_per_pair if n]
    all_samples = [m for n, m in runs_per_pair if n]
    return {
        "pairs_top50": total,
        "breakdown": {k: {"n": v, "share": v / total} for k, v in c.most_common()},
        "share_skippable_without_running": sum(v for kk, v in c.items() if kk.startswith("skip")) / total,
        "stdin_runs_per_executed_pair_early_stop": float(np.mean(executed)) if executed else None,
        "stdin_samples_per_executed_pair": float(np.mean(all_samples)) if all_samples else None,
    }


def main() -> None:
    dev = load_devset("val")
    _, _, qt = load_apps_texts()
    train = set(load_train_qrels())
    test_texts = [t for q, t in qt.items() if q not in train]
    res = {
        "note": "test side uses only mteb's saved scores and query texts (queries not in train qrels); no test qrels opened",
        "1_stage1_loss_buckets": loss_buckets(dev),
        "2_token_lengths": token_lengths(dev.query_texts, test_texts),
        "3_format_mix": format_mix(dev.query_texts, test_texts),
        "4_call_parser_misses": call_misses(dev),
        "5_gold_failures": gold_failures(dev),
        "7_throughput": throughput(dev),
    }
    try:
        res["6_false_passes"] = false_passes(dev)
    except Exception as e:  # noqa: BLE001
        res["6_false_passes"] = {"error": repr(e)}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
