"""Output A (Stage 1, predictions saved) then Output B (mteb two-stage with ExecutionReranker).

    # end-to-end check on validation (train split, seed 13; no test data)
    python -m codeintel.eval.run_two_stage --split validation
    # THE single Stage 2 test run (P approved): Output A must reproduce ndcg_at_10 0.74641 exactly
    python -m codeintel.eval.run_two_stage --split test --expect-a-ndcg 0.74641

Output B uses task.convert_to_reranking(<Output A predictions>, top_k=k) and mteb.evaluate on the
reranker (docs/LOG.md, Phase 0.2 item 6). Test qrels are loaded only inside mteb.evaluate (R3).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import mteb
from datasets import Dataset

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.run_mteb import _hardware
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.stage2.reranker import ExecutionReranker, load_stage2_cfg

log = logging.getLogger("two_stage")
S1 = "configs/stage1_final.yaml"
S2 = "configs/stage2_final.yaml"


def validation_task():
    """AppsRetrieval with the seed-13 validation split from the train qrels as its only split."""
    from mteb.tasks.retrieval.code.apps_retrieval import AppsRetrieval

    from codeintel.eval.devset import load_devset

    class AppsValidationSeed13(AppsRetrieval):
        metadata = AppsRetrieval.metadata.model_copy(update={"name": "AppsValidationSeed13", "eval_splits": ["validation"]})

        def load_data(self, num_proc=None, **kwargs):
            if self.data_loaded:
                return
            dev = load_devset("val")
            corpus = Dataset.from_dict({"id": dev.corpus_ids, "text": dev.corpus_texts, "title": [""] * len(dev.corpus_ids)})
            queries = Dataset.from_dict({"id": dev.query_ids, "text": dev.query_texts})
            self.dataset = {"default": {"validation": {"corpus": corpus, "queries": queries, "relevant_docs": dev.qrels, "top_ranked": None}}}
            self.data_loaded = True

    return AppsValidationSeed13()


def get_task(split: str):
    return mteb.get_task("AppsRetrieval") if split == "test" else validation_task()


def evaluate(model, task, out: Path, pred_folder: Path | None, meta: dict) -> dict:
    t0 = time.perf_counter()
    res = mteb.evaluate(model, task, cache=None, overwrite_strategy="always", encode_kwargs={"batch_size": 64},
                        prediction_folder=pred_folder, co2_tracker=False, num_proc=1)
    tr = res.task_results[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    tr.to_disk(out)
    split = next(iter(tr.scores))
    s = tr.scores[split][0]
    headline = {k: s[k] for k in ("ndcg_at_10", "mrr_at_10", "recall_at_1", "recall_at_10", "recall_at_20", "recall_at_100") if k in s}
    commit = __import__("codeintel.common.proc", fromlist=["run"]).run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    meta = meta | {"git_commit": commit, "elapsed_s": round(time.perf_counter() - t0, 1), "hardware": _hardware(), "headline": headline}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: %s", out.name, headline)
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--expect-a-ndcg", type=float, default=None, help="stop if Output A ndcg_at_10 differs")
    ap.add_argument("--k", type=int, default=None, help="override stage2_final k (validation checks only)")
    ap.add_argument("--stage2-config", default=S2, help="frozen Stage 2 config (test: k comes only from it)")
    ap.add_argument("--reuse-a", action="store_true",
                    help="skip Output A and re-rank its saved predictions (no second Stage 1 test pass)")
    ap.add_argument("--out-suffix", default="", help="e.g. _k50 -> appsretrieval_results_B_k50.json")
    a = ap.parse_args()
    s2_path = a.stage2_config
    s2 = load_stage2_cfg(s2_path)
    k = a.k or s2.k
    if a.split == "test" and a.k is not None and a.k != s2.k:
        raise SystemExit(f"test run must use the frozen k from {s2_path}")
    tag = "" if a.split == "test" else "validation_"
    pred = REPO_ROOT / "results" / "predictions" / f"{tag}stage1_qwen3"

    if a.reuse_a:
        if not any(pred.glob("*_predictions.json")):
            raise SystemExit(f"--reuse-a: no saved Output A predictions in {pred}")
        log.info("reusing Output A predictions from %s", pred)
    else:
        cfg = load_cfg(S1)
        cfg.on_cache_miss = "error"
        enc = PrePostPipelineEncoder(cfg)
        a_head = evaluate(enc, get_task(a.split), REPO_ROOT / "results" / f"{tag}appsretrieval_results_A.json", pred,
                          {"output": "A", "config": S1, "cache_stats": enc.stats})
        if a.expect_a_ndcg is not None and round(a_head["ndcg_at_10"], 5) != round(a.expect_a_ndcg, 5):
            raise SystemExit(f"Output A ndcg_at_10 {a_head['ndcg_at_10']} != expected {a.expect_a_ndcg}: stopping before Output B")

    from codeintel.eval.ceiling import _check_sandbox_ready

    _check_sandbox_ready()  # all hostile-program tests must pass before any APPS code runs (plan 11.3)
    task_b = get_task(a.split).convert_to_reranking(pred, top_k=k)
    rr = ExecutionReranker(s2_path)
    pred_b = REPO_ROOT / "results" / "predictions" / f"{tag}stage2_exec{a.out_suffix}"
    evaluate(rr, task_b, REPO_ROOT / "results" / f"{tag}appsretrieval_results_B{a.out_suffix}.json", pred_b,
             {"output": "B", "config": s2_path, "stage1_config": S1, "k": k, "outcome_counts": rr.stats,
              "reused_output_a_predictions": a.reuse_a})
    if a.split == "validation":
        check_against_tuning(pred_b, k)


def check_against_tuning(pred_b: Path, k: int) -> None:
    """mteb's Output B on the validation stdin subset must equal tune_stage2's unweighted number."""
    from codeintel.eval.devset import load_devset
    from codeintel.eval.metrics import per_query
    from codeintel.stage2.verifier import samples_for

    dev = load_devset("val")
    preds = json.loads(next(pred_b.glob("*_predictions.json")).read_text(encoding="utf-8"))
    run = preds["default"]["validation"]
    stdin_q = {q for q, t in zip(dev.query_ids, dev.query_texts) if samples_for(t).kind == "stdin"}
    ours = per_query({q: run[q] for q in stdin_q}, {q: dev.qrels[q] for q in stdin_q}, (10,))
    tuned = json.loads((REPO_ROOT / "results" / "stage2_tuning.json").read_text(encoding="utf-8"))["best_by_k"][str(k)]["metrics"]
    res = {"mteb_B_stdin_subset_ndcg_at_10": float(ours["ndcg_at_10"].mean()), "tune_stage2_ndcg_at_10": tuned["ndcg_at_10"]}
    res["abs_diff"] = abs(res["mteb_B_stdin_subset_ndcg_at_10"] - res["tune_stage2_ndcg_at_10"])
    (REPO_ROOT / "results" / "validation_two_stage_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("end-to-end check: %s", res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
