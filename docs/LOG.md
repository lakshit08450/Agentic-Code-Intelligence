# Experiment and verification log

Every measured number: command, config, git commit, hardware. Newest entries at the bottom.

Hardware (all entries unless stated): AMD Ryzen 9 270 (8C/16T), 15.3 GB RAM, NVIDIA RTX 5060 Laptop GPU 8 GB (driver 610.88), Windows 11 Home 10.0.26200.

---

## 2026-09-28 Phase 0.1: environment

- Repo moved out of OneDrive to `C:\dev\prism-code-intel` (plan Section 6).
- Python 3.11.9 installed with `winget install Python.Python.3.11 --scope user`; venv `py -3.11 -m venv .venv`.
- Installed: torch 2.11.0+cu128 (from `https://download.pytorch.org/whl/cu128`), mteb 2.21.8, sentence-transformers 6.1.0, transformers 5.17.0, datasets 5.0.1, numpy 2.4.6, pyarrow 25.0.1, pytest 9.1.1, pywin32 312, psutil 7.2.2, PyYAML 6.0.3.
- Test split file `data/apps/data/test-00000-of-00001.parquet` was deleted unopened on 2026-09-28 (R3). Local data: corpus 8,765 docs, queries 8,765, train qrels 5,000 rows (`query-id`, `corpus-id`, `score`). Corpus `title` is empty for every document.

### CUDA check on the RTX 5060 (plan Section 6, [H] resolved)
Command: inline script in session (torch version, arch list, fp16 matmul).
- `torch.__version__` = 2.11.0+cu128, `torch.version.cuda` = 12.8, `cuda.is_available()` = True.
- `get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, **sm_120**; device capability (12, 0). The cu128 build supports the RTX 5060.
- fp16 matmul 4096x4096, 20 iterations: 0.424 s (about 6.5 TFLOPS including warm-up); result matches an fp32 reference within fp16 rounding (max abs err 7.6e-2 on sums of 4,096 products).

## 2026-09-28 Phase 0.2: mteb API verification (installed mteb 2.21.8, source read)

Files read: `mteb/models/abs_encoder.py`, `mteb/models/models_protocols.py`, `mteb/models/search_wrappers.py`, `mteb/evaluate.py`, `mteb/_create_dataloaders.py`, `mteb/abstasks/retrieval.py`, `mteb/abstasks/retrieval_dataset_loaders.py`, `mteb/_evaluators/retrieval_metrics.py`, `mteb/models/model_meta.py`, `mteb/tasks/retrieval/code/apps_retrieval.py`, `mteb/models/model_implementations/{e5_models,gte_models}.py`.

1. **`encode` signature**: `encode(self, inputs: DataLoader[BatchedInput], *, task_metadata, hf_split, hf_subset, prompt_type: PromptType | None = None, **kwargs) -> Array`. Matches the plan's skeleton.
2. **Batch schema**: `inputs` is a torch `DataLoader`; each batch is a dict from `_custom_collate_fn`, with `batch["text"]` a list of strings (plan assumption is correct). Batches also carry other columns (e.g. `id`), which our encoder never reads (R1).
   - Documents: mteb maps each row with `_corpus_to_dict`: text = `(title + " " + text).strip()` if title non-empty, else `text.strip()`. The APPS titles are empty, so doc text = `text.strip()`.
   - Queries: text passed unchanged (no `instruction` column in APPS).
3. **`PromptType`**: `mteb.types.PromptType` enum, values `"query"` / `"document"`. `SearchEncoderWrapper` passes `prompt_type=PromptType.query` for queries and `PromptType.document` for the corpus. The corpus is encoded in one call per 50,000-doc chunk (all 8,765 docs in one call).
4. **`mteb_model_meta`**: `evaluate()` → `_sanitize_model` uses `model.mteb_model_meta` if present, else `ModelMeta.create_empty()`. Not strictly required, but the name must contain `/` (`_check_name` validator). `ModelMeta.create_empty(overwrites={...})` exists; `similarity_fn_name=None` makes `AbsEncoder.similarity` fall back to `cos_sim`. We set `ScoringFunction.COSINE` explicitly. `modalities` defaults to `["text"]`.
5. **Cross-encoder protocol**: `CrossEncoderProtocol.predict(self, inputs1: DataLoader, inputs2: DataLoader, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs) -> Array`. `SearchCrossEncoderWrapper.search` builds aligned loaders `queries.select(query_indices)` and `corpus.select(doc_indices)` (one row per pair, same order) and expects one score per pair. It raises if `top_ranked` is None.
6. **Two-stage reranking**: `AbsTaskRetrieval.convert_to_reranking(top_ranked_path, top_k=10)` exists (the newer "task conversion helper"; there is no `previous_results` argument to `evaluate`). It reads `{prediction_folder}/AppsRetrieval_predictions.json` (written when `evaluate(..., prediction_folder=...)` is set), keeps the top `top_k` per query and sets `dataset[subset][split]["top_ranked"]`. Usage for Output B: `task = mteb.get_task("AppsRetrieval").convert_to_reranking("results/predictions/stage1", top_k=K)` then `mteb.evaluate(reranker, task, ...)`. Note the default `top_k=10`; we must pass K explicitly.
   - The saved first-stage predictions hold `max(k_values)` = 1000 docs per query (`self._top_k = max(self.k_values)`), so K up to 1000 is available.
7. **Result JSON keys**: `make_score_dict` emits `ndcg_at_{k}`, `map_at_{k}`, `recall_at_{k}`, `precision_at_{k}`, `mrr_at_{k}` for k in (1, 3, 5, 10, 20, 100, 1000), plus nAUC keys. **MRR is `mrr_at_10`** (and `mrr_at_1000` ≈ full MRR). `main_score` for AppsRetrieval = `ndcg_at_10`. Eval split = `test`.
   - Tie-breaking in mteb's MRR: sort by (score, doc_id) descending, "to match pytrec_eval". Relevant only for exact score ties.
8. **Result cache**: `evaluate(..., cache=ResultCache | None, overwrite_strategy="only-missing" default)`. `OverwriteStrategy` values: `always`, `never`, `only-missing`, `only-cache`. Our runs pass `cache=None` (no read or write of `~/.cache/mteb`) and `overwrite_strategy="always"`, then write the `TaskResult` with `TaskResult.to_disk(path)` (the MTEB-generated JSON).
9. **Data loading**: the task loads HF `CoIR-Retrieval/apps` at revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`, configs `corpus`, `queries` (single split each) and `default` (qrels). Our `scripts/encode.py` loads only `corpus` and `queries` at the same revision, so cache texts match what mteb passes and no qrels are opened (R3).
10. **Reference prompts in mteb's own model registry**: e5-base-v2 uses `"query: "` / `"passage: "`, max_tokens 512. gte-modernbert-base has no prompts, max_tokens 8192, cosine. Our G1 configs copy these.
11. **Other**: `evaluate()` sets `encode_kwargs["batch_size"]=32` if absent; `num_proc` defaults to 1.

Plan adaptations: none needed to the rules. The encoder follows the plan skeleton; Output B uses `convert_to_reranking`.

## 2026-09-29 Phase 0.4: encodes and G1 (part 1)

### e5-base-v2 encode (GPU)
Command: `.venv/Scripts/python scripts/bg.py logs/encode_e5.log -- .venv/Scripts/python scripts/encode.py --config configs/bakeoff/e5_base_v2.yaml --device cuda`
Config: `configs/bakeoff/e5_base_v2.yaml`; commit b37f852; RTX 5060, fp16.
- Corpus: 8,765 texts, 8,690 embedded (64 from an earlier smoke run) in 43.2 s = 201 texts/s. Queries: 8,701 embedded in 51.5 s = 169 texts/s.
- Finding: the corpus has **8,754 unique texts among 8,765 documents** (11 exact duplicates after mteb's `strip()`). Relevant to the duplication risk in Section 4.

### G1, e5-base-v2 on test (plan-listed test run #1)
Command: `.venv/Scripts/python scripts/bg.py logs/g1_e5.log -- .venv/Scripts/python -m codeintel.eval.run_mteb --config configs/bakeoff/e5_base_v2.yaml --require-cache --out results/g1_e5_base_v2.json`
Config: `configs/bakeoff/e5_base_v2.yaml`; commit f7dadda; CPU (Ryzen 9 270), all vectors from cache.
- **ndcg_at_10 = 0.11532** (reference 11.5 x100: **G1 part 1 PASS**). mrr_at_10 = 0.09882, recall_at_10 = 0.16946, recall_at_100 = 0.34369.
- Cache: 12,519 hits, 0 misses (`--require-cache`). Wall time 35.5 s.
- VERIFY resolved (Section 4): mteb standardised 8,765 corpus documents for the test split, so the test corpus is the full corpus with train solutions as distractors.

### gte-modernbert-base revision pin
- The revision in mteb's registry, `7ca8b4ca700621b67618669f5378fe5f5820b8e4`, does not exist on the Hub (`config.json` → 404, not among the repo commits). Pinned to current main `e7f32e3c00f91d699e8c43b53106206bcc72bb22` (2025-07-04; weights unchanged since the 2025-01-22 safetensors commit; includes the upstream "patch inference on CPU & Windows" commit).

## 2026-09-29 Fix: console windows popping up (plan Section 8 updated by P)

Symptom (reported by P): new terminal windows kept opening while jobs ran.
Cause: `scripts/bg.py` launched jobs with `DETACHED_PROCESS`. A detached process has no console, so every console child it starts (git from `_git_rev`, pip, DataLoader workers, later sandbox runs) allocates its own visible console window. (`CREATE_NO_WINDOW` is ignored when combined with `DETACHED_PROCESS`.)
Changes:
- New `src/codeintel/common/proc.py`: the only place that starts subprocesses. `run()` / `popen()` always add `CREATE_NO_WINDOW` on Windows; `popen(background=True)` uses `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`. `DETACHED_PROCESS` is never used.
- `scripts/bg.py` launches through `proc.popen(background=True)`; new `--stop <log>` terminates the job's process tree from the `.pid` file (terminate, wait 15 s, then kill).
- `encoder._git_rev` and `run_mteb` use `proc.run`.
- DataLoader workers: `run_mteb` passes `num_proc=1` to `mteb.evaluate`, so mteb's `create_dataloader` uses `num_workers=0` (it only sets workers when `num_proc > 1`). sentence-transformers `encode` does not use a DataLoader.
- `tests/test_no_console_windows.py` (AST scan): no direct `subprocess.run/Popen/call/check_*`, `os.system`, `os.startfile`, `DETACHED_PROCESS`, or `start` / `cmd /c start` launches anywhere in `src/` or `scripts/`; the detector is itself tested on a bad snippet. Suite: 10 passed.
Procedure: stopped the running gte encode with `bg.py --stop logs/encode_gte.log` → "pid 27860 STOPPED (3 processes, 0 force-killed)"; no partial `.tmp` chunk left; corpus (18 chunks) kept. Restarted with the fixed `bg.py`; it resumed (corpus all cached, 8,253 queries remaining).
Verification: a watcher (EnumWindows snapshot diff over all processes, 0.25 s polling, 90 s) during the restarted job reported **0 new visible top-level windows**; job tree = venv `python.exe` → base `python.exe` + hidden `conhost.exe`.

## 2026-09-29 Phase 0.5/0.6: G1 part 2 and Output A v0 (plan-listed test run #2)

### gte-modernbert-base encode (GPU)
Command: `.venv/Scripts/python scripts/bg.py logs/encode_gte.log -- .venv/Scripts/python scripts/encode.py --config configs/bakeoff/gte_modernbert.yaml --device cuda`
Config: `configs/bakeoff/gte_modernbert.yaml` (rev e7f32e3c, max_seq_length 8192, no prompts); commits f7dadda → 8706ac5 (stopped and resumed for the window fix); RTX 5060, fp16.
- Corpus: 8,754 unique texts in 136.6 s = 64.1 texts/s. Queries: 8,253 in 134.9 s = 61.2 texts/s (512 embedded before the stop).

### Output A v0 = G1 part 2
Command: `.venv/Scripts/python scripts/bg.py logs/outputA_v0.log -- .venv/Scripts/python -m codeintel.eval.run_mteb --config configs/bakeoff/gte_modernbert.yaml --require-cache --out results/appsretrieval_results_A.json --prediction-folder results/predictions/stage1`
Config: `configs/bakeoff/gte_modernbert.yaml`; commit 8706ac5; CPU (Ryzen 9 270), all vectors from cache (12,519 hits, 0 misses); wall 33.9 s; mteb 2.21.8, dataset rev f22508f9.
- **ndcg_at_10 = 0.57738** (reference 56.4 x100 → **G1 part 2 PASS**, +1.3 points). mrr_at_10 = 0.52966, mrr_at_1000 = 0.53785, ndcg_at_1 = recall_at_1 = 0.44223, recall_at_10 = 0.72961, recall_at_20 = 0.79734, recall_at_100 = 0.91873.
- The +1.3-point gap to the published figure is not investigated on test (R2/R3). Plausible causes: different model revision, full 8,192-token queries vs a shorter cap in the papers. Validation numbers in Phase 1 will set our own baseline.
- Files: `results/appsretrieval_results_A.json` (MTEB `TaskResult.to_disk`), `results/appsretrieval_results_A.meta.json` (config, commit, hardware, cache stats). First-stage predictions `results/predictions/stage1/AppsRetrieval_predictions.json` (108 MB, top-1000 per query) kept locally for Output B, gitignored (over GitHub's 100 MB file limit).
- This is the fallback submission. Per P: no GitHub remote or release yet; commit locally only.

**G1: PASS** (e5-base-v2 0.11532 vs 0.115; gte-modernbert-base 0.57738 vs 0.564).

## Notes for Stage 2 error analysis (Phase 2b)
- The corpus contains 11 exact duplicate documents (8,754 unique texts among 8,765). Per P, they stay in the corpus. Identical texts get identical embeddings and identical execution outcomes, so a duplicate of the gold solution ties with it at every stage; the final order is then decided by mteb's tie-break (doc id), which we do not control. Count how many validation misses at rank 1 involve an exact-duplicate text when doing the Section 13 error analysis.
