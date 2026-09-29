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

---
# Phase 1 + Phase 2a (night of 28/29 Sep; P asleep from ~01:15, working autonomously under P's pre-approved rules)

## Phase 1.1 validation split
Command: `.venv/Scripts/python -m codeintel.eval.devset` -> `results/devset_split.json`. Commit fc4bcdd.
- Train qrels (local parquet, 5,000 pairs, one relevant doc per query, score 1) split by `numpy.random.default_rng(13).permutation` over sorted query IDs: train 4,000 / validation 1,000. val IDs sha1 a9ce25b5..., train IDs sha1 87c49589....
- Validation corpus = full 8,765-doc corpus. All train-qrel query/doc IDs exist in the pinned HF `queries`/`corpus` configs; local query texts equal HF texts.
- Observation (R1): IDs are `q1`/`d1`-style with matching numbers for pairs. Scoring never reads IDs (`tests/test_no_id_leak.py`).

## Phase 1.2 metrics
- `src/codeintel/eval/metrics.py` (NDCG/MRR/recall@k, pytrec_eval tie-break) cross-checked against mteb's `calculate_retrieval_scores` (pytrec_eval) on gte-modernbert validation: max abs diff **3.2e-6** (ndcg_at_10), 0 (mrr_at_10, recall_at_100). Target < 1e-3: PASS.
- `bootstrap.py`: paired bootstrap, 1,000 resamples, seed 13.

## Phase 1.3 bake-off (validation, 1,000 queries, full corpus; GPU-filled caches, fp16)
Command: `.venv/Scripts/python -m codeintel.eval.bakeoff metrics --configs <cfg>`; configs in `configs/bakeoff/`. Results `results/bakeoff.csv`, per-query `results/bakeoff_perquery.json`, top-100 in `cache/stage1/<name>_val_top100.npz`.

| model | NDCG@10 | MRR@10 | R@1 | R@10 | R@20 | R@50 | R@100 | CPU docs/s | CPU queries/s |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | **0.8344** | 0.8051 | 0.738 | 0.924 | 0.956 | **0.977** | 0.987 | 3.65 | 0.90 |
| gte-modernbert-base | 0.7023 | 0.6678 | 0.594 | 0.810 | 0.851 | 0.902 | 0.929 | 0.74 | 0.33 |
| CodeRankEmbed | 0.6373 | 0.6166 | 0.574 | 0.702 | 0.731 | 0.759 | 0.800 | 4.75 | 1.52 |
| e5-base-v2 (harness only) | 0.4266 | 0.3997 | 0.348 | 0.512 | 0.550 | 0.614 | 0.650 | - | - |

- Paired bootstrap Qwen3 minus gte on validation: NDCG@10 +0.132 [0.113, 0.153]; recall@50 +0.075 [0.059, 0.095]; recall@100 +0.058 [0.043, 0.075]. All exclude 0.
- GPU encode rates (RTX 5060, fp16): CodeRankEmbed corpus 112/s, queries 75/s; Qwen3 queries 18.4/s.
- CPU throughput: `python -m codeintel.eval.bakeoff throughput --n 100` (100 random docs + 100 random validation queries, seed 13, fp32, torch 8 threads, CPU otherwise idle). gte-modernbert is unexpectedly slow on CPU (0.74 docs/s, slower than the 4x larger Qwen3); not investigated tonight.
- **Projected full CPU test run for Qwen3 without cache**: corpus 8,765 / 3.65 = 40 min + 3,765 test queries / 0.90 = 70 min = **about 1 h 50 min** (well under P's 8 h limit). From cache (the planned final run) it takes about 35 s.
- Encoder changes for the bake-off: (a) CodeRankEmbed's remote NomicBert code calls `get_extended_attention_mask`, removed in transformers 5 -> shim restoring the transformers 4.x implementation, applied only to trust_remote_code models; (b) CodeRankEmbed uses eager attention; 8 x 8,192 tokens OOMed (12 GB) -> length-aware batching with batch_size x len^2 <= 8192^2 (`attn_budget`); vectors unchanged (each text still encoded independently, padding masked).

### E5 inflation check (P's request)
- e5-base-v2: validation NDCG@10 **0.4266** vs test **0.1153** (ratio 3.7x). gte-modernbert: 0.7023 vs 0.5774 (1.22x).
- e5 is NOT close on validation and test, so the condition "e5 close while gte is not" does not hold. Validation (APPS train problems) is much easier than test for every model, including e5-base-v2, which has no known APPS training. The val-to-test gap is therefore not evidence of APPS-train contamination for gte-modernbert (its relative gap is the smallest).
- Consequence for Qwen3: validation is inflated for all models, so its 0.834 will not transfer to test. Its lead over gte on validation is large and significant, but validation cannot prove it transfers. The pre-declared zero-shot test run (step 3) decides.

## Phase 1.5 licenses and stated training data
| model | license | stated training data | APPS/CoIR listed? |
|---|---|---|---|
| gte-modernbert-base (rev e7f32e3c) | Apache-2.0 | mGTE recipe (paper); mteb registry lists 17 sets (MSMARCO, NQ, HotpotQA, FEVER, MIRACL, MrTyDi, DuRetrieval, T2Retrieval, ...) | no |
| CodeRankEmbed (rev 3c4b6080) | MIT | CoRNStack, 21M GitHub code/docstring pairs; init Arctic-Embed-M-Long | no (not in mteb registry) |
| Qwen3-Embedding-0.6B (rev 97b0c614) | Apache-2.0 | mteb registry lists 11 sets incl. CodeSearchNet, MSMARCO, NQ, HotpotQA, FEVER | no |
None states APPS or CoIR training data; closed training mixes cannot be fully verified.

## Phase 2a sandbox build (plan 11.3)
- `scripts/setup_sandbox.ps1`: python-3.11.9-embed-amd64.zip, SHA256 009D6BF7E3B2DDCA3D784FA09F90FE54336D5B60F0E0F305C37F400BF83CFD3B; `python.exe` Authenticode Valid (CN=Python Software Foundation). Firewall rule created by P as admin (about 01:03).
- Implementation choices (all additive or neutral):
  - `-B` added to `-I -X utf8` (no bytecode writes, which would otherwise trip the write guard).
  - `exit`/`quit` builtins defined by the runner (the embeddable package skips `site`; many APPS solutions call `exit()`).
  - Program runs in a thread with a 128 MB stack (Windows main thread has 1 MB; 256 MB is rejected by Windows).
  - Job also has DIE_ON_UNHANDLED_EXCEPTION and UI restrictions; the runner calls SetErrorMode so crashes never open a dialog.
  - The runner verifies job membership by querying its own job's limits (ActiveProcessLimit=1, 512 MB), because a process may already sit in an inherited job.
  - `_winapi` is already loaded at interpreter start, so the "import" audit event never fires for it; the runner unloads `ctypes*`/`_winapi` from sys.modules before installing the hook; `_winapi.*` audit events are also blocked.
- Bugs found by tests and fixed (each has a regression test): `threading.stack_size(256 MB)` rejected on Windows; `import _winapi` not blocked; MemoryError while printing the traceback left exit code 0 (OK), so the exit code now defaults to failure; T_WALL included process start-up (below).
- `tests/test_sandbox.py` (33 tests): infinite loop, sleep, 2 GB and incremental memory bombs, child processes (subprocess, os.system, startfile, multiprocessing, execv, spawnv), socket via guard, network via firewall only (no guard), job child-process and memory limits without guard, writes outside tmp (open/os.open/pathlib), delete/rename/rmtree, 100 MB output, ctypes/_ctypes/_winapi, swallowed violations, native crash, tmp cleanup, markers in stdout, late READY, parallel start-up, serial TIMEOUT re-run. 33/33 pass (run twice).

### Gold run v0 (sandbox v1; INVALID because of the timer bug, kept for the record)
Command: `python -m codeintel.eval.ceiling gold --workers 12` (log `logs/gold_v0.log`), 85 s for 1,000 queries.
- UNKNOWN 670 (call_based 476, no_samples 177, parse_failure 11, py2 4, nonstdlib 2), PASS 159, TIMEOUT 122, WRONG 35, ERROR 14.
- The 122 TIMEOUTs were an artefact: re-run alone, the gold solutions finish in 0.15-1.1 s with correct output.

### Diagnosis: Windows process creation cost
- Bare `Popen` median: sandbox `python.exe` 0.56 s, venv `python.exe` 1.0 s, `cmd.exe` 0.004 s. The cost is specific to launching Python executables (most likely on-access antivirus/reputation scanning). Defender settings NOT changed (plan: needs P's approval).
- With 12 workers: start-up p50 1.9 s, p95 3.2 s; post-READY time p95 up to 2 s under load.
- Fixes (sandbox v3): the runner writes READY to stderr right before executing the program, and T_WALL (2 s) counts from READY; start-up allowance 15 s (result UNKNOWN startup_timeout, not cached, if exceeded). READY is honoured only as the first bytes of stderr; stdout never carries markers and is compared exactly as printed. (P asked for "start of stdout"; the markers live on stderr, so the same rule is applied there and stdout stays marker-free; tested.) A TIMEOUT is re-run once alone (a shared/exclusive gate pauses new parallel runs) before caching. The Job's CPU-time limit (T_WALL + 1 s user time) is unchanged.

### Worker-count sweep (step 1 decision)
Command: `python -m codeintel.eval.ceiling bench --model gte-modernbert-base --runs 500 --workers W` for W in 4, 8, 12, 16 (same 500 runs: top-10 gte candidates x samples of the first stdin validation queries; uncached; every TIMEOUT re-run alone). Sandbox v3, commit after fc4bcdd (uncommitted sandbox v3). Files `results/sandbox_bench_w*.json`.

| workers | runs/s | TIMEOUTs in parallel | of which spurious (pass alone) | effective runs/s incl. serial re-runs |
|---|---|---|---|---|
| 4 | 1.75 | 23 | 21 | 1.62 |
| 8 | 3.31 | 31 | 29 | 2.83 |
| 12 | 4.86 | 23 | 21 | **4.08** |
| 16 | 5.13 | 35 | 33 | 3.81 |

- **Decision: 12 workers** (best effective throughput; 16 adds spurious timeouts). Spurious timeouts exist even at 4 workers: single runs show random ~1-2 s stalls (process creation 0.2-2 s, occasional post-READY stalls), consistent with on-access scanning. The serial TIMEOUT re-run absorbs them.
- Throughput is about 4 runs/s, far below Linux expectations; this bounds tonight's Stage 2 depth (step 6).

## Step 3 PRE-DECLARATION (written before the run): Qwen3-Embedding-0.6B zero-shot test run
- Authorised by P (night instructions, item 3): ONE test run, zero-shot, nothing tuned, logged like G1. This is test run #3 overall (after G1 e5 and G1/Output A v0 gte).
- Config: `configs/bakeoff/qwen3_0p6b.yaml` exactly as used in the bake-off (card query format with the plan's instruction, no doc prompt, 8,192-token cap, raw text). No change was made to it after seeing validation numbers.
- Command: `python -m codeintel.eval.run_mteb --config configs/bakeoff/qwen3_0p6b.yaml --require-cache --out results/test_qwen3_zeroshot.json` (no prediction folder; CPU, vectors from the GPU-filled cache).
- Decision rule (P's): Stage 1 = Qwen3 if its test ndcg_at_10 > 0.57738 (gte-modernbert G1) AND the projected full CPU test run is under 8 h (projected 1 h 50 min, measured before this run); otherwise keep gte-modernbert. No other test runs of any kind.

## Step 3 RESULT: Qwen3-Embedding-0.6B zero-shot on test (test run #3)
Command as pre-declared; commit 4973978; CPU (Ryzen 9 270), cache 12,519 hits / 0 misses; 25.1 s. Files `results/test_qwen3_zeroshot.json` (+ `.meta.json`).
- **ndcg_at_10 = 0.74641**, mrr_at_10 = 0.70044, mrr_at_1000 = 0.70495, ndcg_at_1 = recall_at_1 = 0.60531, recall_at_10 = 0.89031, recall_at_20 = 0.93493, recall_at_100 = 0.98380.
- Rule check: 0.74641 > 0.57738 (gte-modernbert) and projected CPU full run 1 h 50 min < 8 h. **Decision: Stage 1 model = Qwen3-Embedding-0.6B** (config `configs/bakeoff/qwen3_0p6b.yaml`, unchanged).
- Val to test: 0.8344 -> 0.7464 (ratio 1.12, smallest of the three measured models).
- No further test runs tonight. Output A JSON (`results/appsretrieval_results_A.json`) still holds the gte-modernbert v0 run; regenerating Output A with Qwen3 is a Phase 4 action (frozen config), not done tonight.

## Step 4: gold failures, parser fixes, true call-based share

### Gold run with the fixed sandbox and the OLD parser (sandbox v3)
Command: `python -m codeintel.eval.ceiling gold --workers 12` (log `logs/gold_v3_oldparser.log`, `results/gold_outcomes_v3_oldparser.json`), commit 35441e6. 114 s.
- PASS 246, WRONG 58, ERROR 22, UNKNOWN 674 (call_based 476, no_samples 177, parse_failure 11, py2_syntax 7, nonstdlib 3). PASS among statements with parsed stdin samples: 246/336 = **73.2%**. No TIMEOUTs (all 122 in v0 were the timer artefact).

### 20 gold failures and causes (first 20 WRONG/ERROR by query id, from the run above)
| # | query | outcome | cause | category |
|---|---|---|---|---|
| 1 | q1103 | ERROR ValueError int('') | blank lines inside the CodeChef sample input (`2\n\n5\n\n72`) | B blank lines |
| 2 | q1107 | WRONG | expected output includes author credits ("By: Chintan, ...") | A prose in output |
| 3 | q1140 | ERROR IndexError | blank lines in input + "Explanation:" in expected output | A + B |
| 4 | q1163 | WRONG | gold prints `4.0` where `0` expected (gold bug) + explanation in output | E gold wrong |
| 5 | q1245 | ERROR unpack | blank lines inside the sample input | B |
| 6 | q1289 | WRONG | "Explanation" paragraph in expected output | A |
| 7 | q1379 | WRONG | "Explanation" line directly after the output (no blank line) | A |
| 8 | q1432 | ERROR ValueError | blank lines inside the sample input | B |
| 9 | q1443 | ERROR ValueError | blank lines inside the sample input | B |
| 10 | q1499 | ERROR unpack | blank lines inside the sample input | B |
| 11 | q1505 | WRONG (no output) | gold prints nothing on the parsed input | E / unclear |
| 12 | q1524 | WRONG | "(Explanation: 10+3+7+3)" appended to expected output | A |
| 13 | q1538 | ERROR ValueError | blank lines inside the sample input | B |
| 14 | q1562 | WRONG | several valid outputs (any valid construction accepted) | C multiple answers |
| 15 | q1601 | ERROR ValueError | blank lines inside the sample input | B |
| 16 | q1607 | WRONG sample 2 | gold solution wrong on sample 2 | E |
| 17 | q1682 | WRONG | several valid outputs | C |
| 18 | q1687 | WRONG | "Note: Your program should not print ..." in expected output | A |
| 19 | q1692 | ERROR EOFError | sample input truncated/mis-split by the statement layout | parse |
| 20 | q1704 | WRONG (no output) | gold defines functions only (call-based misparsed as stdin) | D call-based |
Beyond the first 20: 13 more "explanation prose after a blank line" (AtCoder `-----Sample Output-----` sections: q2063, q2211, q2243, q2245, q2263-q2266, q2272, q2289, q2290, q2302, ...), 5 more multiple-valid-answer problems (q2032, q2096, q2106, q2157, q2158), 3 more LeetCode-style (tree / list / class-design examples: q1889, q1907, q1922).

### Main parser-miss patterns (statements whose gold reads stdin but no samples were parsed)
- P1 data on the header line inside a dashed Example block: `Input:4` / `Output:Henry` (5 statements).
- P2 unusual titles or header lines with trailing text: `-----Example Text Case-----`, `-----EXAMPLE-----Input:`, `-----Sample Input:-----Sample Input:`, glued `ExampleInput:`.
- P3 `-----Sample Input-----` followed by a plain `-----Output-----`.
- P4 HackerRank `=====Input Format=====` statements with no sample in the text at all (about 15): cannot be fixed.

### Fixes (all in `src/codeintel/stage2/sampleio.py`, one test each in `tests/test_sampleio_formats.py`)
1. Inline header data in dashed blocks only (plain LeetCode `Example 1:` blocks stay call-based).
2. Titles `example text case`, glued `exampleinput`, `=====X=====` headers, header lines with trailing text (kept as the first content line unless it repeats the title).
3. Sample input section followed by a generic `Output` section.
4. Expected output trimmed where explanation prose starts (an `Explanation`/`Note`/`Hint` line anywhere, or after a blank line a line starting with typical prose openers or a >=6-word sentence). Blank-line-separated answers (`YES\n\nNO`) are kept.
5. Blank lines dropped from sample inputs.
6. Verifier routing by the candidate program's text (`verifier.program_mode`): programs that read stdin use stdin samples; programs that only define functions / `class Solution` go to the call-based path (fixes category D).

### Coverage before and after (validation, 1,000 statements)
| | before | after |
|---|---|---|
| statements with stdin samples | 336 | 344 |
| call_based (statement text) | 476 | 476 |
| none | 177 | 170 |
| parse_failure | 11 | 10 |
| gold PASS among stdin-parsed | 246/336 = 73.2% | 299/344 = **86.9%** |

Gold run after the fixes (`logs/gold_v3_newparser.log`, `results/gold_outcomes_v3_newparser.json`): PASS 299, WRONG 25, ERROR 4, UNKNOWN 672 (call_program 606, no_samples 32, call_based-statement-but-stdin-program 20, py2_syntax 7, parse_failure 4, nonstdlib 3). Remaining WRONG are mostly multiple-valid-answer problems and gold bugs.

### True call-based share
- By the gold solution's own form (`program_mode`: defines functions / `class Solution`, never reads stdin): **606 / 1,000 = 60.6%** of validation queries (456 whose statement the parser also marks call_based, 138 with no recognised examples, 12 other).
- That is above P's 25% threshold, so the call-based harness is built (step 5).

## Step 6 depth decision (01:48)
- Measured throughput 4.08 effective runs/s (12 workers). Estimated work: stdin part 344 queries x k x ~1.05 runs; call-based part ~600 queries x k x 1 run (one process per program running all examples). top-100 total ~96k runs = ~6.5 h -> cannot finish by 07:00 (harness still to be built). **Decision: top-50** (also the largest k in the Section 11.5 grid).
- Order: stdin part now (Qwen3 validation candidates, `cache/stage1/qwen3-embedding-0.6b_val_top100.npz`, first 50); build the call-based harness meanwhile without running sandbox tests under load; run the full sandbox suite on an idle CPU before any harness run; then re-run the same top-50 job (stdin pairs are cache hits) to add call-based pairs.

---
## 2026-09-29 morning: stdin top-50 completion and gap diagnosis

### Stage 2 stdin run (Qwen3 validation top-50) completed
- The overnight run crashed at about 14,000/17,200 pairs (about 03:00): `compare.tokens_match` called `int()` on a 101,110-digit token (CPython's 4,300-digit limit). Fixed by comparing canonical digit strings, with a regression test (`tests/test_compare.py::test_huge_integers_do_not_crash`). Resumed with `logs/stage2_top50_stdin_resume.log`: every pair was already cached (the crash was in a comparison after the run). Final counts over 17,200 pairs: ERROR 8,734, WRONG 5,890, UNKNOWN 2,086, PASS 443, TIMEOUT 47. Exec cache 16,028 runs.

### Gap diagnosis (`scripts/gap_analysis.py` -> `results/gap_analysis.json`; also `scripts/stage2_preview.py`, `scripts/test_stage2_cost.py`)
Test labels were not opened. Test-side numbers use only mteb's saved scores and test query TEXTS (queries not in the train qrels).

1. **Stage 1 loss buckets.** Test (derived from mteb recall_at_k, one relevant doc per query): rank 1 60.5%, 2-10 28.5%, 11-20 4.5%, 21-100 4.9%, 101-1000 1.5%, >1000 0.2%. Validation: rank 1 73.8%, 2-10 18.6%, 11-50 5.3%, 51-100 1.0%, >100 1.3%. Most test losses sit at ranks 2-10, which Stage 2 re-ranking targets.
2. **Qwen3 config and lengths.** max_seq_length 8,192 (+20 prompt tokens). Validation query tokens p50 324, p99 1,257, max 4,046; test p50 477, p99 1,138, max 1,861. Truncated share: 0% on both. The query-cap ablation (Phase 3a) cannot matter.
3. **Format mix (label-free).** Validation: Codewars call 42.0%, CodeChef 19.5%, LeetCode call 11.9%, no samples/unknown 10.5%, Codeforces 10.4%, AtCoder 3.0%, HackerRank 2.2%. Test: **Codeforces 78.0%, AtCoder 18.5%**, CodeChef 1.3%, LeetCode 0.8%, no samples 0.8%, HackerRank 0.4%. The parser finds stdin samples in **97.9% of test statements** (3,686/3,765) vs 34.4% of validation. Validation (APPS train) is a very different mix from test, which explains much of the val-test gap and means the call-based harness barely affects screening.
4. **Call-based parser misses** (339 unparsed call-based validation statements; 40 sampled, seed 13, reviewed by hand): prose-only examples 14, no examples in the text 13, named variables / multi-line outputs / expression arguments 5, fixable parser gaps 4, other-language syntax 3, keyword args 1, tree/linked-list 0, in-place 0. (The regex pre-bucketing over-counted other-language syntax, 15; the manual review is authoritative.)
5. **Gold non-PASS on validation (1,000).** Call-based, harness not built: 606. Among stdin gold: no samples 32 (mostly HackerRank), statement flagged call-based but program reads stdin 20, **multiple valid answers 16** (e.g. q1562), genuine mismatch or label noise 12 (e.g. q1163, gold prints `4.0` for `0`), py2 syntax 7 (q1120), parse failure 4 (q1821), non-stdlib import 3 (q1400), version-removed API 1 (q965; the static scan finds 4 gold programs using removed APIs). Zero Linux-only imports, zero recursion/stack failures, zero sandbox-guard false positives, zero remaining parser contamination by the heuristic check.
   - Gold PASS by family: Codeforces 85/104 (81.7%; all 17 others WRONG, mostly "print any" problems), AtCoder 24/30 (80%), CodeChef 181/195 (92.8%).
6. **Distractor false passes** (validation top-50, stdin statements, cache only). Trivial expected outputs (every sample output a single token such as 0/1/YES/NO/-1): 62 queries; at least one non-gold candidate passes in **56.5%**; gold passes in 85.5%. Non-trivial outputs: 282 queries; a non-gold candidate passes in **11.0%**; gold passes in 84.0%. No false pass was an exact duplicate of the gold text.
7. **Throughput.** Of the 50,000 validation top-50 pairs, 58.1% can be skipped without running (no samples/examples 37.9%, call statement with a stdin program 8.6%, function-name mismatch 8.9%, stdin statement with a call program 2.8%). Stdin pairs execute 1.01 processes on average (wrong candidates fail on sample 1), so running all samples in one process would gain about 1%. On test only about 12% of pairs are skippable; execution needs 33.6k / 66.2k / 162k pairs for k = 10 / 20 / 50, i.e. **2.4 h / 4.6 h / 11.4 h at the measured 4 runs/s**. Process start-up (~0.56 s per python.exe CreateProcess) dominates.

### Stage 2 preview (diagnostic, NOT a tuning run)
`scripts/stage2_preview.py` -> `results/stage2_preview.json`: one fixed, untuned rule (cosine + 1.0 x PASS) on the 344 validation stdin queries, from the exec cache. NDCG@10 **0.718 -> 0.920** (+0.203, 95% CI [0.169, 0.235]); MRR@10 0.670 -> 0.907; rank-1 0.576 -> 0.875. The real selection remains the Section 11.5 grid with the G3 gate.

## 2026-09-29 midday: start-up cause, Stage 2 tuning, freeze, end-to-end check

### Start-up cost (P's decision)
P reports McAfee real-time scanning off and Defender not running; no exclusions added. Evidence (bare `python.exe -c pass` 0.56 s vs `cmd.exe` 0.004 s; venv python 1.0 s) shows every python.exe launch is slow, so the cause is system-level (likely McAfee process monitoring or Smart App Control). Logged and accepted; current throughput about 4 runs/s. The runner's Job-Object wait (2 ms poll) is not the cause.

### Fixes 3 and 4 (`src/codeintel/stage2/reranker.py`, tests `tests/test_reranker.py`)
- Fix 3: `multi_answer_neutral` - no WRONG penalty when the statement accepts any valid answer (regex on statement text).
- Fix 4: `trivial_pass_factor` - PASS weight multiplied by this factor when every sample output is a single trivial token.

### Stage 2 grid + G3 (`python -m codeintel.eval.tune_stage2 --k-for-test 20`, commit d85063b)
- Scope: 344 validation queries with stdin samples; exec cache only (no code executed). Queries weighted by test_share/val_share of their statement family (Codeforces 104 and AtCoder 30 queries carry most weight). Grid 162 configs: k {10,20,50} x w_pass {0.1,0.3,1.0} x w_wrong {0,0.05,0.1} x w_err {0,0.05} x trivial_pass_factor {1.0,0.5,0.1}; fix 3 on. Files: `results/stage2_grid.csv`, `results/stage2_tuning.json`.
- Stage 1 (Qwen3) on this subset: weighted NDCG@10 0.7827, MRR@10 0.7324 (unweighted 0.7178 / 0.6701).

| k | best config (w_pass, w_wrong, w_err, trivial) | weighted NDCG@10 | weighted MRR@10 | G3 weighted delta NDCG@10 [95% CI] | fix 3 off | fix 4 off |
|---|---|---|---|---|---|---|
| 10 | 0.3, 0, 0.05, 1.0 | 0.9141 | 0.9055 | +0.131 [0.095, 0.169] | 0.9141 | 0.9141 |
| 20 | 0.3, 0.05, 0.05, 0.1 | **0.9224** | 0.9138 | **+0.140 [0.101, 0.179]** | 0.9159 | 0.9202 |
| 50 | 0.3, 0.05, 0.05, 0.1 | 0.9444 | 0.9357 | +0.162 [0.118, 0.206] | 0.9379 | 0.9405 |

- k for test: 20 (the run can start before 18:00; k=50 would need ~11 h). **G3 passes. Frozen `configs/stage2_final.yaml`**: k 20, w_pass 0.3, w_wrong 0.05, w_err 0.05, trivial_pass_factor 0.1, multi_answer_neutral true. **Frozen `configs/stage1_final.yaml`** = the Qwen3 bake-off config (Phase 3 ablations skipped: no query is truncated; time goes to Stage 2).

### End-to-end two-stage check on validation (`python -m codeintel.eval.run_two_stage --split validation`)
- mteb Output A (stage1_final, predictions saved) -> `convert_to_reranking(top_k=20)` -> `mteb.evaluate(ExecutionReranker)`. Validation Output B over all 1,000 queries: ndcg_at_10 0.89177, mrr_at_10 0.87510, recall_at_1 0.831 (call-based queries keep Stage 1 order).
- On the stdin subset, mteb's Output B NDCG@10 = **0.8846696**, identical to tune_stage2 (abs diff 0.0). `results/validation_two_stage_check.json`.

### PRE-DECLARATION: the single Stage 2 test run (P approved)
- Command: `python -m codeintel.eval.run_two_stage --split test --expect-a-ndcg 0.74641` via bg.py (log `logs/test_two_stage.log`).
- Step A: Output A with `configs/stage1_final.yaml` (identical to the logged Qwen3 zero-shot run), predictions saved to `results/predictions/stage1_qwen3`; the job stops if ndcg_at_10 != 0.74641.
- Step B: the full sandbox suite must pass, then Output B = mteb two-stage with ExecutionReranker and frozen `configs/stage2_final.yaml` (k 20) -> `results/appsretrieval_results_B.json`.
- Estimated work: 66,163 stdin pairs, about 4.6 h at 4 runs/s.

### Stage 2 test run (started 12:11, pid in logs/test_two_stage.log.pid)
- Output A reproduced exactly: ndcg_at_10 **0.74641**, mrr_at_10 0.70044 (same as the zero-shot run); `results/appsretrieval_results_A.json` now holds this Qwen3 Output A (the gte-modernbert v0 file is superseded; its numbers stay in this log). Predictions saved to `results/predictions/stage1_qwen3/`.
- Sandbox suite inside the job: 33 passed. Output B in progress: 75,300 pairs (3,765 x k=20; UNKNOWN pairs cost nothing). Progress at 13:56: 19,000 pairs in 103 min (~3.1 pairs/s while Path B was being developed on the same CPU); ETA ~19:05.

## Path B (P1 --rev first, then Bonus --range), commit a59bd00
- `src/codeintel/versioned/`: `units.py` (files > 300 lines -> 80-line windows, 20 overlap), `ingest.py` (git first-parent revisions with `git diff -M` rename detection, snapshot dirs, JSONL), `store.py` (index.sqlite per plan 12.2 plus `units.key` and a `snippets` table; vectors.npy; incremental add_version: embed only missing emb_keys, close changed/vanished revisions, identity by key, git rename, or unchanged content moved), `search.py` (`search_rev`, `search_range` with best revision per unit (ties -> most recent), versions matched, history introduced/modified/removed; optional `--verify` Stage 2 re-rank), `cli.py` (index / search --rev / search --range / versions; CPU default).
- `tests/test_versioned.py`: 5 passed (tiny git repo v1-v3 with add/rename/modify/remove; JSONL and snapshot sources; windowing).
- Not yet done: `scripts/simulate_versions.py` and the Section 12.5 metrics / CPU latency (they need the CPU; scheduled after the test job).

## 2026-09-29 17:31 k=50 vs k=20 (validation) and PRE-DECLARATION of the final Stage 2 test run
Command: `python scripts/k50_vs_k20.py` -> `results/k50_vs_k20.json` (exec cache only; frozen weights w_pass 0.3, w_wrong 0.05, w_err 0.05, trivial_pass_factor 0.1, multi_answer_neutral true; 344 validation stdin queries, test-mix weighted).
- Weighted NDCG@10: k=20 0.9224, k=50 0.9444. Paired weighted bootstrap k=50 minus k=20: **+0.0220 [0.0008, 0.0503]**, excludes 0. MRR@10 +0.0219 [0.0008, 0.0502]. Unweighted NDCG@10 +0.0404 [0.0211, 0.0626]. 15 queries improve, 0 get worse.
- k=20 had been chosen for time, not merit (see midday entry). The CI excludes 0, so per P's rule the k=50 run is authorised.

**PRE-DECLARATION (written at 17:31, before Output B finishes and before either test score is seen; text given by P):**
"Final Stage 2 config = k=50 with the frozen weights. If the k=50 test run completes by 07:00 Wed, results/appsretrieval_results_B_k50.json is the submission regardless of its score relative to k=20. If it fails or doesn't finish, the k=20 Output B is the submission."
- Config file: `configs/stage2_final_k50.yaml` (identical to `configs/stage2_final.yaml` except k: 50).
- Command (starts automatically when the k=20 job's process exits; nothing else runs on the CPU meanwhile): `python -m codeintel.eval.run_two_stage --split test --reuse-a --stage2-config configs/stage2_final_k50.yaml --out-suffix _k50` (re-ranks the saved Output A predictions; no second Stage 1 test pass; sandbox suite runs first). Log `logs/test_two_stage_k50.log`.
- This is the last Stage 2 test run. Estimated 96,086 new executions (ranks 21-50) at 2.7-4 per s: 6.7-10 h.
- Format-match bonus: skipped (P's decision).
