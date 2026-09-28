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
