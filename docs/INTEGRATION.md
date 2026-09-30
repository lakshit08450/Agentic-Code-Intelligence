# Integration guide: code search API (schema 1.0, frozen 30 Sep 20:00)

Audience: the app developer (and their AI assistant). This API wraps the benchmark pipeline for live use:
Stage 1 dense retrieval (fusion of Qwen3-Embedding-0.6B and EmbeddingGemma-300m, or Qwen3 only) and
optional Stage 2 execution re-ranking in the Windows sandbox. **Frozen**: after 20:00 on 30 Sep, fields may
only be added, never renamed or removed. Integration code lives in `src/codeintel/app/`; it never touches
the benchmark results or caches.

## 1. Quick start (fresh Windows 10/11 x64 machine)
| Step | Command | Verify |
|---|---|---|
| 1. Python 3.11 | `winget install Python.Python.3.11` | `py -3.11 --version` prints 3.11.x |
| 2. Clone | `git clone <repo-url> prism-code-intel` then `cd prism-code-intel` | `dir src\codeintel\app` lists `service.py` |
| 3. venv + packages | `py -3.11 -m venv .venv` then `.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` then `.venv\Scripts\python -m pip install -e .[dev]` | `.venv\Scripts\python -c "import codeintel, mteb, torch"` exits silently |
| 4. Hugging Face (Gemma is gated) | Accept the terms at https://huggingface.co/google/embeddinggemma-300m, then `.venv\Scripts\huggingface-cli login` (paste a read token) | `.venv\Scripts\huggingface-cli whoami` prints your user |
| 5. Demo data (index + embeddings, no encoding needed) | `.venv\Scripts\python scripts\fetch_demo_data.py <URL-or-path-of-prism_demo_data.zip>` | `demo_data\manifest.json` exists |
| 6. Sandbox interpreter | `powershell -ExecutionPolicy Bypass -File scripts\setup_sandbox.ps1` | prints `sandbox interpreter 3.11.9` and a firewall command |
| 7. Firewall rule (**admin PowerShell, once**) | `New-NetFirewallRule -DisplayName "PRISM APPS sandbox" -Direction Outbound -Action Block -Program "<repo>\tools\sandbox-python\python.exe"` (exact line printed by step 6) | `.venv\Scripts\python -m pytest -q tests\test_sandbox.py` passes |
| 8. Smoke test | `.venv\Scripts\python scripts\integration_smoke_test.py` | last line `PASS: n/n checks passed` |

Set `PYTHONUTF8=1` in the shell. The first model load downloads ~2.5 GB of weights into `%USERPROFILE%\.cache\huggingface`.
UI work can start before any of this with **stub mode** (Section 4), which needs only steps 1-3.

## 2. Interface
Python:
```python
from codeintel.app.service import SearchService
svc = SearchService()            # SearchService(stub=True) for canned results
svc.load()                       # loads both models + demo stores, warms up
resp = svc.search(query, version=None, k=10, verify=True, pipeline="fusion_stage2")
```
HTTP (standard library server, binds 127.0.0.1):
```powershell
.venv\Scripts\python -m codeintel.app.server --port 8765          # add --stub for canned results
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/search -H "Content-Type: application/json" -d "{\"query\": \"reverse a string\", \"k\": 5}"
```

### Parameters
| name | type | default | meaning |
|---|---|---|---|
| `query` | str | required | problem statement or natural-language query. If it contains sample Input/Output blocks, Stage 2 can run |
| `version` | str or null | null | null = APPS corpus (8,765 solutions); `"v2"` = one version of the demo git repo; `"v1..v4"` = all versions in a range (one result per file, with history) |
| `k` | int 1-100 | 10 | results returned |
| `verify` | bool | true | allow Stage 2 execution re-ranking (only for `*_stage2` pipelines) |
| `pipeline` | str | `fusion_stage2` | `fusion_stage2` (submission pipeline), `fusion`, `qwen3_stage2`, `qwen3` |
| `stub` | bool | false | canned results with the same schema |
| `require_stage2` | bool | false | return error `SANDBOX_UNAVAILABLE` instead of falling back when the sandbox is unavailable |

### Response schema (1.0)
| field | type | meaning |
|---|---|---|
| `schema_version` | str | "1.0" |
| `stub` | bool | true for canned results |
| `query`, `pipeline`, `version`, `k` | | echo of the request |
| `corpus` | str | `apps` or `repo` |
| `stage2.ran` | bool | true if Stage 2 re-ranked the results |
| `stage2.fallback_reason` | str or null | why Stage 2 did not run: `verify_false`, `pipeline_without_stage2`, `no_sample_io`, `sandbox_unavailable` (results are then Stage 1 / fusion-only) |
| `stage2.k_verified` | int | candidates executed (20) |
| `results[]` | list | ranked hits |
| `results[].rank` | int | 1-based |
| `results[].doc_id` | str | APPS document id (e.g. `d4512`) or repo file path |
| `results[].path` | str | file path (APPS: same as doc_id) |
| `results[].lines` | [int, int] | first and last line of the snippet (1-based, inclusive) |
| `results[].snippet` | str | code text (APPS: whole program) |
| `results[].score` | float | final score (cosine + Stage 2 adjustment) |
| `results[].stage1_score` | float | cosine similarity |
| `results[].stage2_outcome` | str or null | `PASS`, `WRONG`, `ERROR`, `TIMEOUT`, `UNKNOWN`; null if not executed |
| `results[].version` | object | `label` (version searched / best revision), `from`, `to` (null = still present) |
| `results[].history` | list | range queries only: `introduced` / `modified` / `removed` entries |
| `results[].versions_matched` | list | range queries only |
| `timings_ms` | object | `encode`, `search`, `verify`, `total` |
| `warnings` | list[str] | e.g. `SANDBOX_UNAVAILABLE: ...`, `STUB_MODE: ...` |
| `error` | object or null | `{code, http_status, message}` on failure; `results` is then empty |

### Errors
| code | HTTP | when | what to do |
|---|---|---|---|
| `BAD_REQUEST` | 400 | empty query, bad `k`, unknown pipeline, invalid JSON | fix the request |
| `BAD_VERSION` | 400 | unknown version or range | use `/health` -> `repo_versions` |
| `MODELS_LOADING` | 503 | request before warm-up finished | poll `/health` until `ready: true` |
| `SANDBOX_UNAVAILABLE` | 503 | only with `require_stage2=true` | otherwise Stage 2 falls back with a warning |
| `TIMEOUT` | 504 | request took longer than `--timeout` (180 s) | retry; searches are serialized |
| `INTERNAL` | 500 | unexpected exception (message included) | report it |

### Example response
[filled from a real call after the demo data is built]

## 3. Behaviour
- **Concurrency**: the server accepts concurrent connections, but searches run one at a time (a lock); a
  request waits at most `--timeout` seconds (default 180) and then gets `TIMEOUT` (the running search
  finishes in the background). One UI user at a time is the intended use.
- **Without the sandbox** (no `tools\sandbox-python`, no firewall rule, or the self-test fails): Stage 2 is
  skipped, results are fusion-only, `stage2.fallback_reason = "sandbox_unavailable"` and a warning explains
  why. `/health` -> `sandbox.available` shows this at start-up.
- **Stage 2 depth**: the app executes the top 20 candidates per query to keep latency interactive. The
  benchmark submission executes 50 (fusion + Stage 2 k=50, test NDCG@10 0.956); the same configuration at
  k=20 scored 0.952 on test (reported ablation). Executions are cached in `app_runtime\exec.sqlite`.
- **Latency on CPU, RAM, load time**: [measured values filled in after the demo data is built].

## 4. Stub mode
`SearchService(stub=True)` or `python -m codeintel.app.server --stub`: realistic canned results with the
same schema, instantly, no models, no sandbox, no demo data. `stub: true` and a `STUB_MODE` warning mark them.

## 5. Troubleshooting
| symptom | cause | fix |
|---|---|---|
| `/health` status `error`, "manifest.json missing" | demo data not installed | step 5 |
| 401 / "gated repo" when loading | Gemma terms not accepted or not logged in | step 4 |
| `stage2.fallback_reason: sandbox_unavailable` | sandbox not set up or firewall rule missing | steps 6-7, then restart the server |
| every Stage 2 outcome `TIMEOUT` | machine heavily loaded | close other CPU-heavy programs; timeouts are re-run once |
| new console windows pop up | old code or another launcher | use the commands above (no `start`) |
| `MODELS_LOADING` for minutes | first run downloads weights | wait; `/health` shows `loading` |
| port 8765 in use | another server running | `--port 8766` |
