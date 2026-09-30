"""Integration service: the frozen `search` interface over pre-built demo stores (docs/INTEGRATION.md).

    from codeintel.app.service import SearchService
    svc = SearchService(demo_dir="demo_data")      # or SearchService(stub=True)
    svc.load()                                     # loads models + stores, warms up (1-2 min on CPU)
    resp = svc.search("statement text ...", version=None, k=10, verify=True, pipeline="fusion_stage2")

Corpora: version=None searches the APPS corpus (8,765 solutions); version="<label>" searches one
version of the demo git repo; version="<A>..<B>" searches all versions in a range (one result per
file, with history). Calls are serialized (one search at a time). Kept separate from the benchmark
code paths: it reads only demo_data/ and writes only app_runtime/.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codeintel.app.schema import ERROR_CODES, PIPELINES, SCHEMA_VERSION, error_response
from codeintel.common.config import REPO_ROOT

DEFAULT_DEMO_DIR = REPO_ROOT / "demo_data"
RUNTIME_DIR = REPO_ROOT / "app_runtime"
VERIFY_K = 20  # candidates executed per query in the app (the benchmark submission used 50)


class ApiError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class SearchService:
    def __init__(self, demo_dir: str | Path = DEFAULT_DEMO_DIR, stub: bool = False, verify_k: int = VERIFY_K) -> None:
        self.demo_dir = Path(demo_dir)
        self.stub = stub
        self.verify_k = verify_k
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.load_error: str | None = None
        self.load_time_s: float | None = None
        self.sandbox = {"available": False, "reason": "not checked"}
        self._lock = threading.Lock()
        self.stores: dict[tuple[str, str], Any] = {}
        self.manifest: dict = {}

    # ---------------- loading ----------------
    def load(self) -> None:
        if self.stub:
            self.state, self.load_time_s = "ready", 0.0
            self.sandbox = {"available": False, "reason": "stub mode"}
            return
        self.state = "loading"
        t0 = time.perf_counter()
        try:
            self._load_models_and_stores()
            self.sandbox = self._check_sandbox()
            for pipeline in ("fusion", "qwen3"):  # warm-up: first encode is slow
                self.stores[(pipeline, "apps")].encoder.embed(["warm up"], is_query=True)
            self.load_time_s = round(time.perf_counter() - t0, 1)
            self.state = "ready"
        except Exception as e:  # noqa: BLE001 - reported through /health
            self.state, self.load_error = "error", f"{type(e).__name__}: {e}"
            raise

    def _load_models_and_stores(self) -> None:
        import copy

        from codeintel.common.config import load_cfg
        from codeintel.stage1.encoder import PrePostPipelineEncoder, SentenceTransformerBackend
        from codeintel.versioned.store import Store

        mf = self.demo_dir / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(f"{mf} missing: download/extract the demo data first (docs/INTEGRATION.md)")
        self.manifest = json.loads(mf.read_text(encoding="utf-8"))
        cfgs = {p: copy.deepcopy(load_cfg(REPO_ROOT / c)) for p, c in self.manifest["configs"].items()}
        backends = {}
        for cfg in cfgs.values():
            for m in cfg.models:
                if m.id not in backends:
                    backends[m.id] = SentenceTransformerBackend(m, "cpu", cfg.batch_size, fp16_on_cuda=False, attn_budget=cfg.attn_budget)
        RUNTIME_DIR.mkdir(exist_ok=True)
        for pipeline, cfg in cfgs.items():
            cfg.device, cfg.cache_dir = "cpu", str(RUNTIME_DIR / "query_cache")
            enc = PrePostPipelineEncoder(cfg, backends=[backends[m.id] for m in cfg.models])
            for corpus in ("apps", "repo"):
                self.stores[(pipeline, corpus)] = Store(self.demo_dir / f"{corpus}_{pipeline}", enc)

    def _check_sandbox(self) -> dict:
        from codeintel.common import proc
        from codeintel.stage2.sandbox import PY, run_program

        if not PY.exists():
            return {"available": False, "reason": "tools/sandbox-python missing (run scripts/setup_sandbox.ps1)"}
        fw = proc.run(["powershell", "-NoProfile", "-Command",
                       "Get-NetFirewallRule -DisplayName 'PRISM APPS sandbox' -ErrorAction Stop | Out-Null"],
                      capture_output=True)
        if fw.returncode != 0:
            return {"available": False, "reason": "firewall rule 'PRISM APPS sandbox' missing (admin command in docs)"}
        r = run_program("print(int(input()) * 2)\n", "21\n")
        if (r.status, r.stdout.strip()) != ("OK", "42"):
            return {"available": False, "reason": f"sandbox self-test failed: {r.status} {r.reason}"}
        return {"available": True, "reason": "ok"}

    def health(self) -> dict:
        versions = []
        if ("fusion", "repo") in self.stores:
            versions = [label for _, label, _ in self.stores[("fusion", "repo")].versions()]
        return {
            "schema_version": SCHEMA_VERSION, "status": self.state, "ready": self.state == "ready", "stub": self.stub,
            "load_time_s": self.load_time_s, "error": self.load_error, "sandbox": self.sandbox,
            "pipelines": list(PIPELINES), "repo_versions": versions if not self.stub else ["v1", "v2", "v3", "v4"],
            "verify_k": self.verify_k,
        }

    # ---------------- search ----------------
    def search(self, query: str, version: str | None = None, k: int = 10, verify: bool = True,
               pipeline: str = "fusion_stage2", stub: bool | None = None, require_stage2: bool = False) -> dict:
        ctx = {"query": query, "pipeline": pipeline, "version": version, "k": k,
               "corpus": "apps" if version is None else "repo"}
        try:
            if not isinstance(query, str) or not query.strip():
                raise ApiError("BAD_REQUEST", "query must be a non-empty string")
            if pipeline not in PIPELINES:
                raise ApiError("BAD_REQUEST", f"pipeline must be one of {PIPELINES}")
            if not isinstance(k, int) or not 1 <= k <= 100:
                raise ApiError("BAD_REQUEST", "k must be an integer in 1..100")
            if stub or (stub is None and self.stub):
                from codeintel.app.stub import stub_response

                return stub_response(query, version, k, verify, pipeline)
            if self.state != "ready":
                raise ApiError("MODELS_LOADING", f"models are {self.state}; poll GET /health until ready")
            with self._lock:
                return self._search(query, version, k, verify, pipeline, require_stage2, ctx)
        except ApiError as e:
            return error_response(e.code, e.message, **ctx)
        except Exception as e:  # noqa: BLE001
            return error_response("INTERNAL", f"{type(e).__name__}: {e}", **ctx)

    def _search(self, query, version, k, verify, pipeline, require_stage2, ctx) -> dict:
        from codeintel.versioned.search import search_range, search_rev

        t0 = time.perf_counter()
        base = pipeline.replace("_stage2", "")
        corpus = ctx["corpus"]
        store = self.stores[(base, corpus)]
        want_stage2 = pipeline.endswith("_stage2") and verify
        depth = max(k, self.verify_k) if want_stage2 else k
        labels = {o: l for o, l, _ in store.versions()}
        try:
            if corpus == "apps":
                raw = search_rev(store, query, store.n_versions() - 1, depth)
            elif ".." in version:
                a, b = version.split("..", 1)
                raw = search_range(store, query, a, b, depth)
            else:
                raw = search_rev(store, query, version, depth)
        except KeyError as e:
            raise ApiError("BAD_VERSION", f"unknown version {version!r}; available: {list(labels.values())}") from e
        keys = dict(store.db.execute("SELECT unit_id, key FROM units"))
        results = []
        for r in raw["results"]:
            if corpus == "apps":
                vinfo = {"label": "apps", "from": "apps", "to": None}
            elif "best_revision" in r:  # range search: labels already; to=None means still alive
                b = r["best_revision"]
                vinfo = {"label": b["to"] if b["to"] is not None else raw["range"][1], "from": b["from"], "to": b["to"]}
            else:  # single version: ordinals -> labels
                vf, vt = r["version"]["from"], r["version"]["to"]
                vinfo = {"label": raw["rev"], "from": labels.get(vf, vf), "to": labels.get(vt) if vt is not None else None}
            res = {
                "rank": r["rank"], "doc_id": keys.get(r["unit_id"], str(r["unit_id"])), "path": r["path"],
                "lines": list(r["lines"]), "snippet": r["snippet"], "score": float(r["score"]),
                "stage1_score": float(r["score"]), "stage2_outcome": None, "version": vinfo,
            }
            for extra in ("history", "versions_matched"):
                if extra in r:
                    res[extra] = r[extra]
            results.append(res)
        warnings, fallback, verified, t_ver = [], None, 0, 0.0
        if not want_stage2:
            fallback = "verify_false" if pipeline.endswith("_stage2") else "pipeline_without_stage2"
        else:
            from codeintel.stage2.verifier import samples_for

            if not self.sandbox["available"]:
                fallback = "sandbox_unavailable"
                warnings.append("SANDBOX_UNAVAILABLE: " + self.sandbox["reason"])
                if require_stage2:
                    raise ApiError("SANDBOX_UNAVAILABLE", self.sandbox["reason"])
            elif samples_for(query).kind != "stdin":
                fallback = "no_sample_io"
            else:
                t_ver, verified = self._stage2(query, results)
        results = results[:k]
        for i, r in enumerate(results, 1):
            r["rank"] = i
        total = 1000 * (time.perf_counter() - t0)
        return {
            "schema_version": SCHEMA_VERSION, "stub": False, "query": query, "pipeline": pipeline, "corpus": corpus,
            "version": version, "k": k,
            "stage2": {"ran": fallback is None, "fallback_reason": fallback, "k_verified": verified},
            "results": results,
            "timings_ms": {"encode": round(raw["ms"]["encode"], 1), "search": round(raw["ms"]["search"], 1),
                           "verify": round(t_ver, 1), "total": round(total, 1)},
            "warnings": warnings, "error": None,
        }

    def _stage2(self, query: str, results: list[dict]) -> tuple[float, int]:
        from codeintel.stage2.reranker import load_stage2_cfg, stage2_score
        from codeintel.stage2.sandbox import ExecCache
        from codeintel.stage2.verifier import verify

        t0 = time.perf_counter()
        cfg = load_stage2_cfg("configs/stage2_final_k50_fusion.yaml")
        cache = ExecCache(RUNTIME_DIR / "exec.sqlite")
        cands = results[: self.verify_k]
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            verdicts = list(ex.map(lambda r: verify(query, r["snippet"], cache), cands))
        for r, v in zip(cands, verdicts):
            r["stage2_outcome"] = v.outcome
            r["score"] = float(stage2_score(r["stage1_score"], v.outcome, query, cfg))
        results.sort(key=lambda r: -r["score"])
        return 1000 * (time.perf_counter() - t0), len(cands)


def http_status(resp: dict) -> int:
    return 200 if resp.get("error") is None else ERROR_CODES[resp["error"]["code"]]
