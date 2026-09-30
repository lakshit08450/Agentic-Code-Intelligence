"""
PRISM client: calls the PRISM code-intel server (its own Python 3.11 venv, see ../../README.md) over
its frozen HTTP API (docs/INTEGRATION.md in the main repo) and maps the responses into CodeLens types.

Standard library only (urllib), so the two venvs never share dependencies.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from app.storage.schemas import AgentTraceStep, QueryResponse, SearchResult

PRISM_URL = os.environ.get("PRISM_API_URL", "http://127.0.0.1:8765").rstrip("/")
TIMEOUT_S = float(os.environ.get("PRISM_TIMEOUT_S", "180"))

# CodeLens "version" values that select PRISM datasets instead of a locally indexed repo
APPS_DATASET = "apps"


class PrismError(RuntimeError):
    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.http_status = code, message, http_status


def _request(method: str, path: str, body: dict | None = None, timeout: float = TIMEOUT_S) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{PRISM_URL}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # PRISM returns JSON bodies with an error object
        try:
            payload = json.loads(e.read().decode("utf-8"))
            err = payload.get("error") or {}
            raise PrismError(err.get("code", "HTTP_ERROR"), err.get("message", str(e)), e.code) from None
        except (ValueError, AttributeError):
            raise PrismError("HTTP_ERROR", str(e), e.code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise PrismError("PRISM_UNREACHABLE", f"cannot reach PRISM server at {PRISM_URL}: {e}", 503) from None


def health() -> dict:
    """PRISM /health, or a synthetic 'unreachable' status (never raises)."""
    try:
        return _request("GET", "/health", timeout=5)
    except PrismError as e:
        return {"status": "unreachable", "ready": False, "error": e.message, "sandbox": {"available": False},
                "repo_versions": []}


def is_prism_version(version: str | None) -> bool:
    """True if a CodeLens 'version' selects a PRISM dataset: 'apps', 'v1'..'v4' or a range 'v1..v4'."""
    if not version:
        return False
    if version == APPS_DATASET:
        return True
    parts = version.split("..")
    return all(p[:1] == "v" and p[1:].isdigit() for p in parts) and 1 <= len(parts) <= 2


def search(query: str, version: str, k: int = 10) -> dict:
    """Raw PRISM search response (schema 1.0). version 'apps' -> the APPS corpus."""
    return _request("POST", "/search", {
        "query": query, "version": None if version == APPS_DATASET else version, "k": k,
        "verify": True, "pipeline": "fusion_stage2",
    })


def embed(texts: list[str], kind: str) -> np.ndarray:
    """L2-normalised fusion embeddings from PRISM POST /embed (kind: 'query' or 'document')."""
    resp = _request("POST", "/embed", {"texts": texts, "type": kind})
    if resp.get("error"):
        err = resp["error"]
        raise PrismError(err.get("code", "EMBED_ERROR"), err.get("message", ""))
    return np.asarray(resp["vectors"], dtype=np.float32)


def to_query_response(query_id: str, query: str, resp: dict) -> QueryResponse:
    """Map a PRISM response into CodeLens' QueryResponse; PRISM-specific fields go into `context`."""
    stage2 = resp.get("stage2", {})
    timings = resp.get("timings_ms", {})
    results = []
    for r in resp.get("results", []):
        results.append(SearchResult(
            rank=r["rank"], score=float(r["score"]), file=r["path"],
            start_line=int(r["lines"][0]), end_line=int(r["lines"][1]), snippet=r["snippet"],
            language="python", type="stage2" if stage2.get("ran") else "semantic",
            version=str((r.get("version") or {}).get("label") or ""),
            context={
                "source": "prism",
                "doc_id": r.get("doc_id"),
                "stage1_score": r.get("stage1_score"),
                "stage2_outcome": r.get("stage2_outcome"),
                "stage2_ran": stage2.get("ran", False),
                "fallback_reason": stage2.get("fallback_reason"),
                "version_from": (r.get("version") or {}).get("from"),
                "version_to": (r.get("version") or {}).get("to"),
                "history": r.get("history"),
                "timings_ms": timings,
                "warnings": resp.get("warnings", []),
            },
        ))
    trace = [
        AgentTraceStep(phase="plan", detail=f"PRISM pipeline {resp.get('pipeline')} on {resp.get('corpus')}"
                                           f"{' @ ' + resp['version'] if resp.get('version') else ''}",
                       duration_ms=0.0),
        AgentTraceStep(phase="search", detail="Stage 1: fused Qwen3-Embedding-0.6B + EmbeddingGemma-300m cosine search",
                       duration_ms=float(timings.get("encode", 0)) + float(timings.get("search", 0)),
                       candidates_count=len(results)),
        AgentTraceStep(phase="refine",
                       detail=(f"Stage 2: executed top {stage2.get('k_verified', 0)} candidates on the statement's sample I/O"
                               if stage2.get("ran") else f"Stage 2 not run: {stage2.get('fallback_reason')}"),
                       duration_ms=float(timings.get("verify", 0))),
    ]
    return QueryResponse(query_id=query_id, query=query, results=results, trace=trace,
                         total_ms=float(timings.get("total", 0.0)))
