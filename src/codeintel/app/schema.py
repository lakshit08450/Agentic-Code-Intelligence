"""Frozen response schema for the integration API (docs/INTEGRATION.md).

SCHEMA_VERSION 1.0 is frozen as of 30 Sep 20:00: fields may be ADDED later, never renamed or removed.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"

PIPELINES = ("fusion_stage2", "fusion", "qwen3_stage2", "qwen3")
OUTCOMES = ("PASS", "WRONG", "ERROR", "TIMEOUT", "UNKNOWN", None)
FALLBACK_REASONS = (None, "verify_false", "pipeline_without_stage2", "no_sample_io", "sandbox_unavailable")
ERROR_CODES = {
    "BAD_REQUEST": 400,        # missing/invalid fields
    "BAD_VERSION": 400,        # unknown version / range
    "MODELS_LOADING": 503,     # models not ready yet (retry after /health says ready)
    "SANDBOX_UNAVAILABLE": 503,  # only if a caller REQUIRES Stage 2 (require_stage2=true); otherwise a warning
    "TIMEOUT": 504,            # request exceeded the server timeout
    "INTERNAL": 500,
}

# field -> allowed python types (None allowed where listed)
RESULT_FIELDS: dict[str, tuple] = {
    "rank": (int,),
    "doc_id": (str,),
    "path": (str,),
    "lines": (list,),
    "snippet": (str,),
    "score": (float,),
    "stage1_score": (float,),
    "stage2_outcome": (str, type(None)),
    "version": (dict,),
}
RESPONSE_FIELDS: dict[str, tuple] = {
    "schema_version": (str,),
    "stub": (bool,),
    "query": (str,),
    "pipeline": (str,),
    "corpus": (str,),
    "version": (str, type(None)),
    "k": (int,),
    "stage2": (dict,),
    "results": (list,),
    "timings_ms": (dict,),
    "warnings": (list,),
    "error": (dict, type(None)),
}
STAGE2_FIELDS = {"ran": (bool,), "fallback_reason": (str, type(None)), "k_verified": (int,)}
TIMING_FIELDS = ("encode", "search", "verify", "total")
VERSION_FIELDS = ("label", "from", "to")


def error_response(code: str, message: str, **ctx: Any) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "stub": False, "query": ctx.get("query", ""), "pipeline": ctx.get("pipeline", ""),
        "corpus": ctx.get("corpus", ""), "version": ctx.get("version"), "k": int(ctx.get("k", 0) or 0),
        "stage2": {"ran": False, "fallback_reason": None, "k_verified": 0}, "results": [],
        "timings_ms": {t: 0.0 for t in TIMING_FIELDS}, "warnings": [],
        "error": {"code": code, "http_status": ERROR_CODES[code], "message": message},
    }


def validate(resp: dict) -> list[str]:
    """Returns a list of schema violations (empty = valid)."""
    errs = []
    for f, types in RESPONSE_FIELDS.items():
        if f not in resp:
            errs.append(f"missing field {f}")
        elif not isinstance(resp[f], types):
            errs.append(f"{f}: {type(resp[f]).__name__} not in {[t.__name__ for t in types]}")
    for f, types in STAGE2_FIELDS.items():
        if f not in resp.get("stage2", {}) or not isinstance(resp["stage2"][f], types):
            errs.append(f"stage2.{f} missing or wrong type")
    if resp.get("stage2", {}).get("fallback_reason") not in FALLBACK_REASONS:
        errs.append("stage2.fallback_reason not an allowed value")
    for t in TIMING_FIELDS:
        if not isinstance(resp.get("timings_ms", {}).get(t), (int, float)):
            errs.append(f"timings_ms.{t} missing")
    for i, r in enumerate(resp.get("results", [])):
        for f, types in RESULT_FIELDS.items():
            if f not in r or not isinstance(r[f], types):
                errs.append(f"results[{i}].{f} missing or wrong type")
        if r.get("stage2_outcome") not in OUTCOMES:
            errs.append(f"results[{i}].stage2_outcome not allowed")
        if not all(v in r.get("version", {}) for v in VERSION_FIELDS):
            errs.append(f"results[{i}].version missing label/from/to")
        if len(r.get("lines", [])) != 2:
            errs.append(f"results[{i}].lines must be [start, end]")
    return errs
