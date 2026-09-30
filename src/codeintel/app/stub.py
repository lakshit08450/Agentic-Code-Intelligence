"""Stub mode: realistic canned results with the frozen schema, returned instantly (no models, no
sandbox, no demo data). For building the UI before models load or on a machine without the setup."""

from __future__ import annotations

from codeintel.app.schema import SCHEMA_VERSION

_APPS = [
    ("d4512", "n, k = map(int, input().split())\na = sorted(map(int, input().split()))\nprint(sum(a[:k]))\n", "PASS"),
    ("d1377", "n = int(input())\nxs = list(map(int, input().split()))\nprint(max(xs) - min(xs))\n", "WRONG"),
    ("d8021", "import sys\ndata = sys.stdin.read().split()\nprint(len(set(data[1:])))\n", "WRONG"),
    ("d0299", "def solve(a):\n    return sorted(a)[0]\n", "UNKNOWN"),
    ("d6630", "n = int(input())\nprint(n // 0)\n", "ERROR"),
]
_REPO = [
    ("src/codeintel/stage2/compare.py", [20, 44], "def tokens_match(a: str, b: str, tol: float = 1e-6) -> bool:\n    ..."),
    ("src/codeintel/stage2/sampleio.py", [150, 190], "def parse(statement: str) -> Samples:\n    ..."),
    ("src/codeintel/eval/metrics.py", [1, 40], "def per_query(run, qrels, k_values=(1, 10, 20, 50, 100)):\n    ..."),
]


def stub_response(query: str, version: str | None, k: int, verify: bool, pipeline: str) -> dict:
    stage2 = pipeline.endswith("_stage2") and verify
    has_io = "Input" in query and "Output" in query
    results = []
    if version is None:
        for i, (doc, code, outcome) in enumerate(_APPS[:k], 1):
            s1 = round(0.72 - 0.03 * i, 4)
            ran = stage2 and has_io
            results.append({
                "rank": i, "doc_id": doc, "path": doc, "lines": [1, code.count("\n")], "snippet": code,
                "score": round(s1 + (0.3 if ran and outcome == "PASS" else 0.0), 4), "stage1_score": s1,
                "stage2_outcome": outcome if ran else None, "version": {"label": "apps", "from": "apps", "to": None},
            })
    else:
        label = version.split("..")[-1]
        for i, (path, lines, snip) in enumerate(_REPO[:k], 1):
            results.append({
                "rank": i, "doc_id": path, "path": path, "lines": lines, "snippet": snip, "score": round(0.61 - 0.04 * i, 4),
                "stage1_score": round(0.61 - 0.04 * i, 4), "stage2_outcome": None,
                "version": {"label": label, "from": "v1", "to": None},
            })
    ran = stage2 and version is None and has_io
    fallback = None if ran else ("pipeline_without_stage2" if not pipeline.endswith("_stage2")
                                 else "verify_false" if not verify else "no_sample_io")
    return {
        "schema_version": SCHEMA_VERSION, "stub": True, "query": query, "pipeline": pipeline,
        "corpus": "apps" if version is None else "repo", "version": version, "k": k,
        "stage2": {"ran": ran, "fallback_reason": fallback, "k_verified": 20 if ran else 0},
        "results": sorted(results, key=lambda r: -r["score"])[:k] if ran else results,
        "timings_ms": {"encode": 1.0, "search": 1.0, "verify": 2.0 if ran else 0.0, "total": 4.0},
        "warnings": ["STUB_MODE: canned results"], "error": None,
    }
