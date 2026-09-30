"""Integration smoke test (docs/INTEGRATION.md). Checks the setup, loads the models, runs three queries,
validates the frozen JSON schema and prints timings with PASS/FAIL. Exit code 0 only if all checks pass.

    .venv\\Scripts\\python scripts\\integration_smoke_test.py            # full test (models + sandbox)
    .venv\\Scripts\\python scripts\\integration_smoke_test.py --stub     # schema/stub only, no models
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

from codeintel.app.schema import validate
from codeintel.app.service import SearchService

Q_WITH_IO = """You are given two integers a and b. Print their sum.

-----Input-----
The only line contains two integers a and b (0 <= a, b <= 1000).

-----Output-----
Print a + b.

-----Examples-----
Input
2 3

Output
5

Input
10 20

Output
30
"""
Q_NO_IO = "Given a string, reverse it and print the reversed string."
Q_REPO = "parse sample input and output pairs from a problem statement"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)
    return ok


def run_query(svc: SearchService, label: str, **kw) -> dict:
    t0 = time.perf_counter()
    r = svc.search(**kw)
    wall = 1000 * (time.perf_counter() - t0)
    errs = validate(r)
    check(f"{label}: schema valid", not errs, "; ".join(errs[:3]))
    check(f"{label}: no error", r["error"] is None, str(r["error"]))
    top = r["results"][0] if r["results"] else {}
    print(f"       stage2={r['stage2']}  timings_ms={r['timings_ms']}  wall_ms={wall:.0f}")
    if top:
        print(f"       top1: {top['doc_id']} {top['path']} lines={top['lines']} score={top['score']:.4f} "
              f"stage2={top['stage2_outcome']} version={top['version']}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    check("python 3.11", sys.version_info[:2] == (3, 11), platform.python_version())

    stub = SearchService(stub=True)
    stub.load()
    for label, kw in (("stub apps", {"query": Q_WITH_IO}), ("stub repo", {"query": Q_REPO, "version": "v2"})):
        r = stub.search(**kw)
        check(f"{label}: schema valid", not validate(r), "; ".join(validate(r)[:3]))
    if a.stub:
        return finish()

    from pathlib import Path

    from codeintel.app.service import DEFAULT_DEMO_DIR
    from codeintel.stage2.sandbox import PY

    check("demo data present", (Path(DEFAULT_DEMO_DIR) / "manifest.json").exists(), str(DEFAULT_DEMO_DIR))
    check("sandbox interpreter present", PY.exists(), str(PY))
    svc = SearchService()
    t0 = time.perf_counter()
    try:
        svc.load()
        check("models + stores loaded", True, f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        check("models + stores loaded", False, f"{type(e).__name__}: {e}")
        return finish()
    check("sandbox available", svc.sandbox["available"], svc.sandbox["reason"])
    h = svc.health()
    check("health ready", h["ready"], str({k: h[k] for k in ("status", "load_time_s", "repo_versions")}))

    r1 = run_query(svc, "Q1 with sample I/O (Stage 2)", query=Q_WITH_IO, k=5)
    check("Q1: Stage 2 ran", r1["stage2"]["ran"], str(r1["stage2"]))
    r2 = run_query(svc, "Q2 no sample I/O (fallback)", query=Q_NO_IO, k=5)
    check("Q2: fallback no_sample_io", r2["stage2"]["fallback_reason"] == "no_sample_io", str(r2["stage2"]))
    versions = h["repo_versions"]
    r3 = run_query(svc, f"Q3 repo version {versions[1] if len(versions) > 1 else '?'}", query=Q_REPO, version=versions[1], k=5)
    check("Q3: results from that version", all(x["version"]["label"] == versions[1] for x in r3["results"]), "")
    r4 = run_query(svc, f"Q4 repo range {versions[0]}..{versions[-1]}", query=Q_REPO, version=f"{versions[0]}..{versions[-1]}", k=5)
    check("Q4: range has history", all("history" in x for x in r4["results"]), "")
    bad = svc.search(query=Q_REPO, version="nope")
    check("bad version -> BAD_VERSION", (bad["error"] or {}).get("code") == "BAD_VERSION", str(bad["error"]))
    return finish()


def finish() -> None:
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'PASS' if n_fail == 0 else 'FAIL'}: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
