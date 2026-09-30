"""Build the integration demo data (demo_data/): per pipeline (fusion, qwen3) an APPS store (8,765
solutions, one version "apps", no windowing so Stage 2 runs whole programs) and a git demo store
(this repo's .py files at 4 commits, labels v1..v4). Document vectors come from the CPU fp32 cache
(cache/embeddings_cpu_fp32), so the app's CPU query vectors match the index. Then export one zip.

    .venv\\Scripts\\python scripts\\build_demo_data.py            # build demo_data/ and dist/prism_demo_data.zip
"""

from __future__ import annotations

import copy
import json
import shutil
import time
import zipfile
from pathlib import Path

from codeintel.common.config import REPO_ROOT, load_cfg
from codeintel.eval.devset import load_apps_texts
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.versioned import store as store_mod
from codeintel.versioned.ingest import Version, git_versions
from codeintel.versioned.units import split

OUT = REPO_ROOT / "demo_data"
DIST = REPO_ROOT / "dist" / "prism_demo_data.zip"
CONFIGS = {"fusion": "configs/fusion/qwen3_gemma_w0.3.yaml", "qwen3": "configs/stage1_final.yaml"}
# four commits of this repo spanning Path B -> fusion (first-parent history), relabelled v1..v4
REPO_REVS = ["a59bd00", "d85063b", "981a1ed", "HEAD"]


def encoder(cfg_path: str) -> PrePostPipelineEncoder:
    cfg = copy.deepcopy(load_cfg(cfg_path))
    cfg.cache_dir, cfg.device = "cache/embeddings_cpu_fp32", "cpu"
    return PrePostPipelineEncoder(cfg)


def main() -> None:
    t0 = time.perf_counter()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    ids, texts, _ = load_apps_texts()
    apps = Version("apps", "CoIR-Retrieval/apps@f22508f9 corpus", {d: (d, t) for d, t in zip(ids, texts)})
    repo = git_versions(REPO_ROOT, REPO_REVS)
    for i, v in enumerate(repo, 1):
        v.label, v.source_ref = f"v{i}", f"{v.label} {v.source_ref}"
    stats = {}
    for name, cfg_path in CONFIGS.items():
        enc = encoder(cfg_path)
        orig = store_mod.split
        store_mod.split = lambda k, p, t: split(k, p, t, max_lines=10**9)  # whole programs for APPS
        try:
            s = store_mod.Store(OUT / f"apps_{name}", enc)
            stats[f"apps_{name}"] = s.add_version(apps)
            s.db.close()
        finally:
            store_mod.split = orig
        s = store_mod.Store(OUT / f"repo_{name}", enc)
        stats[f"repo_{name}"] = [s.add_version(v) for v in repo]
        s.db.close()
    manifest = {
        "schema_version": "1.0", "built": time.strftime("%Y-%m-%d %H:%M"), "configs": CONFIGS,
        "repo_versions": {v.label: v.source_ref for v in repo}, "apps_docs": len(ids),
        "embeddings": "CPU fp32 (Qwen3-Embedding-0.6B rev 97b0c614, EmbeddingGemma-300m rev 57c266a7)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    DIST.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(DIST, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(REPO_ROOT).as_posix())
    print(json.dumps({"built_s": round(time.perf_counter() - t0, 1), "zip": str(DIST),
                      "zip_mb": round(DIST.stat().st_size / 2**20, 1),
                      "embedded": {k: (v["embedded"] if isinstance(v, dict) else [x["embedded"] for x in v]) for k, v in stats.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
