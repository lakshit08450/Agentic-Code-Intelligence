from __future__ import annotations

from pathlib import Path

import yaml

from codeintel.common.config import load_cfg
from codeintel.stage2.reranker import load_stage2_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load():
    for p in (ROOT / "configs").rglob("*.yaml"):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if "models" in raw:  # Stage 1 config
            cfg = load_cfg(p)
            assert cfg.device == "cpu", f"{p.name}: submitted configs must default to CPU (R5)"
            assert cfg.models
        else:  # Stage 2 config
            assert load_stage2_cfg(p).k > 0
