from __future__ import annotations

from pathlib import Path

from codeintel.common.config import load_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load():
    for p in (ROOT / "configs").rglob("*.yaml"):
        cfg = load_cfg(p)
        assert cfg.device == "cpu", f"{p.name}: submitted configs must default to CPU (R5)"
        assert cfg.models
