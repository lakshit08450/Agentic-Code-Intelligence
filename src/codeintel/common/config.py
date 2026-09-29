"""YAML configs for Stage 1. Every preprocessing/scoring option is a flag here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from codeintel.common.hashing import fingerprint

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ModelCfg:
    id: str
    revision: str | None = None
    query_prompt: str = ""
    doc_prompt: str = ""
    max_seq_length: int = 512
    trust_remote_code: bool = False
    cuda_dtype: str = "float16"  # GPU encode precision (float16 | bfloat16 | float32); not part of the fingerprint

    def fingerprint(self) -> str:
        """Everything that changes the vector for a given preprocessed text.

        Device and dtype are deliberately excluded: the GPU fills the cache and the CPU
        run reads it; the CPU equivalence check (plan Section 6) guards that choice.
        """
        return fingerprint({k: v for k, v in asdict(self).items() if k not in ("trust_remote_code", "cuda_dtype")})


@dataclass
class QueryPreCfg:
    normalize_ws: bool = False
    strip_latex_dollar: bool = False


@dataclass
class DocPreCfg:
    normalize_ws: bool = False


@dataclass
class Stage1Cfg:
    name: str
    models: list[ModelCfg]
    fusion_weights: list[float] = field(default_factory=lambda: [1.0])
    device: str = "cpu"
    batch_size: int = 16
    fp16_on_cuda: bool = True
    attn_budget: int = 8192**2  # max batch_size * seq_len^2 per encode batch (GPU memory; no effect on vectors)
    query_pre: QueryPreCfg = field(default_factory=QueryPreCfg)
    doc_pre: DocPreCfg = field(default_factory=DocPreCfg)
    cache_dir: str = "cache/embeddings"
    on_cache_miss: str = "compute"  # "compute" | "error"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if len(self.models) != len(self.fusion_weights):
            raise ValueError("models and fusion_weights must have the same length")
        if abs(sum(self.fusion_weights) - 1.0) > 1e-6:
            raise ValueError(f"fusion_weights must sum to 1, got {self.fusion_weights}")
        if self.on_cache_miss not in {"compute", "error"}:
            raise ValueError(f"on_cache_miss must be compute|error, got {self.on_cache_miss}")

    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else REPO_ROOT / p


def load_cfg(path: str | Path, **overrides: Any) -> Stage1Cfg:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update({k: v for k, v in overrides.items() if v is not None})
    raw["models"] = [ModelCfg(**m) for m in raw["models"]]
    raw["query_pre"] = QueryPreCfg(**raw.get("query_pre", {}))
    raw["doc_pre"] = DocPreCfg(**raw.get("doc_pre", {}))
    raw["source_path"] = str(path)
    return Stage1Cfg(**raw)
