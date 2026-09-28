from __future__ import annotations

import hashlib

import numpy as np
import pytest

from codeintel.common.config import ModelCfg, Stage1Cfg


class FakeBackend:
    """Deterministic text-only embedder: hashed character trigrams, unit norm."""

    def __init__(self, dim: int = 64, salt: str = "") -> None:
        self.dim, self.salt = dim, salt
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), np.float32)
        for r, t in enumerate(texts):
            s = self.salt + t
            for i in range(max(1, len(s) - 2)):
                h = int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16)
                out[r, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


def make_cfg(tmp_path, n_models: int = 1, weights=None) -> Stage1Cfg:
    models = [ModelCfg(id=f"fake/m{i}", query_prompt="q: ", doc_prompt="") for i in range(n_models)]
    return Stage1Cfg(
        name="test",
        models=models,
        fusion_weights=weights or [1.0 / n_models] * n_models,
        cache_dir=str(tmp_path / "emb"),
    )


@pytest.fixture
def fake_backend_cls():
    return FakeBackend
