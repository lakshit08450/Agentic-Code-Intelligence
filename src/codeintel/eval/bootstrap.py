"""Paired bootstrap over queries (gate G3)."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 13) -> dict[str, float]:
    """CI for mean(b - a). G3 passes when the 95% CI excludes 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}
