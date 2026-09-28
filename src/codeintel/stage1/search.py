"""Stage 1 dense search over texts. Inputs and outputs are texts/positions only (R1)."""

from __future__ import annotations

import numpy as np

from codeintel.stage1.encoder import PrePostPipelineEncoder


def top_k(q: np.ndarray, d: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k for unit vectors. Returns (positions, scores), both (n_q, k), best first."""
    sims = q @ d.T
    k = min(k, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_scores, order, axis=1)


def search(enc: PrePostPipelineEncoder, query_texts: list[str], corpus_texts: list[str], k: int):
    q = enc.embed(query_texts, is_query=True)
    d = enc.embed(corpus_texts, is_query=False)
    return top_k(q, d, k)
