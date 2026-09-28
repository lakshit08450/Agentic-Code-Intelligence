"""NDCG@k, MRR@k, recall@k per query, with pytrec_eval's tie-break (score desc, then doc id desc).

The tie-break reads doc IDs, which is evaluation, not scoring: it only orders documents whose
scores are exactly equal, the same way mteb's evaluator does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Run = Mapping[str, Mapping[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def ranked(doc_scores: Mapping[str, float]) -> list[str]:
    return [d for d, _ in sorted(doc_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def per_query(run: Run, qrels: Qrels, k_values: Sequence[int] = (1, 10, 20, 50, 100)) -> dict[str, np.ndarray]:
    """Per-query metric arrays, aligned with sorted(qrels). Queries missing from run score 0."""
    qids = sorted(qrels)
    out: dict[str, list[float]] = {}
    for q in qids:
        rel = {d: g for d, g in qrels[q].items() if g > 0}
        docs = ranked(run.get(q, {}))
        ideal = sorted(rel.values(), reverse=True)
        first = next((i for i, d in enumerate(docs) if d in rel), None)
        for k in k_values:
            dcg = sum((2 ** rel[d] - 1) / math.log2(i + 2) for i, d in enumerate(docs[:k]) if d in rel)
            idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
            out.setdefault(f"ndcg_at_{k}", []).append(dcg / idcg if idcg else 0.0)
            out.setdefault(f"mrr_at_{k}", []).append(1.0 / (first + 1) if first is not None and first < k else 0.0)
            hits = sum(1 for d in docs[:k] if d in rel)
            out.setdefault(f"recall_at_{k}", []).append(hits / len(rel) if rel else 0.0)
    return {m: np.asarray(v) for m, v in out.items()}


def summarize(pq: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: float(v.mean()) for m, v in pq.items()}


def run_from_topk(query_ids: list[str], corpus_ids: list[str], pos: np.ndarray, scores: np.ndarray) -> dict:
    """Attach IDs (as dictionary keys only) to a position/score top-k matrix."""
    return {
        q: {corpus_ids[p]: float(s) for p, s in zip(pos[i], scores[i])} for i, q in enumerate(query_ids)
    }
