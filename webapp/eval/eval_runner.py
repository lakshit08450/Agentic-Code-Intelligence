"""
CodeLens Evaluation Runner
Computes retrieval metrics: NDCG@10, MRR, Precision@K, Recall@K
Outputs results in MTEB/CoIR-compatible JSON format.
"""

import json
import time
import math
import argparse
from pathlib import Path
from typing import Optional


def ndcg_at_k(relevances: list[int], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.
    relevances: list of relevance scores (e.g., 0, 1, 2) for the ranked results.
    """
    dcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )

    # Ideal DCG — sort relevances in descending order
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal)
    )

    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevances: list[int]) -> float:
    """
    Mean Reciprocal Rank.
    Returns 1/rank of the first relevant result.
    """
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: list[int], k: int = 5) -> float:
    """Precision@K — fraction of top-K results that are relevant."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r > 0) / len(top_k)


def recall_at_k(relevances: list[int], total_relevant: int, k: int = 20) -> float:
    """Recall@K — fraction of all relevant documents found in top-K."""
    if total_relevant == 0:
        return 0.0
    found = sum(1 for r in relevances[:k] if r > 0)
    return found / total_relevant


def evaluate_query(
    query_results: list[dict],
    ground_truth: list[str],
    k_values: dict = None,
) -> dict:
    """
    Evaluate a single query's results against ground truth.

    Args:
        query_results: list of {file, start_line, end_line, ...} from the system
        ground_truth: list of relevant file:line_range identifiers
        k_values: dict with keys ndcg_k, precision_k, recall_k

    Returns:
        dict with metric scores
    """
    if k_values is None:
        k_values = {"ndcg_k": 10, "precision_k": 5, "recall_k": 20}

    # Build relevance vector
    gt_set = set(ground_truth)
    relevances = []
    for result in query_results:
        key = f"{result['file']}:{result['start_line']}-{result['end_line']}"
        # Also check file-level match
        if key in gt_set or result['file'] in gt_set:
            relevances.append(1)
        else:
            relevances.append(0)

    total_relevant = len(ground_truth)

    return {
        "ndcg@10": round(ndcg_at_k(relevances, k_values["ndcg_k"]), 4),
        "mrr": round(mrr(relevances), 4),
        f"precision@{k_values['precision_k']}": round(
            precision_at_k(relevances, k_values["precision_k"]), 4
        ),
        f"recall@{k_values['recall_k']}": round(
            recall_at_k(relevances, total_relevant, k_values["recall_k"]), 4
        ),
    }


def run_evaluation(
    dataset_path: str,
    api_url: str = "http://localhost:8000",
    output_path: str = None,
) -> dict:
    """
    Run full evaluation over a dataset of query-ground_truth pairs.

    Dataset JSON format:
    {
        "queries": [
            {
                "query": "How does auth work?",
                "relevant": ["src/auth.js:10-50", "src/middleware.js"],
                "version": "main"
            },
            ...
        ]
    }
    """
    import requests

    with open(dataset_path) as f:
        dataset = json.load(f)

    queries = dataset.get("queries", [])
    all_metrics = []
    latencies = []

    for i, item in enumerate(queries):
        query = item["query"]
        ground_truth = item["relevant"]
        version = item.get("version", "main")

        start = time.time()
        try:
            resp = requests.post(f"{api_url}/api/query", json={
                "query": query,
                "version": version,
                "max_results": 20,
                "include_trace": True,
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Query {i+1} failed: {e}")
            continue

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        results = data.get("results", [])
        metrics = evaluate_query(results, ground_truth)
        metrics["query"] = query
        metrics["latency_ms"] = round(latency_ms, 1)
        all_metrics.append(metrics)

        print(f"  [{i+1}/{len(queries)}] {query[:50]}... NDCG@10={metrics['ndcg@10']:.3f} MRR={metrics['mrr']:.3f} ({latency_ms:.0f}ms)")

    # Aggregate
    if not all_metrics:
        return {"error": "No successful queries"}

    aggregate = {
        "num_queries": len(all_metrics),
        "ndcg@10": round(sum(m["ndcg@10"] for m in all_metrics) / len(all_metrics), 4),
        "mrr": round(sum(m["mrr"] for m in all_metrics) / len(all_metrics), 4),
        "precision@5": round(sum(m.get("precision@5", 0) for m in all_metrics) / len(all_metrics), 4),
        "recall@20": round(sum(m.get("recall@20", 0) for m in all_metrics) / len(all_metrics), 4),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
        "per_query": all_metrics,
    }

    # Output
    output = output_path or "eval/results/eval_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*50}")
    print(f"NDCG@10:      {aggregate['ndcg@10']:.4f}")
    print(f"MRR:          {aggregate['mrr']:.4f}")
    print(f"Precision@5:  {aggregate['precision@5']:.4f}")
    print(f"Recall@20:    {aggregate['recall@20']:.4f}")
    print(f"Latency p50:  {aggregate['latency_p50_ms']:.0f}ms")
    print(f"Latency p95:  {aggregate['latency_p95_ms']:.0f}ms")
    print(f"{'='*50}")
    print(f"Results saved to {output}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeLens Evaluation Runner")
    parser.add_argument("--dataset", default="eval/dataset/test_queries.json", help="Path to evaluation dataset")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="eval/results/eval_results.json", help="Output path for results JSON")

    args = parser.parse_args()
    run_evaluation(args.dataset, args.api_url, args.output)
