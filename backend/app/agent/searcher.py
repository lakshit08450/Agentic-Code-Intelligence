"""
Searcher — executes sub-queries against semantic (FAISS) and structural (SQLite) indices.
Merges results using Reciprocal Rank Fusion (RRF).
"""

import time
from typing import Optional
from app.agent.planner import SubQuery
from app.agent.model_adapter import RetrievalModel
from app.storage.faiss_store import FAISSStore
from app.storage.sqlite_store import SQLiteStore
from app.storage.schemas import CodeChunk, AgentTraceStep
from app.config import SEMANTIC_TOP_K, STRUCTURAL_TOP_K


class Searcher:
    """
    Executes search queries against both semantic and structural indices.
    Uses Reciprocal Rank Fusion to merge results from different sources.
    """

    def __init__(self, faiss_store: FAISSStore, sqlite_store: SQLiteStore):
        self.faiss = faiss_store
        self.sqlite = sqlite_store
        self.retrieval_model = RetrievalModel(faiss_store)
        self.retrieval_model.load()

    def search(
        self,
        sub_queries: list[SubQuery],
        version: str = "main",
    ) -> tuple[list[tuple[CodeChunk, float]], AgentTraceStep]:
        """
        Execute all sub-queries and merge results.
        Returns (ranked_candidates, trace_step).
        """
        start = time.time()
        all_results: list[list[tuple[str, float]]] = []  # List of ranked lists

        semantic_count = 0
        structural_count = 0

        for sq in sub_queries:
            if sq.query_type == "semantic":
                results = self._semantic_search(sq.text)
                all_results.append(results)
                semantic_count += len(results)

            elif sq.query_type == "structural":
                results = self._structural_search(
                    version=version,
                    node_type=sq.structural_type,
                    name_pattern=sq.structural_name,
                    is_async=sq.is_async,
                    is_exported=sq.is_exported,
                )
                all_results.append(results)
                structural_count += len(results)

            elif sq.query_type == "hybrid":
                # Run both
                sem = self._semantic_search(sq.text)
                struct = self._structural_search(
                    version=version,
                    node_type=sq.structural_type,
                    name_pattern=sq.structural_name,
                )
                all_results.append(sem)
                all_results.append(struct)
                semantic_count += len(sem)
                structural_count += len(struct)

        # Merge via RRF
        merged = self._reciprocal_rank_fusion(all_results)

        # Fetch full chunk data from SQLite
        chunk_ids = [cid for cid, _ in merged[:SEMANTIC_TOP_K]]
        chunks_by_id = {c.id: c for c in self.sqlite.get_chunks_by_ids(chunk_ids)}

        ranked = []
        for chunk_id, score in merged[:SEMANTIC_TOP_K]:
            chunk = chunks_by_id.get(chunk_id)
            if chunk:
                ranked.append((chunk, score))

        duration_ms = (time.time() - start) * 1000

        trace = AgentTraceStep(
            phase="search",
            detail=f"Semantic: {semantic_count} hits, Structural: {structural_count} hits. "
                   f"Merged to {len(ranked)} candidates via RRF.",
            duration_ms=round(duration_ms, 2),
            candidates_count=len(ranked),
        )

        return ranked, trace

    def _semantic_search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[tuple[str, float]]:
        """Embed query and search using the ML model adapter."""
        return self.retrieval_model.search(query, top_k=top_k)

    def _structural_search(
        self,
        version: str,
        node_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        is_async: Optional[bool] = None,
        is_exported: Optional[bool] = None,
        limit: int = STRUCTURAL_TOP_K,
    ) -> list[tuple[str, float]]:
        """Search the SQLite AST index."""
        chunks = self.sqlite.search_structural(
            version=version,
            node_type=node_type,
            node_name_pattern=name_pattern,
            is_exported=is_exported,
            is_async=is_async,
            limit=limit,
        )
        # Assign decreasing scores based on position
        results = []
        for i, chunk in enumerate(chunks):
            score = 1.0 / (i + 1)  # Simple rank-based score
            results.append((chunk.id, score))
        return results

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[tuple[str, float]]],
        k: int = 60
    ) -> list[tuple[str, float]]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.
        RRF score = Σ 1/(k + rank_i) for each list where the item appears.

        Higher k makes the fusion more conservative (less influenced by rank).
        """
        scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, (chunk_id, _original_score) in enumerate(ranked_list):
                if chunk_id not in scores:
                    scores[chunk_id] = 0.0
                scores[chunk_id] += 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
