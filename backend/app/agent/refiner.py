"""
Refiner — re-ranks search candidates and applies diversity filtering.
Uses a lightweight cross-encoder for CPU-efficient re-ranking.
"""

import time
from typing import Optional
from app.storage.schemas import CodeChunk, SearchResult, AgentTraceStep
from app.config import FINAL_TOP_K


class Refiner:
    """
    Re-ranks candidates using:
    1. Heuristic scoring (file relevance, name match, etc.)
    2. Optional cross-encoder re-ranking (CPU)
    3. Diversity filter (avoid too many results from same file)
    """

    def __init__(self, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

    def _load_cross_encoder(self):
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is None and self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                from app.config import CROSS_ENCODER_MODEL
                self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            except Exception:
                self.use_cross_encoder = False

    def refine(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float, dict]],
        max_results: int = FINAL_TOP_K,
        max_per_file: int = 3,
    ) -> tuple[list[SearchResult], AgentTraceStep]:
        """
        Re-rank and filter candidates.
        Returns (final_results, trace_step).
        """
        start = time.time()

        if not candidates:
            trace = AgentTraceStep(
                phase="refine",
                detail="No candidates to refine.",
                duration_ms=0,
                candidates_count=0,
            )
            return [], trace

        # Step 1: Score candidates
        scored = []
        for chunk, base_score, context in candidates:
            refined_score = self._compute_score(query, chunk, base_score, context)
            scored.append((chunk, refined_score, context))

        # Step 2: Optional cross-encoder re-ranking
        if self.use_cross_encoder and len(scored) > 0:
            scored = self._cross_encoder_rerank(query, scored)

        # Step 3: Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Diversity filter
        filtered = self._diversity_filter(scored, max_per_file=max_per_file)

        # Step 5: Build final results
        results = []
        for rank, (chunk, score, context) in enumerate(filtered[:max_results], 1):
            results.append(SearchResult(
                rank=rank,
                score=round(score, 4),
                file=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                snippet=chunk.content,
                language=chunk.language,
                version=chunk.version,
                commit_sha=chunk.commit_sha,
                node_type=chunk.node_type,
                node_name=chunk.node_name,
                context={
                    "enclosing": context.get("enclosing", "module"),
                    "imports": context.get("imports", []),
                },
            ))

        duration_ms = (time.time() - start) * 1000

        trace = AgentTraceStep(
            phase="refine",
            detail=f"Re-ranked {len(candidates)} candidates. "
                   f"Diversity-filtered to {len(results)} results.",
            duration_ms=round(duration_ms, 2),
            candidates_count=len(results),
        )

        return results, trace

    def _compute_score(
        self,
        query: str,
        chunk: CodeChunk,
        base_score: float,
        context: dict,
    ) -> float:
        """
        Compute a refined score combining base retrieval score with heuristics.
        """
        score = base_score

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Boost: chunk name matches query terms
        if chunk.node_name:
            name_lower = chunk.node_name.lower()
            # Exact name match
            if name_lower in query_lower:
                score += 0.3
            # Partial match
            elif any(w in name_lower for w in query_words if len(w) > 3):
                score += 0.15

        # Boost: content contains query keywords
        content_lower = chunk.content.lower()
        matching_words = sum(1 for w in query_words if w in content_lower and len(w) > 3)
        score += matching_words * 0.05

        # Boost: named functions/classes are generally more useful
        if chunk.node_type in ("function_declaration", "class_declaration"):
            score += 0.1

        # Slight penalty for very short or very long chunks
        lines = chunk.end_line - chunk.start_line + 1
        if lines < 3:
            score -= 0.1
        elif lines > 80:
            score -= 0.05

        # Boost exported items slightly
        if chunk.is_exported:
            score += 0.05

        return score

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float, dict]],
    ) -> list[tuple[CodeChunk, float, dict]]:
        """Re-rank using cross-encoder model."""
        self._load_cross_encoder()
        if self._cross_encoder is None:
            return candidates

        pairs = [(query, c.content[:512]) for c, _, _ in candidates]
        ce_scores = self._cross_encoder.predict(pairs)

        reranked = []
        for (chunk, _old_score, ctx), ce_score in zip(candidates, ce_scores):
            reranked.append((chunk, float(ce_score), ctx))

        return reranked

    def _diversity_filter(
        self,
        candidates: list[tuple[CodeChunk, float, dict]],
        max_per_file: int = 3,
    ) -> list[tuple[CodeChunk, float, dict]]:
        """Limit results per file to ensure diversity."""
        file_counts: dict[str, int] = {}
        filtered = []

        for item in candidates:
            chunk = item[0]
            count = file_counts.get(chunk.file_path, 0)
            if count < max_per_file:
                filtered.append(item)
                file_counts[chunk.file_path] = count + 1

        return filtered
