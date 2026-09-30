"""
Reader — loads surrounding context for search candidates.
Extracts imports, enclosing scope, and neighboring code.
"""

import time
from pathlib import Path
from typing import Optional
from app.storage.schemas import CodeChunk, AgentTraceStep
from app.indexer.ast_parser import parse_file


class Reader:
    """
    Loads full context for search candidates.
    Adds enclosing scope, imports, and neighboring definitions.
    """

    def __init__(self, context_lines: int = 30):
        self.context_lines = context_lines

    def read(
        self,
        candidates: list[tuple[CodeChunk, float]],
        max_candidates: int = 15,
    ) -> tuple[list[tuple[CodeChunk, float, dict]], AgentTraceStep]:
        """
        Load context for top candidates.
        Returns (enriched_candidates, trace_step).
        Each enriched candidate is (chunk, score, context_dict).
        """
        start = time.time()

        enriched = []
        files_read = set()

        for chunk, score in candidates[:max_candidates]:
            context = self._load_context(chunk)
            enriched.append((chunk, score, context))
            files_read.add(chunk.file_path)

        duration_ms = (time.time() - start) * 1000

        trace = AgentTraceStep(
            phase="read",
            detail=f"Loaded context for {len(enriched)} candidates from {len(files_read)} files.",
            duration_ms=round(duration_ms, 2),
            candidates_count=len(enriched),
        )

        return enriched, trace

    def _load_context(self, chunk: CodeChunk) -> dict:
        """
        Load surrounding context for a chunk.
        Returns dict with: enclosing, imports, exports, surrounding_code.
        """
        context = {
            "enclosing": chunk.node_type or "module",
            "imports": [],
            "exports": [],
            "surrounding_snippet": "",
        }

        # Try to read the full file for context
        repo_path = Path(chunk.repo_path)
        full_path = repo_path / chunk.file_path

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError):
            return context

        lines = content.split("\n")

        # Extract surrounding lines
        start = max(0, chunk.start_line - 1 - self.context_lines)
        end = min(len(lines), chunk.end_line + self.context_lines)
        context["surrounding_snippet"] = "\n".join(lines[start:end])

        # Parse for imports/exports
        try:
            parsed = parse_file(content)
            context["imports"] = [
                imp["source"] for imp in parsed.get("imports", [])
            ]
            context["exports"] = [
                exp["name"] for exp in parsed.get("exports", [])
            ]

            # Find enclosing scope
            for cls in parsed.get("classes", []):
                if cls["start_line"] <= chunk.start_line and cls["end_line"] >= chunk.end_line:
                    context["enclosing"] = f"class {cls['name']}"
                    break

        except Exception:
            pass  # AST parsing failed, continue with basic context

        return context
