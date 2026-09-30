"""
Planner — decomposes natural-language queries into actionable sub-queries.
Uses rule-based heuristics for CPU efficiency, no LLM required.
"""

import re
import time
from typing import Optional
from app.storage.schemas import AgentTraceStep


# Structural keywords that hint at AST-level search
STRUCTURAL_KEYWORDS = {
    "function": "function_declaration",
    "functions": "function_declaration",
    "class": "class_declaration",
    "classes": "class_declaration",
    "method": "method_definition",
    "methods": "method_definition",
    "export": None,  # filter flag
    "exported": None,
    "async": None,
    "arrow": "arrow_function",
    "variable": "variable_declaration",
    "const": "lexical_declaration",
    "import": "import_statement",
}

# Intent patterns
INTENT_PATTERNS = [
    (r"how\s+does|how\s+is|explain|what\s+does", "explain"),
    (r"find|show|list|get|search|where\s+is", "find"),
    (r"compare|difference|diff|versus|vs", "compare"),
    (r"fix|bug|error|issue|problem", "debug"),
    (r"optimize|improve|refactor|performance", "optimize"),
]


class SubQuery:
    """A decomposed sub-query with metadata."""
    def __init__(self, text: str, query_type: str = "semantic",
                 structural_type: str = None, structural_name: str = None,
                 is_async: bool = None, is_exported: bool = None):
        self.text = text
        self.query_type = query_type  # "semantic" | "structural" | "hybrid"
        self.structural_type = structural_type
        self.structural_name = structural_name
        self.is_async = is_async
        self.is_exported = is_exported

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.query_type,
            "structural_type": self.structural_type,
            "structural_name": self.structural_name,
        }


class Planner:
    """
    Decomposes a natural language query into sub-queries.

    Strategy:
    1. Detect intent (find, explain, compare, debug, optimize)
    2. Extract structural hints (function, class, async, export)
    3. Extract identifiers (camelCase, snake_case tokens)
    4. Generate 2-4 sub-queries: always a semantic query + any structural queries
    """

    def plan(self, query: str, mode: str = "hybrid") -> tuple[list[SubQuery], AgentTraceStep]:
        """
        Analyze query and produce sub-queries.
        Returns (sub_queries, trace_step).
        """
        start = time.time()

        intent = self._detect_intent(query)
        structural_hints = self._extract_structural_hints(query)
        identifiers = self._extract_identifiers(query)
        keywords = self._extract_keywords(query)

        sub_queries = []

        # 1. Always include a semantic query (the full NL query)
        if mode in ("semantic", "hybrid"):
            sub_queries.append(SubQuery(
                text=query,
                query_type="semantic"
            ))

            # 2. If identifiers found, add a focused semantic query
            if identifiers:
                identifier_query = " ".join(identifiers)
                sub_queries.append(SubQuery(
                    text=identifier_query,
                    query_type="semantic"
                ))

            # 4. If we have keywords but no structural hints, add a keyword-focused query
            if keywords and not structural_hints:
                kw_query = " ".join(keywords)
                if kw_query != query:
                    sub_queries.append(SubQuery(
                        text=kw_query,
                        query_type="semantic"
                    ))

        # 3. Add structural queries based on detected hints
        if mode in ("structural", "hybrid"):
            for hint in structural_hints:
                sq = SubQuery(
                    text=query,
                    query_type="structural",
                    structural_type=hint.get("node_type"),
                    structural_name=hint.get("name_pattern"),
                    is_async=hint.get("is_async"),
                    is_exported=hint.get("is_exported"),
                )
                sub_queries.append(sq)
                
            # If structural mode is forced but no hints found, we just do a generic structural query
            if mode == "structural" and not structural_hints:
                 sub_queries.append(SubQuery(
                    text=query,
                    query_type="structural"
                ))

        # Deduplicate
        seen = set()
        unique = []
        for sq in sub_queries:
            key = (sq.text, sq.query_type, sq.structural_type)
            if key not in seen:
                seen.add(key)
                unique.append(sq)

        duration_ms = (time.time() - start) * 1000

        trace = AgentTraceStep(
            phase="plan",
            detail=f"Intent: {intent}. Decomposed into {len(unique)} sub-queries.",
            duration_ms=round(duration_ms, 2),
            sub_queries=[sq.text for sq in unique],
            candidates_count=0,
        )

        return unique, trace

    def _detect_intent(self, query: str) -> str:
        """Detect the user's intent from the query."""
        query_lower = query.lower()
        for pattern, intent in INTENT_PATTERNS:
            if re.search(pattern, query_lower):
                return intent
        return "find"

    def _extract_structural_hints(self, query: str) -> list[dict]:
        """Extract structural query hints from keywords."""
        hints = []
        query_lower = query.lower()
        words = query_lower.split()

        node_type = None
        is_async = None
        is_exported = None

        for word in words:
            if word in STRUCTURAL_KEYWORDS:
                nt = STRUCTURAL_KEYWORDS[word]
                if nt:
                    node_type = nt
                if word in ("async",):
                    is_async = True
                if word in ("export", "exported"):
                    is_exported = True

        # Look for name patterns (identifiers near structural keywords)
        identifiers = self._extract_identifiers(query)
        name_pattern = identifiers[0] if identifiers else None

        if node_type or is_async is not None or is_exported is not None:
            hints.append({
                "node_type": node_type,
                "name_pattern": name_pattern,
                "is_async": is_async,
                "is_exported": is_exported,
            })

        return hints

    def _extract_identifiers(self, query: str) -> list[str]:
        """
        Extract probable code identifiers from the query.
        Matches camelCase, snake_case, PascalCase, and dot-notation.
        """
        # Match patterns that look like code identifiers
        patterns = [
            r'[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*',  # camelCase
            r'[A-Z][a-z]+[A-Z][a-zA-Z0-9]*',         # PascalCase
            r'[a-z]+_[a-z_]+',                         # snake_case
            r'[a-zA-Z]+\.[a-zA-Z]+',                   # dot.notation
            r'`([^`]+)`',                               # backtick-quoted
        ]

        identifiers = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            identifiers.extend(matches)

        # Also extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        identifiers.extend(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        identifiers.extend(quoted)

        return list(dict.fromkeys(identifiers))  # Deduplicate preserving order

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords (remove stop words)."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "its", "this", "that", "these", "those", "my", "your",
            "his", "her", "our", "their", "what", "which", "who", "whom",
            "how", "where", "when", "why", "all", "each", "every", "both",
            "and", "or", "but", "not", "no", "if", "then", "else",
            "me", "i", "you", "he", "she", "we", "they",
            "find", "show", "list", "get", "search", "explain",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
