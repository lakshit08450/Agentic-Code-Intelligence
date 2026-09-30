"""
Code Chunker — splits source files into semantically meaningful chunks
using Tree-sitter for AST-aware boundaries.
"""

import hashlib
from pathlib import Path
from typing import Optional
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from app.storage.schemas import CodeChunk
from app.config import CHUNK_MIN_LINES, CHUNK_MAX_LINES


# Initialize Tree-sitter parser for JavaScript
JS_LANGUAGE = Language(tsjs.language())
_parser = Parser(JS_LANGUAGE)


# AST node types that represent top-level code boundaries
BOUNDARY_TYPES = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "export_statement",
    "variable_declaration",
    "lexical_declaration",
    "expression_statement",
    "arrow_function",
    "generator_function_declaration",
}


def _get_function_name(node) -> Optional[str]:
    """Extract function/class name from an AST node."""
    # function_declaration / class_declaration → has a 'name' child
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
        if child.type == "property_identifier":
            return child.text.decode("utf-8")
    # For variable declarations: const foo = () => {}
    if node.type in ("variable_declaration", "lexical_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        return sub.text.decode("utf-8")
    # For export statements, look inside
    if node.type == "export_statement":
        for child in node.children:
            name = _get_function_name(child)
            if name:
                return name
    return None


def _is_async(node) -> bool:
    """Check if a node represents an async function."""
    text = node.text.decode("utf-8") if node.text else ""
    return "async" in text[:50]  # Quick heuristic check


def _is_exported(node) -> bool:
    """Check if a node is an export statement or has export parent."""
    return node.type == "export_statement"


def _node_to_type(node) -> Optional[str]:
    """Map AST node to a simplified type string."""
    if node.type == "export_statement":
        # Look at what's being exported
        for child in node.children:
            if child.type in BOUNDARY_TYPES:
                return child.type
        return "export_statement"
    return node.type if node.type in BOUNDARY_TYPES else None


def chunk_file(
    file_path: str,
    content: str,
    version: str,
    commit_sha: str,
    repo_path: str,
) -> list[CodeChunk]:
    """
    Split a JavaScript file into semantically meaningful chunks.

    Strategy:
    1. Parse with Tree-sitter to find top-level declarations
    2. Each function/class/export becomes its own chunk
    3. Consecutive small statements are grouped together
    4. Fall back to line-based chunking for unparseable sections
    """
    lines = content.split("\n")
    if len(lines) < CHUNK_MIN_LINES:
        # Entire file is one chunk
        return [_make_chunk(
            file_path, content, 1, len(lines),
            version, commit_sha, repo_path
        )]

    tree = _parser.parse(bytes(content, "utf-8"))
    root = tree.root_node
    chunks = []
    covered_lines = set()

    # Pass 1: Extract top-level AST nodes as chunks
    for child in root.children:
        start_line = child.start_point[0] + 1  # 1-indexed
        end_line = child.end_point[0] + 1
        num_lines = end_line - start_line + 1

        if child.type in BOUNDARY_TYPES or child.type == "comment":
            if num_lines >= CHUNK_MIN_LINES or child.type in ("function_declaration", "class_declaration", "method_definition"):
                node_type = _node_to_type(child)
                node_name = _get_function_name(child)
                is_exp = _is_exported(child)
                is_asn = _is_async(child)

                chunk_content = "\n".join(lines[start_line - 1:end_line])
                chunk = _make_chunk(
                    file_path, chunk_content, start_line, end_line,
                    version, commit_sha, repo_path,
                    node_type=node_type, node_name=node_name,
                    is_exported=is_exp, is_async=is_asn
                )
                chunks.append(chunk)
                covered_lines.update(range(start_line, end_line + 1))

            elif num_lines > 0:
                # Too small — will be grouped with neighbors
                pass

    # Pass 2: Group uncovered lines into chunks
    uncovered = sorted(set(range(1, len(lines) + 1)) - covered_lines)
    if uncovered:
        groups = _group_consecutive(uncovered, max_gap=3)
        for group in groups:
            if len(group) >= CHUNK_MIN_LINES:
                start = group[0]
                end = group[-1]
                chunk_content = "\n".join(lines[start - 1:end])
                chunk = _make_chunk(
                    file_path, chunk_content, start, end,
                    version, commit_sha, repo_path
                )
                chunks.append(chunk)

    # If somehow no chunks were produced, use the whole file
    if not chunks:
        chunks = [_make_chunk(
            file_path, content, 1, len(lines),
            version, commit_sha, repo_path
        )]

    # Pass 3: Split oversized chunks
    final_chunks = []
    for chunk in chunks:
        chunk_lines = chunk.content.split("\n")
        if len(chunk_lines) > CHUNK_MAX_LINES:
            splits = _split_large_chunk(chunk, chunk_lines)
            final_chunks.extend(splits)
        else:
            final_chunks.append(chunk)

    return sorted(final_chunks, key=lambda c: c.start_line)


def _make_chunk(
    file_path: str, content: str, start_line: int, end_line: int,
    version: str, commit_sha: str, repo_path: str,
    node_type: str = None, node_name: str = None,
    is_exported: bool = False, is_async: bool = False
) -> CodeChunk:
    """Create a CodeChunk with a deterministic ID."""
    chunk_id = hashlib.sha256(
        f"{file_path}:{start_line}:{end_line}:{version}".encode()
    ).hexdigest()[:16]

    return CodeChunk(
        id=chunk_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        version=version,
        commit_sha=commit_sha,
        repo_path=repo_path,
        node_type=node_type,
        node_name=node_name,
        is_exported=is_exported,
        is_async=is_async,
    )


def _group_consecutive(numbers: list[int], max_gap: int = 3) -> list[list[int]]:
    """Group consecutive numbers allowing small gaps."""
    if not numbers:
        return []
    groups = [[numbers[0]]]
    for n in numbers[1:]:
        if n - groups[-1][-1] <= max_gap:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _split_large_chunk(chunk: CodeChunk, lines: list[str]) -> list[CodeChunk]:
    """Split an oversized chunk into smaller pieces."""
    result = []
    for i in range(0, len(lines), CHUNK_MAX_LINES):
        sub_lines = lines[i:i + CHUNK_MAX_LINES]
        start = chunk.start_line + i
        end = start + len(sub_lines) - 1
        result.append(_make_chunk(
            chunk.file_path, "\n".join(sub_lines), start, end,
            chunk.version, chunk.commit_sha, chunk.repo_path,
            node_type=chunk.node_type, node_name=chunk.node_name,
            is_exported=chunk.is_exported, is_async=chunk.is_async,
        ))
    return result
