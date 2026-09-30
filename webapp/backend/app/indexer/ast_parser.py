"""
AST Parser — extracts structural metadata from JavaScript files using Tree-sitter.
Powers structural queries like "find async functions" or "exported classes".
"""

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from typing import Optional


JS_LANGUAGE = Language(tsjs.language())
_parser = Parser(JS_LANGUAGE)


def parse_file(content: str) -> dict:
    """
    Parse a JavaScript file and extract structural metadata.

    Returns dict with:
    - functions: list of {name, start_line, end_line, is_async, is_exported, params}
    - classes: list of {name, start_line, end_line, methods, is_exported}
    - imports: list of {source, specifiers}
    - exports: list of {name, type}
    """
    tree = _parser.parse(bytes(content, "utf-8"))
    root = tree.root_node

    result = {
        "functions": [],
        "classes": [],
        "imports": [],
        "exports": [],
        "variables": [],
    }

    _walk_node(root, result, is_exported=False)
    return result


def _walk_node(node, result: dict, is_exported: bool = False):
    """Recursively walk AST and extract structural info."""

    if node.type == "function_declaration":
        func = _extract_function(node, is_exported)
        if func:
            result["functions"].append(func)

    elif node.type == "generator_function_declaration":
        func = _extract_function(node, is_exported)
        if func:
            func["is_generator"] = True
            result["functions"].append(func)

    elif node.type == "class_declaration":
        cls = _extract_class(node, is_exported)
        if cls:
            result["classes"].append(cls)

    elif node.type in ("variable_declaration", "lexical_declaration"):
        _extract_variable_functions(node, result, is_exported)

    elif node.type == "export_statement":
        for child in node.children:
            _walk_node(child, result, is_exported=True)
        # Track what's exported
        export_info = _extract_export(node)
        if export_info:
            result["exports"].append(export_info)
        return  # Already recursed into children

    elif node.type == "import_statement":
        imp = _extract_import(node)
        if imp:
            result["imports"].append(imp)

    # Recurse into children
    for child in node.children:
        _walk_node(child, result, is_exported=False)


def _extract_function(node, is_exported: bool) -> Optional[dict]:
    """Extract function metadata from AST node."""
    name = None
    params = []
    is_async = False

    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
        elif child.type == "formal_parameters":
            params = _extract_params(child)
        elif child.type == "async":
            is_async = True

    # Check if parent text starts with async
    text = node.text.decode("utf-8") if node.text else ""
    if text.strip().startswith("async"):
        is_async = True

    if name:
        return {
            "name": name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "is_async": is_async,
            "is_exported": is_exported,
            "params": params,
        }
    return None


def _extract_class(node, is_exported: bool) -> Optional[dict]:
    """Extract class metadata from AST node."""
    name = None
    methods = []

    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
        elif child.type == "class_body":
            for member in child.children:
                if member.type == "method_definition":
                    method = _extract_method(member)
                    if method:
                        methods.append(method)

    if name:
        return {
            "name": name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "is_exported": is_exported,
            "methods": methods,
        }
    return None


def _extract_method(node) -> Optional[dict]:
    """Extract method metadata from a class body."""
    name = None
    is_async = False
    is_static = False
    params = []

    text = node.text.decode("utf-8") if node.text else ""
    if "async" in text[:30]:
        is_async = True
    if "static" in text[:30]:
        is_static = True

    for child in node.children:
        if child.type == "property_identifier":
            name = child.text.decode("utf-8")
        elif child.type == "formal_parameters":
            params = _extract_params(child)

    if name:
        return {
            "name": name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "is_async": is_async,
            "is_static": is_static,
            "params": params,
        }
    return None


def _extract_params(node) -> list[str]:
    """Extract parameter names from formal_parameters node."""
    params = []
    for child in node.children:
        if child.type == "identifier":
            params.append(child.text.decode("utf-8"))
        elif child.type == "assignment_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    params.append(sub.text.decode("utf-8"))
                    break
        elif child.type == "rest_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    params.append(f"...{sub.text.decode('utf-8')}")
                    break
    return params


def _extract_import(node) -> Optional[dict]:
    """Extract import metadata."""
    source = None
    specifiers = []

    for child in node.children:
        if child.type == "string":
            source = child.text.decode("utf-8").strip("'\"")
        elif child.type == "import_clause":
            for sub in child.children:
                if sub.type == "identifier":
                    specifiers.append(sub.text.decode("utf-8"))
                elif sub.type == "named_imports":
                    for named in sub.children:
                        if named.type == "import_specifier":
                            for s in named.children:
                                if s.type == "identifier":
                                    specifiers.append(s.text.decode("utf-8"))
                                    break

    if source:
        return {"source": source, "specifiers": specifiers}
    return None


def _extract_export(node) -> Optional[dict]:
    """Extract export metadata."""
    for child in node.children:
        if child.type == "function_declaration":
            name = _get_name(child)
            if name:
                return {"name": name, "type": "function"}
        elif child.type == "class_declaration":
            name = _get_name(child)
            if name:
                return {"name": name, "type": "class"}
        elif child.type in ("variable_declaration", "lexical_declaration"):
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name = _get_name(sub)
                    if name:
                        return {"name": name, "type": "variable"}
    return None


def _extract_variable_functions(node, result: dict, is_exported: bool):
    """Extract arrow functions / function expressions assigned to variables."""
    for child in node.children:
        if child.type == "variable_declarator":
            name = None
            has_function = False
            is_async = False

            for sub in child.children:
                if sub.type == "identifier":
                    name = sub.text.decode("utf-8")
                elif sub.type in ("arrow_function", "function_expression", "function"):
                    has_function = True
                    text = sub.text.decode("utf-8") if sub.text else ""
                    if text.strip().startswith("async"):
                        is_async = True

            if name and has_function:
                result["functions"].append({
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "is_async": is_async,
                    "is_exported": is_exported,
                    "params": [],
                })


def _get_name(node) -> Optional[str]:
    """Get the name identifier from a node."""
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return None
