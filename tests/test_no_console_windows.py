"""Plan Section 8: no console windows. All subprocesses go through codeintel.common.proc."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from codeintel.common import proc

ROOT = Path(__file__).resolve().parents[1]
BANNED_CALLS = {
    ("subprocess", n) for n in ("run", "Popen", "call", "check_call", "check_output")
} | {("os", "system"), ("os", "startfile")}


def _offences(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if (node.func.value.id, node.func.attr) in BANNED_CALLS:
                out.append((node.lineno, f"{node.func.value.id}.{node.func.attr}()"))
        if isinstance(node, (ast.Attribute, ast.Name)) and "DETACHED_PROCESS" in (
            node.attr if isinstance(node, ast.Attribute) else node.id
        ):
            out.append((node.lineno, "DETACHED_PROCESS"))
        if isinstance(node, ast.List) and node.elts:
            words = [e.value.lower() for e in node.elts[:3] if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if words and (words[0] == "start" or words[:3] == ["cmd", "/c", "start"]):
                out.append((node.lineno, "start / cmd /c start"))
    return out


def test_only_proc_helper_starts_processes():
    offenders = []
    for d in ("src", "scripts"):
        for f in (ROOT / d).rglob("*.py"):
            if f.name == "proc.py":
                continue
            tree = ast.parse(f.read_text(encoding="utf-8"))
            offenders += [f"{f.relative_to(ROOT)}:{n}: {what}" for n, what in _offences(tree)]
    assert not offenders, "\n".join(offenders)


def test_detector_catches_violations():
    bad = "import subprocess, os\nsubprocess.Popen(['x'], creationflags=subprocess.DETACHED_PROCESS)\nos.system('x')\nproc.run(['cmd', '/c', 'start', 'x'])\n"
    kinds = {w for _, w in _offences(ast.parse(bad))}
    assert kinds == {"subprocess.Popen()", "DETACHED_PROCESS", "os.system()", "start / cmd /c start"}


def test_flags():
    if proc.NO_WINDOW:  # Windows
        assert proc.NO_WINDOW == subprocess.CREATE_NO_WINDOW
        assert proc.BACKGROUND == subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        assert not proc.BACKGROUND & subprocess.DETACHED_PROCESS
    r = proc.run(["git", "--version"], capture_output=True, text=True)
    assert r.returncode == 0
