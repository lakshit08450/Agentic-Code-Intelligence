"""Call-based examples from a statement (plan step 5 harness, parent side; reads statement text only, R1).

Two formats cover the validation statements:
- LeetCode: `Input: n = 12, k = 3` ... `Output: 3` (also `Input: [1,2,3]` positional); JSON-style
  `true` / `false` / `null` are accepted. The method is found in the candidate's `class Solution`.
- Codewars: call lines such as `solve([1,2,3]) = "A"`, `f(3) == 6`, `vaporcode("x") # output => "X"`,
  `f(1, 2) => 3`, `f(2) returns 4`, `f(0) -> True`. The function name comes from the examples.
Arguments and expected values are parsed with `ast` and must be pure literals.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field

_JSON_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
_LC_INPUT = re.compile(r"^\s*(?:\*\*)?Input(?:\*\*)?\s*:\s*(.*)$", re.I)
_LC_OUTPUT = re.compile(r"^\s*(?:\*\*)?Output(?:\*\*)?\s*:\s*(.*)$", re.I)
_LC_STOP = re.compile(r"^\s*(?:Explanation|Example|Note|Constraints|Follow)", re.I)
_SEP = (
    r"(?:===|==>|==|=+>|-+>|→|=|"
    r"(?:#|//|--)\s*(?:output\s*)?(?:=>|->|returns?|==|:)|"
    r"(?:#|//)?\s*,?\s*(?:should\s+(?:return|be|equal|give)|returns?|gives?|outputs?)\s*:?)"
)
_CW_CALL = re.compile(r"^[\s>*`\-]*([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*" + _SEP + r"\s*(.+?)\s*$")
_BARE = re.compile(r"^[\s>*`\-]*([\[\(\"'{\d-].*?)\s*(?:==>|=+>|-+>|→)\s*(.+?)\s*$")


@dataclass
class CallExamples:
    kind: str                                   # leetcode | codewars | none
    func: str | None = None                     # codewars function name; None = Solution method
    calls: list[tuple[str, str]] = field(default_factory=list)  # (repr(args tuple), repr(expected))


class _Names(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name):
        if node.id in _JSON_NAMES:
            return ast.copy_location(ast.Constant(_JSON_NAMES[node.id]), node)
        raise ValueError(f"not a literal: {node.id}")


def literal(node: ast.AST):
    return ast.literal_eval(_Names().visit(node))


def _parse_expected(text: str):
    """Expected value from the text after the separator; trailing comments are cut."""
    text = text.strip().rstrip(";").strip("`").strip()
    text = re.sub(r"^(?:returns?|return:|should return)\s*", "", text, flags=re.I).strip()
    cands = [text]
    for cut in (" --", " //", " #", "  ", " (", ", ", ". "):
        if cut in text:
            cands.append(text.split(cut, 1)[0])
    if text.endswith("."):
        cands.append(text[:-1])
    for c in cands:
        try:
            return literal(ast.parse(c.strip().strip("`"), mode="eval").body)
        except (SyntaxError, ValueError, TypeError, RecursionError):
            continue
    raise ValueError(f"expected value not a literal: {text[:60]!r}")


def _parse_args(argtext: str) -> tuple:
    call = ast.parse(f"_f({argtext})", mode="eval").body
    if not isinstance(call, ast.Call):
        raise ValueError("not a call")
    args = [literal(a) for a in call.args]
    args += [literal(k.value) for k in call.keywords]  # keyword examples -> positional, in order
    return tuple(args)


def _leetcode(statement: str) -> list[tuple[str, str]]:
    calls = []
    lines = statement.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _LC_INPUT.match(lines[i])
        if not m:
            i += 1
            continue
        inp = [m.group(1)]
        j = i + 1
        while j < len(lines) and not _LC_OUTPUT.match(lines[j]) and not _LC_STOP.match(lines[j]) and j - i < 30:
            inp.append(lines[j])
            j += 1
        if j >= len(lines) or not _LC_OUTPUT.match(lines[j]):
            i = j
            continue
        out = [_LC_OUTPUT.match(lines[j]).group(1)]
        k = j + 1
        while k < len(lines) and lines[k].strip() and not _LC_STOP.match(lines[k]) and not _LC_INPUT.match(lines[k]) and k - j < 30:
            out.append(lines[k])
            k += 1
        try:
            args = _parse_args(" ".join(x.strip() for x in inp))
            expected = _parse_expected(" ".join(x.strip() for x in out))
            calls.append((repr(args), repr(expected)))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            pass
        i = k
    return calls


def _codewars(statement: str) -> tuple[str | None, list[tuple[str, str]]]:
    found: list[tuple[str, str, str]] = []
    for line in statement.replace("\r\n", "\n").split("\n"):
        m = _CW_CALL.match(line)
        if not m:
            continue
        name, argtext, exp = m.groups()
        try:
            found.append((name, repr(_parse_args(argtext)), repr(_parse_expected(exp))))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            continue
    if not found:
        return _bare(statement)
    name = Counter(n for n, _, _ in found).most_common(1)[0][0]
    seen, calls = set(), []
    for n, a, e in found:
        if n == name and (a, e) not in seen:  # statements often repeat examples per language
            seen.add((a, e))
            calls.append((a, e))
    return name, calls


def _bare(statement: str) -> tuple[str | None, list[tuple[str, str]]]:
    """`[1, 2, 3] => 6` style single-argument examples (function name resolved in the candidate)."""
    seen, calls = set(), []
    for line in statement.replace("\r\n", "\n").split("\n"):
        m = _BARE.match(line)
        if not m:
            continue
        try:
            arg = literal(ast.parse(m.group(1).strip().strip("`"), mode="eval").body)
            item = (repr((arg,)), repr(_parse_expected(m.group(2))))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            continue
        if item not in seen:
            seen.add(item)
            calls.append(item)
    return None, calls


def parse_calls(statement: str, max_calls: int = 8) -> CallExamples:
    lc = _leetcode(statement)
    if lc:
        return CallExamples("leetcode", None, lc[:max_calls])
    name, cw = _codewars(statement)
    if cw:
        return CallExamples("codewars", name, cw[:max_calls])  # name None: bare examples
    return CallExamples("none")
