"""Parse sample (input, expected output) pairs from a problem statement (plan Section 11.1).

Reads only the statement text (R1). Formats seen in 1,000 validation statements (docs/LOG.md):
- dashed section headers: `-----Examples-----` / `-----Example-----` blocks containing `Input` / `Output`
  (optionally `Input:`) sub-headers (Codeforces style);
- separate dashed input/output sections: `-----Sample Input-----` + `-----Sample Output-----`,
  `-----Sample Input:-----`, `-----Example Input-----`, `-----Sample Input 1-----` (AtCoder / CodeChef);
  typo `Ouput` is accepted;
- the same titles as plain lines without dashes.
`-----Input-----` / `-----Input:-----` alone is the input *specification*, never sample data.
Call-based problems (LeetCode `Input: nums = [...]`, Codewars markdown) are reported as call_based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_DASHED = re.compile(r"^\s*-{3,}\s*(.*?)\s*-{3,}\s*$")
_IN = re.compile(r"^(?:sample|example)s?\s*(?:tests?\s*)?(?:cases?\s*)?(?:input|in)s?\s*\d*$")
_OUT = re.compile(r"^(?:sample|example)s?\s*(?:tests?\s*)?(?:cases?\s*)?(?:output|ouput|out)s?\s*\d*$")
_BLOCK = re.compile(r"^(?:sample|example)s?\s*(?:tests?\s*)?(?:cases?\s*)?\d*$")
_INNER_IN = re.compile(r"^\s*(?:sample\s+|example\s+)?input\s*\d*\s*:?\s*$", re.I)
_INNER_OUT = re.compile(r"^\s*(?:sample\s+|example\s+)?(?:output|ouput)\s*\d*\s*:?\s*$", re.I)
_LEETCODE = re.compile(r"^\s*Input:\s*(?:[A-Za-z_]\w*\s*=|\[|\")", re.M)
_CODEWARS = re.compile(r"^\s*#+\s*(?:Example|Examples|Input/Output|Task)\b|\[input\]|\[output\]|```", re.M | re.I)


@dataclass
class Samples:
    kind: str                                  # stdin | call_based | none | parse_failure
    pairs: list[tuple[str, str]] = field(default_factory=list)
    detail: str = ""


def _title(line: str) -> str | None:
    m = _DASHED.match(line)
    raw = m.group(1) if m else line
    t = re.sub(r"[:#*]", " ", raw).strip().lower()
    t = re.sub(r"\s+", " ", t)
    if m:
        return t
    # plain (undashed) header lines: only the sample/example titles count
    return t if (_IN.match(t) or _OUT.match(t) or _BLOCK.match(t)) and len(line.strip()) <= 40 else None


def _clean(lines: list[str]) -> str:
    lines = [l.rstrip() for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def _sections(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    cur_title, cur = None, []
    for line in text.replace("\r\n", "\n").split("\n"):
        t = _title(line)
        if t is not None:
            if cur_title is not None:
                out.append((cur_title, cur))
            cur_title, cur = t, []
        elif cur_title is not None:
            cur.append(line)
    if cur_title is not None:
        out.append((cur_title, cur))
    return out


def _block_pairs(lines: list[str]) -> tuple[list[str], list[str]]:
    ins, outs, mode, buf = [], [], None, []

    def flush():
        if mode == "in":
            ins.append(_clean(buf))
        elif mode == "out":
            outs.append(_clean(buf))

    for line in lines:
        if _INNER_IN.match(line):
            flush()
            mode, buf = "in", []
        elif _INNER_OUT.match(line):
            flush()
            mode, buf = "out", []
        else:
            buf.append(line)
    flush()
    return ins, outs


def parse(statement: str) -> Samples:
    ins: list[str] = []
    outs: list[str] = []
    for title, lines in _sections(statement):
        if _IN.match(title):
            ins.append(_clean(lines))
        elif _OUT.match(title):
            outs.append(_clean(lines))
        elif _BLOCK.match(title):
            bi, bo = _block_pairs(lines)
            ins += bi
            outs += bo
    if ins or outs:
        if len(ins) != len(outs):
            return Samples("parse_failure", detail=f"{len(ins)} inputs vs {len(outs)} outputs")
        pairs = [(i, o) for i, o in zip(ins, outs)]
        if any(not i.strip() or not o.strip() for i, o in pairs):
            return Samples("parse_failure", detail="empty sample input or output")
        return Samples("stdin", pairs)
    if _LEETCODE.search(statement) or _CODEWARS.search(statement) or "class Solution" in statement:
        return Samples("call_based")
    return Samples("none")
