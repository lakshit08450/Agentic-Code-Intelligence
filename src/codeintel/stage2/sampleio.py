"""Parse sample (input, expected output) pairs from a problem statement (plan Section 11.1).

Reads only the statement text (R1). Formats seen in the 1,000 validation statements (docs/LOG.md):
- dashed section headers: `-----Examples-----` / `-----Example-----` blocks containing `Input` / `Output`
  sub-headers, optionally `Input:` and, in dashed blocks only, data on the same line (`Input:4`);
- separate input/output sections: `-----Sample Input-----` + `-----Sample Output-----`,
  `-----Sample Input:-----`, `-----Example Input-----`, `-----Sample Input 1-----`, typo `Ouput`,
  glued `ExampleInput:`; a sample input section followed by a plain `-----Output-----`;
- block titles such as `-----Example Text Case-----`; header lines with trailing text
  (`-----EXAMPLE-----Input:`, `-----Sample Input:-----Sample Input:`); `=====X=====` headers;
- the same titles as plain lines without dashes.
`-----Input-----` / `-----Input:-----` alone is the input *specification*, never sample data.
Post-processing (from the gold-failure review): expected outputs are cut where explanation prose
starts; blank lines inside sample inputs (a CodeChef formatting artefact) are dropped.
Call-based problems (LeetCode `Input: nums = [...]`, Codewars markdown) are reported as call_based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_DASHED = re.compile(r"^\s*(?:-{3,}|={3,})\s*(.*?)\s*(?:-{3,}|={3,})\s*(.*?)\s*$")
_IN = re.compile(r"^(?:sample|example)s?\s*(?:tests?\s*)?(?:cases?\s*)?(?:input|in)s?\s*\d*$")
_OUT = re.compile(r"^(?:sample|example)s?\s*(?:tests?\s*)?(?:cases?\s*)?(?:output|ouput|out)s?\s*\d*$")
_BLOCK = re.compile(r"^(?:sample|example)s?\s*(?:text\s*|tests?\s*)?(?:cases?\s*)?\d*$")
_GENERIC_OUT = re.compile(r"^(?:output|ouput)$")
_INNER_IN = re.compile(r"^\s*(?:sample\s+|example\s+)?input\s*\d*\s*:?\s*$", re.I)
_INNER_OUT = re.compile(r"^\s*(?:sample\s+|example\s+)?(?:output|ouput)\s*\d*\s*:?\s*$", re.I)
_INLINE_IN = re.compile(r"^\s*(?:sample\s+|example\s+)?input\s*\d*\s*:\s*(\S.*)$", re.I)
_INLINE_OUT = re.compile(r"^\s*(?:sample\s+|example\s+)?(?:output|ouput)\s*\d*\s*:\s*(\S.*)$", re.I)
_LEETCODE = re.compile(r"^\s*Input:\s*(?:[A-Za-z_]\w*\s*=|\[|\")", re.M)
_CODEWARS = re.compile(r"^\s*#+\s*(?:Example|Examples|Input/Output|Task)\b|\[input\]|\[output\]|```", re.M | re.I)
_EXPLANATION_LINE = re.compile(r"^\s*\(?\s*(?:explanation|explanations|note|notes|hint)\b", re.I)
_PROSE_START = re.compile(
    r"^\s*(?:in the |in test|in this |in case|for the |for first|for second|for example|for each |there (?:are|is|was) |"
    r"if |the |we |here|this |these |after |initially|so |thus |as |since |one of|- |by:|testcase|test case|"
    r"example case|case \d|sample case|it |you |note:)",
    re.I,
)


@dataclass
class Samples:
    kind: str                                  # stdin | call_based | none | parse_failure
    pairs: list[tuple[str, str]] = field(default_factory=list)
    detail: str = ""


def _norm_title(raw: str) -> str:
    t = re.sub(r"[:#*]", " ", raw).strip().lower()
    t = re.sub(r"(sample|example)s?(input|output|ouput)", r"\1 \2", t)
    return re.sub(r"\s+", " ", t)


def _is_title(t: str) -> bool:
    return bool(_IN.match(t) or _OUT.match(t) or _BLOCK.match(t))


def _clean(lines: list[str]) -> str:
    lines = [l.rstrip() for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def _sections(text: str) -> list[tuple[str, list[str], bool]]:
    """(normalised title, content lines, dashed?) for every header line."""
    out: list[tuple[str, list[str], bool]] = []
    cur_title, cur, cur_dashed = None, [], False
    for line in text.replace("\r\n", "\n").split("\n"):
        m = _DASHED.match(line)
        if m:
            title, rest, dashed = _norm_title(m.group(1)), m.group(2), True
        else:
            t = _norm_title(line)
            title, rest, dashed = (t if _is_title(t) and len(line.strip()) <= 40 else None), "", False
        if title is not None:
            if cur_title is not None:
                out.append((cur_title, cur, cur_dashed))
            cur_title, cur, cur_dashed = title, [], dashed
            if rest and _norm_title(rest) != title:  # `-----EXAMPLE-----Input:` keeps `Input:`
                cur.append(rest)
        elif cur_title is not None:
            cur.append(line)
    if cur_title is not None:
        out.append((cur_title, cur, cur_dashed))
    return out


def _block_pairs(lines: list[str], inline: bool) -> tuple[list[str], list[str]]:
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
        elif inline and (m := _INLINE_IN.match(line)):
            flush()
            mode, buf = "in", [m.group(1)]
        elif inline and (m := _INLINE_OUT.match(line)):
            flush()
            mode, buf = "out", [m.group(1)]
        else:
            buf.append(line)
    flush()
    return ins, outs


def _is_sentence(line: str) -> bool:
    words = line.split()
    alpha = [w for w in words if re.fullmatch(r"[A-Za-z][a-z]*[,.:;!?)]?", w)]
    return len(words) >= 6 and len(alpha) >= 0.6 * len(words)


def trim_output(text: str) -> str:
    """Cut an expected output where explanation prose starts."""
    kept: list[str] = []
    seen_blank = False
    for line in text.split("\n"):
        if _EXPLANATION_LINE.match(line):
            break
        if not line.strip():
            seen_blank = True
            kept.append(line)
            continue
        if kept and seen_blank and (_PROSE_START.match(line) or _is_sentence(line)):
            break
        kept.append(line)
    return _clean(kept)


def drop_blank_lines(text: str) -> str:
    return _clean([l for l in text.split("\n") if l.strip()])


def parse(statement: str) -> Samples:
    ins: list[str] = []
    outs: list[str] = []
    prev_in = False
    for title, lines, dashed in _sections(statement):
        if _IN.match(title):
            ins.append(_clean(lines))
            prev_in = True
            continue
        if _OUT.match(title) or (dashed and prev_in and _GENERIC_OUT.match(title) and len(outs) < len(ins)):
            outs.append(_clean(lines))
        elif _BLOCK.match(title):
            bi, bo = _block_pairs(lines, inline=dashed)
            ins += bi
            outs += bo
        prev_in = False
    if ins or outs:
        if len(ins) != len(outs):
            return Samples("parse_failure", detail=f"{len(ins)} inputs vs {len(outs)} outputs")
        pairs = [(drop_blank_lines(i), trim_output(o)) for i, o in zip(ins, outs)]
        if any(not i.strip() or not o.strip() for i, o in pairs):
            return Samples("parse_failure", detail="empty sample input or output")
        return Samples("stdin", pairs)
    if _LEETCODE.search(statement) or _CODEWARS.search(statement) or "class Solution" in statement:
        return Samples("call_based")
    return Samples("none")
