"""Label-free source-format family of a statement text (used for weighting and reporting)."""

from __future__ import annotations

import re

_F = [
    ("atcoder_stdin", re.compile(r"Input is given from Standard Input|^-----Sample Input(?: \d+)?-----\s*$", re.M)),
    ("codeforces_stdin", re.compile(r"^-----Examples?-----\s*$[\s\S]*^Input\s*$", re.M)),
    ("codechef_stdin", re.compile(r"-----Sample Input:?-----|-----Example Input-----|-----Input:-----|Subtask|Example case|-----EXAMPLE-----|-----Example-----", re.M)),
    ("kattis_stdin", re.compile(r"Sample Input 1")),
    ("hackerrank_stdin", re.compile(r"=====")),
    ("leetcode_call", re.compile(r"^\s*Input:\s*(?:[A-Za-z_]\w*\s*=|\[|\")|^Example 1:\s*$|class Solution", re.M)),
    ("codewars_call", re.compile(r"```|^#+ |\bkata\b|^\s*[A-Za-z_]\w*\([^)]*\)\s*(?:==|=>|->|=|#|//)", re.M | re.I)),
]

# Test query-text mix (results/gap_analysis.json, item 3), used as importance weights on validation
TEST_MIX = {
    "codeforces_stdin": 0.7798, "atcoder_stdin": 0.1849, "codechef_stdin": 0.0127, "leetcode_call": 0.0080,
    "no_samples_or_unknown": 0.0077, "hackerrank_stdin": 0.0042, "other_stdin": 0.0021, "kattis_stdin": 0.0005,
    "codewars_call": 0.0,
}


def family(text: str) -> str:
    for name, rx in _F:
        if rx.search(text):
            return name
    return "other_stdin" if re.search(r"-----Input-----|^Input\s*$", text, re.M) else "no_samples_or_unknown"
