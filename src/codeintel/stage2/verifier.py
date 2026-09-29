"""Execution verification of one (statement, program) pair (plan Section 11.4).

Outcome in {PASS, WRONG, ERROR, TIMEOUT, UNKNOWN}. Samples run in order and stop at the first
non-passing one; PASS needs every sample to pass. Only texts are used (R1).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from codeintel.stage2.compare import outputs_match
from codeintel.stage2.sampleio import Samples, parse
from codeintel.stage2.sandbox import ExecCache, run_cached


@dataclass
class Verdict:
    outcome: str        # PASS | WRONG | ERROR | TIMEOUT | UNKNOWN
    reason: str         # e.g. all_samples, sample_1_wrong, exit_1, guard, py2_syntax, no_samples, call_based
    runs: int           # sandbox executions performed (cache misses)


@lru_cache(maxsize=20000)
def samples_for(statement: str) -> Samples:
    return parse(statement)


_READS_STDIN = re.compile(r"\binput\s*\(|sys\.stdin|open\(\s*0\s*[,)]|os\.read\(\s*0|\bstdin\b")
_DEFINES_CALLABLE = re.compile(r"^class\s+Solution\b|^def\s+[A-Za-z_]\w*\s*\(", re.M)


def program_mode(program: str) -> str:
    """'stdin' if the program reads standard input, 'call' if it only defines functions/classes.

    Reads only the candidate's text (R1). Used to route LeetCode/Codewars solutions to the
    call-based path even when the statement also contains Input/Output blocks.
    """
    if _READS_STDIN.search(program):
        return "stdin"
    return "call" if _DEFINES_CALLABLE.search(program) else "stdin"


def verify(
    statement: str, program: str, cache: ExecCache | None, case_insensitive: bool = False, cache_only: bool = False
) -> Verdict:
    """cache_only=True never executes anything: a missing run gives outcome MISSING (analysis/tuning)."""
    if program_mode(program) == "call":
        return Verdict("UNKNOWN", "call_program", 0)
    s = samples_for(statement)
    if s.kind != "stdin":
        return Verdict("UNKNOWN", {"none": "no_samples"}.get(s.kind, s.kind), 0)
    runs = 0
    for n, (inp, expected) in enumerate(s.pairs, 1):
        if cache_only:
            r, hit = (cache.get(cache.key(program, inp)) if cache is not None else None), True
            if r is None:
                return Verdict("MISSING", f"sample_{n}_not_cached", 0)
        else:
            r, hit = run_cached(cache, program, inp)
        runs += 0 if hit else 1
        if r.status == "OK":
            if not outputs_match(r.stdout, expected, case_insensitive=case_insensitive):
                return Verdict("WRONG", f"sample_{n}_wrong", runs)
        else:
            return Verdict(r.status, r.reason, runs)
    return Verdict("PASS", "all_samples", runs)


def verify_many(
    pairs: Iterable[tuple[str, str]], cache: ExecCache | None, workers: int = 12
) -> list[Verdict]:
    pairs = list(pairs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: verify(p[0], p[1], cache), pairs))
