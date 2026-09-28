"""Execution verification of one (statement, program) pair (plan Section 11.4).

Outcome in {PASS, WRONG, ERROR, TIMEOUT, UNKNOWN}. Samples run in order and stop at the first
non-passing one; PASS needs every sample to pass. Only texts are used (R1).
"""

from __future__ import annotations

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


def verify(statement: str, program: str, cache: ExecCache | None, case_insensitive: bool = False) -> Verdict:
    s = samples_for(statement)
    if s.kind != "stdin":
        return Verdict("UNKNOWN", {"none": "no_samples"}.get(s.kind, s.kind), 0)
    runs = 0
    for n, (inp, expected) in enumerate(s.pairs, 1):
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
