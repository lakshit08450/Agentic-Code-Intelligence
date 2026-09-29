"""Output comparison (plan Section 11.2): token-wise after split(); numbers within 1e-6 (relative
or absolute) unless both tokens are integers (then exact); optional case-insensitive mode."""

from __future__ import annotations

import math
import re

_INT = re.compile(r"^[+-]?\d+$")


def _num(tok: str) -> float | None:
    try:
        v = float(tok)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _canon_int(tok: str) -> str:
    """Canonical digit string (no int(): CPython refuses > 4,300-digit conversions)."""
    sign = "-" if tok[0] == "-" else ""
    digits = tok.lstrip("+-").lstrip("0") or "0"
    return ("" if digits == "0" else sign) + digits


def tokens_match(a: str, b: str, tol: float = 1e-6, case_insensitive: bool = False) -> bool:
    if a == b:
        return True
    if case_insensitive and a.lower() == b.lower():
        return True
    if _INT.match(a) and _INT.match(b):
        return _canon_int(a) == _canon_int(b)
    x, y = _num(a), _num(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= tol * max(1.0, abs(y))


def outputs_match(got: str, expected: str, tol: float = 1e-6, case_insensitive: bool = False) -> bool:
    g = got.replace("\r\n", "\n").split()
    e = expected.replace("\r\n", "\n").split()
    return len(g) == len(e) and all(tokens_match(x, y, tol, case_insensitive) for x, y in zip(g, e))
