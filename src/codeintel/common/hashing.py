"""Stable hashes used as cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any, n: int = 12) -> str:
    """Short hash of a JSON-serialisable config fragment (key order independent)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha1_text(blob)[:n]
