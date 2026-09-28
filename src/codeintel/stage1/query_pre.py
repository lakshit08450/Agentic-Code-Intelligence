"""Query preprocessing. Sees only the query text (R1)."""

from __future__ import annotations

import re
from collections.abc import Callable

from codeintel.common.config import QueryPreCfg

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def build_query_pre(cfg: QueryPreCfg) -> Callable[[str], str]:
    def pre(text: str) -> str:
        if cfg.strip_latex_dollar:
            text = text.replace("$", "")
        if cfg.normalize_ws:
            text = text.replace("\r\n", "\n")
            text = _WS.sub(" ", text)
            text = _BLANKS.sub("\n\n", text).strip()
        return text

    return pre
