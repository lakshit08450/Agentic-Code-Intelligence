"""Document preprocessing. Code is never rewritten; indentation is preserved (R1: text only)."""

from __future__ import annotations

from collections.abc import Callable

from codeintel.common.config import DocPreCfg


def build_doc_pre(cfg: DocPreCfg) -> Callable[[str], str]:
    def pre(text: str) -> str:
        if cfg.normalize_ws:
            lines = text.replace("\r\n", "\n").split("\n")
            text = "\n".join(line.rstrip() for line in lines).strip("\n")
        return text

    return pre
