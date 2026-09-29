"""Units for versioned retrieval (plan Section 12.1): one file or one JSONL snippet; long files are
split into ~80-line windows (20 lines overlap) that share the unit key and collapse back at search time."""

from __future__ import annotations

from dataclasses import dataclass

MAX_LINES = 300  # files longer than this are windowed
WINDOW = 80
OVERLAP = 20


@dataclass(frozen=True)
class Piece:
    key: str          # unit identity hint: file path or JSONL id
    path: str
    start_line: int   # 1-based, inclusive
    end_line: int
    text: str


def split(key: str, path: str, text: str, max_lines: int = MAX_LINES, window: int = WINDOW, overlap: int = OVERLAP) -> list[Piece]:
    lines = text.replace("\r\n", "\n").split("\n")
    if len(lines) <= max_lines:
        return [Piece(key, path, 1, len(lines), text)]
    step = window - overlap
    out = []
    for start in range(0, len(lines), step):
        chunk = lines[start : start + window]
        out.append(Piece(key, path, start + 1, start + len(chunk), "\n".join(chunk)))
        if start + window >= len(lines):
            break
    return out
