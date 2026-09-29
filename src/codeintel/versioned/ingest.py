"""Version sources (plan Section 12.1): git revisions (first-parent), snapshot directories, JSONL files.

Each source yields Version(label, source_ref, items, renames) where items maps unit key -> (path, text)
and renames maps an old key to its new key (git rename detection; empty for other sources).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.common import proc

DEFAULT_EXTS = (".py",)


@dataclass
class Version:
    label: str
    source_ref: str
    items: dict[str, tuple[str, str]]
    renames: dict[str, str] = field(default_factory=dict)


def _git(repo: str | Path, *args: str) -> str:
    r = proc.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return r.stdout.decode("utf-8", "replace")


def git_versions(repo: str | Path, revs: list[str], exts: tuple[str, ...] = DEFAULT_EXTS) -> list[Version]:
    out, prev = [], None
    for rev in revs:
        sha = _git(repo, "rev-parse", rev).strip()
        paths = [p for p in _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines() if p.endswith(exts)]
        items = {p: (p, _git(repo, "show", f"{sha}:{p}")) for p in paths}
        renames = {}
        if prev is not None:
            for line in _git(repo, "diff", "--name-status", "-M", prev, sha).splitlines():
                parts = line.split("\t")
                if parts and parts[0].startswith("R") and len(parts) == 3 and parts[2].endswith(exts):
                    renames[parts[1]] = parts[2]
        out.append(Version(rev, sha, items, renames))
        prev = sha
    return out


def git_first_parent(repo: str | Path, rev_range: str) -> list[str]:
    """Revisions of `A..B` (or a single ref) along the first-parent chain, oldest first."""
    return list(reversed(_git(repo, "rev-list", "--first-parent", rev_range).split()))


def snapshot_versions(dirs: list[str | Path], exts: tuple[str, ...] = DEFAULT_EXTS) -> list[Version]:
    out = []
    for d in map(Path, dirs):
        items = {}
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix in exts:
                rel = f.relative_to(d).as_posix()
                items[rel] = (rel, f.read_text(encoding="utf-8", errors="replace"))
        out.append(Version(d.name, str(d), items))
    return out


def jsonl_versions(files: list[str | Path]) -> list[Version]:
    out = []
    for f in map(Path, files):
        items = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    items[str(rec["id"])] = (rec.get("path") or str(rec["id"]), rec["text"])
        out.append(Version(f.stem, str(f), items))
    return out
