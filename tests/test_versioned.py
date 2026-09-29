"""Path B (plan Section 12): incremental indexing, unit identity, --rev and --range, on a tiny git repo."""

from __future__ import annotations

import json

from codeintel.common import proc
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.versioned.ingest import git_versions, jsonl_versions, snapshot_versions
from codeintel.versioned.search import search_range, search_rev
from codeintel.versioned.store import Store
from codeintel.versioned.units import split
from conftest import FakeBackend, make_cfg

ADD = "def add(a, b):\n    return a + b\n"
REV = "def reverse_string(s):\n    return s[::-1]\n"
SORT = "def sort_numbers(xs):\n    return sorted(xs)\n"
SORT2 = "def sort_numbers(xs):\n    return sorted(xs, reverse=True)\n"


def git(repo, *args):
    return proc.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], capture_output=True, check=True)


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "add.py").write_text(ADD, encoding="utf-8")
    (repo / "rev.py").write_text(REV, encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "v1"); git(repo, "tag", "v1")
    (repo / "sort.py").write_text(SORT, encoding="utf-8")                   # added
    git(repo, "mv", "rev.py", "strings.py")                                 # renamed, unchanged
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "v2"); git(repo, "tag", "v2")
    (repo / "sort.py").write_text(SORT2, encoding="utf-8")                  # modified
    git(repo, "rm", "-q", "add.py")                                         # removed
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "v3"); git(repo, "tag", "v3")
    return repo


def new_store(tmp_path, name="store"):
    backend = FakeBackend()
    enc = PrePostPipelineEncoder(make_cfg(tmp_path / f"emb_{name}"), backends=[backend])
    return Store(tmp_path / name, enc), backend


def test_incremental_git_index_and_identity(tmp_path):
    repo = make_repo(tmp_path)
    store, _ = new_store(tmp_path)
    s1, s2, s3 = (store.add_version(v) for v in git_versions(repo, ["v1", "v2", "v3"]))
    assert (s1["added"], s1["embedded"]) == (2, 2)
    assert s2["renamed_or_moved"] == 1 and s2["added"] == 1 and s2["embedded"] == 1 and s2["reused"] == 2
    assert s3["modified"] == 1 and s3["removed"] == 1 and s3["embedded"] == 1
    uid_v1 = store.db.execute("SELECT unit_id FROM revisions WHERE path='rev.py'").fetchone()[0]
    uid_v2 = store.db.execute("SELECT unit_id FROM revisions WHERE path='strings.py'").fetchone()[0]
    assert uid_v1 == uid_v2  # the rename kept the unit


def test_search_rev_respects_versions(tmp_path):
    repo = make_repo(tmp_path)
    store, _ = new_store(tmp_path)
    for v in git_versions(repo, ["v1", "v2", "v3"]):
        store.add_version(v)
    paths_v1 = {r["path"] for r in search_rev(store, "add two numbers", "v1", k=10)["results"]}
    paths_v3 = {r["path"] for r in search_rev(store, "add two numbers", "v3", k=10)["results"]}
    assert paths_v1 == {"add.py", "rev.py"}
    assert paths_v3 == {"strings.py", "sort.py"}
    snippet = next(r for r in search_rev(store, "sort", "v3")["results"] if r["path"] == "sort.py")["snippet"]
    assert "reverse=True" in snippet
    assert {"encode", "search"} <= set(search_rev(store, "x", "v2")["ms"])


def test_search_range_collapses_units_with_history(tmp_path):
    repo = make_repo(tmp_path)
    store, _ = new_store(tmp_path)
    for v in git_versions(repo, ["v1", "v2", "v3"]):
        store.add_version(v)
    res = search_range(store, "sort numbers", "v1", "v3", k=10)["results"]
    assert len({r["unit_id"] for r in res}) == len(res) == 3  # add, rev->strings (renamed), sort (2 revisions)
    sort = next(r for r in res if r["path"] == "sort.py")
    assert [h["status"] for h in sort["history"]][:2] == ["introduced", "modified"]
    add = next(r for r in res if r["path"] == "add.py")
    assert add["history"][-1]["status"] == "removed"


def test_windows_split_long_files():
    text = "\n".join(f"x{i} = {i}" for i in range(400))
    pieces = split("k", "big.py", text)
    assert len(pieces) > 1 and pieces[0].start_line == 1 and pieces[1].start_line == 61
    assert pieces[-1].end_line == 400


def test_jsonl_and_snapshot_sources(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"id": "u1", "text": ADD}) + "\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps({"id": "u1", "text": SORT}) + "\n" + json.dumps({"id": "u2", "text": ADD}) + "\n", encoding="utf-8")
    store, _ = new_store(tmp_path, "js")
    s = [store.add_version(v) for v in jsonl_versions([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])]
    assert s[1]["modified"] == 1 and s[1]["added"] == 1 and s[1]["embedded"] == 1  # ADD reused
    for name, files in (("s1", {"m.py": ADD}), ("s2", {"m.py": ADD, "n.py": REV})):
        d = tmp_path / "snaps" / name
        d.mkdir(parents=True)
        for f, t in files.items():
            (d / f).write_text(t, encoding="utf-8")
    store2, _ = new_store(tmp_path, "snap")
    s = [store2.add_version(v) for v in snapshot_versions([tmp_path / "snaps" / "s1", tmp_path / "snaps" / "s2"])]
    assert s[1]["unchanged"] == 1 and s[1]["added"] == 1
