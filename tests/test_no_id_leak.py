"""R1: scoring sees only query and document text.

1. Behavioural: run retrieval through mteb's own SearchEncoderWrapper, then shuffle every
   query and document ID (and the row order). The ranking of *texts* must not change.
2. Static: scoring packages must not reference qrels, ID columns or meta_information.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import mteb
from datasets import Dataset
from mteb.models.search_wrappers import SearchEncoderWrapper

from codeintel.stage1.encoder import PrePostPipelineEncoder
from conftest import FakeBackend, make_cfg

DOCS = [
    "n = int(input())\nprint(n * 2)",
    "a, b = map(int, input().split())\nprint(a + b)",
    "s = input()\nprint(s[::-1])",
    "import sys\nprint(max(map(int, sys.stdin.read().split()[1:])))",
    "t = int(input())\nfor _ in range(t):\n    print(sum(map(int, input().split())))",
    "print('YES' if input() == input()[::-1] else 'NO')",
]
QUERIES = [
    "Given an integer n, print twice n.",
    "Given two integers a and b, print a + b.",
    "Reverse the given string.",
    "Print the maximum of the n numbers.",
]


def ranked_texts(doc_ids, query_ids, tmp_path, seed_order):
    task = mteb.get_task("AppsRetrieval")
    order = list(range(len(DOCS)))
    random.Random(seed_order).shuffle(order)
    corpus = Dataset.from_dict(
        {"id": [doc_ids[i] for i in order], "text": [DOCS[i] for i in order], "title": [""] * len(DOCS)}
    )
    queries = Dataset.from_dict({"id": query_ids, "text": QUERIES})
    enc = PrePostPipelineEncoder(make_cfg(tmp_path), backends=[FakeBackend()])
    wrapper = SearchEncoderWrapper(enc)
    kw = dict(task_metadata=task.metadata, hf_split="test", hf_subset="default", encode_kwargs={"batch_size": 2})
    wrapper.index(corpus, **kw)
    res = wrapper.search(queries, top_k=len(DOCS), **kw)
    id2doc = dict(zip(doc_ids, DOCS))
    return [
        [id2doc[d] for d, _ in sorted(res[qid].items(), key=lambda kv: -kv[1])]
        for qid in query_ids
    ]


def test_ranking_invariant_to_id_shuffle(tmp_path):
    base_docs = [f"doc{i}" for i in range(len(DOCS))]
    base_qs = [f"q{i}" for i in range(len(QUERIES))]
    ref = ranked_texts(base_docs, base_qs, tmp_path / "a", seed_order=0)
    rng = random.Random(13)
    for trial in range(3):
        doc_ids = [str(rng.randrange(10**9)) for _ in DOCS]
        q_ids = list(base_docs[: len(QUERIES)])  # adversarial: query IDs equal to other doc IDs
        rng.shuffle(q_ids)
        got = ranked_texts(doc_ids, q_ids, tmp_path / f"b{trial}", seed_order=trial + 1)
        assert got == ref


FORBIDDEN = re.compile(r"qrels|meta_information|query-id|corpus-id|relevant_docs|\[\s*[\"']id[\"']\s*\]")
SCORING_PACKAGES = ["stage1", "stage2"]


def test_scoring_code_never_touches_ids_or_labels():
    root = Path(__file__).resolve().parents[1] / "src" / "codeintel"
    offenders = []
    for pkg in SCORING_PACKAGES:
        for f in (root / pkg).rglob("*.py"):
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if FORBIDDEN.search(code):
                    offenders.append(f"{f.name}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
