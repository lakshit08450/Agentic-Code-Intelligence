"""Smoke, cache and fusion tests for the Stage 1 encoder (no model downloads)."""

from __future__ import annotations

import numpy as np
from mteb.types import PromptType

from codeintel.stage1.encoder import PrePostPipelineEncoder
from conftest import FakeBackend, make_cfg

DOCS = ["def f(x):\n    return x + 1\n", "print(sum(map(int, input().split())))", "n = int(input())\nprint(n * 2)"]
QUERIES = ["Add one to x.", "Read numbers and print their sum.", "Double the integer n."]


def batches(texts, ids=None, size=2):
    ids = ids or [str(i) for i in range(len(texts))]
    return [{"id": ids[i : i + size], "text": texts[i : i + size]} for i in range(0, len(texts), size)]


def test_shape_and_unit_norm(tmp_path):
    enc = PrePostPipelineEncoder(make_cfg(tmp_path), backends=[FakeBackend()])
    v = enc.encode(batches(DOCS), prompt_type=PromptType.document)
    assert v.shape == (3, 64) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)


def test_query_prompt_applied_and_cache_reused(tmp_path):
    cfg = make_cfg(tmp_path)
    be = FakeBackend()
    enc = PrePostPipelineEncoder(cfg, backends=[be])
    q = enc.encode(batches(QUERIES), prompt_type=PromptType.query)
    d = enc.encode(batches(QUERIES), prompt_type=PromptType.document)
    assert not np.allclose(q, d)  # "q: " prompt only on the query side
    calls = be.calls
    # new encoder instance, same cache dir: all hits, backend never called
    be2 = FakeBackend()
    enc2 = PrePostPipelineEncoder(make_cfg(tmp_path), backends=[be2])
    np.testing.assert_allclose(enc2.encode(batches(QUERIES), prompt_type=PromptType.query), q, atol=1e-7)
    assert be2.calls == 0 and calls > 0
    assert enc2.stats == {"hits": 3, "misses": 0}


def test_cache_miss_error_mode(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.on_cache_miss = "error"
    enc = PrePostPipelineEncoder(cfg, backends=[FakeBackend()])
    try:
        enc.encode(batches(DOCS), prompt_type=PromptType.document)
    except RuntimeError as e:
        assert "cache misses" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_fusion_identity(tmp_path):
    w = [0.7, 0.3]
    b1, b2 = FakeBackend(salt="a"), FakeBackend(salt="b")
    fused = PrePostPipelineEncoder(make_cfg(tmp_path / "f", 2, w), backends=[b1, b2])
    q = fused.encode(batches(QUERIES), prompt_type=PromptType.query)
    d = fused.encode(batches(DOCS), prompt_type=PromptType.document)
    single = []
    for be in (b1, b2):
        e = PrePostPipelineEncoder(make_cfg(tmp_path / be.salt), backends=[be])
        single.append(
            e.encode(batches(QUERIES), prompt_type=PromptType.query)
            @ e.encode(batches(DOCS), prompt_type=PromptType.document).T
        )
    np.testing.assert_allclose(q @ d.T, w[0] * single[0] + w[1] * single[1], atol=1e-5)
