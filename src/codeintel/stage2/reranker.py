"""Stage 2 scoring and the mteb cross-encoder (plan Sections 11.4 and 11.6).

score = cosine + w_pass * [PASS] * (trivial_pass_factor if the expected outputs are trivial else 1)
               - w_wrong * [WRONG] * (0 if multi_answer_neutral and the statement allows several answers)
               - w_err * [ERROR]
TIMEOUT and UNKNOWN add nothing. Only query text and document text are used (R1).

Fix 3 (multi-answer neutral): statements that accept any valid answer ("print any of them") make a
correct program look WRONG on the sample, so WRONG is not penalised there.
Fix 4 (trivial outputs): when every sample output is a single trivial token (0, 1, -1, YES, NO, ...)
many wrong programs pass by chance (56% of such validation queries), so PASS is down-weighted.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from codeintel.common.config import REPO_ROOT
from codeintel.stage2.sampleio import Samples
from codeintel.stage2.sandbox import ExecCache
from codeintel.stage2.verifier import samples_for, verify

log = logging.getLogger(__name__)

_MULTI = re.compile(
    r"(?i)any of them|multiple (?:possible |valid |correct )?(?:answers|solutions)|any (?:valid|correct|such|possible) "
    r"(?:answer|solution|one|sequence|permutation|array|string|arrangement|way|order)|if there are (?:several|many|multiple)|"
    r"print any|output any|you (?:can|may) print any|any one of them"
)
_TRIVIAL = re.compile(r"^(?:-?[01]|-1|YES|NO|Yes|No|yes|no|True|False|IMPOSSIBLE|Impossible|POSSIBLE|Possible)$")


def is_multi_answer(statement: str) -> bool:
    return bool(_MULTI.search(statement))


def is_trivial_output(samples: Samples) -> bool:
    return bool(samples.pairs) and all(_TRIVIAL.match(o.strip()) for _, o in samples.pairs)


@dataclass
class Stage2Cfg:
    k: int = 20
    w_pass: float = 1.0
    w_wrong: float = 0.0
    w_err: float = 0.0
    trivial_pass_factor: float = 1.0
    multi_answer_neutral: bool = True
    workers: int = 12
    stage1_config: str = "configs/stage1_final.yaml"


def load_stage2_cfg(path: str | Path) -> Stage2Cfg:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return Stage2Cfg(**yaml.safe_load(p.read_text(encoding="utf-8")))


def stage2_score(cos: float, outcome: str, statement: str, cfg: Stage2Cfg) -> float:
    s = cos
    if outcome == "PASS":
        s += cfg.w_pass * (cfg.trivial_pass_factor if is_trivial_output(samples_for(statement)) else 1.0)
    elif outcome == "WRONG":
        if not (cfg.multi_answer_neutral and is_multi_answer(statement)):
            s -= cfg.w_wrong
    elif outcome == "ERROR":
        s -= cfg.w_err
    return s


class ExecutionReranker:
    """mteb CrossEncoderProtocol: predict() scores aligned (query, document) pairs.

    No `encode` method on purpose, so mteb routes it through SearchCrossEncoderWrapper over the
    first-stage top-k (task.convert_to_reranking). Cosine comes from the Stage 1 embedding cache.
    """

    def __init__(self, cfg: Stage2Cfg | str | Path = "configs/stage2_final.yaml", device: str | None = None) -> None:
        from mteb.models.model_meta import ModelMeta

        from codeintel.stage1.encoder import PrePostPipelineEncoder

        self.cfg = cfg if isinstance(cfg, Stage2Cfg) else load_stage2_cfg(cfg)
        self.encoder = PrePostPipelineEncoder(self.cfg.stage1_config, device=device)
        self.cache = ExecCache()
        self.stats: dict[str, int] = {}
        self._lock = threading.Lock()
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name="prism-codeintel/exec-rerank", revision=self.encoder.mteb_model_meta.revision)
        )

    @staticmethod
    def _texts(loader) -> list[str]:
        return [t for batch in loader for t in batch["text"]]

    def predict(self, inputs1, inputs2, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs: Any):
        queries, docs = self._texts(inputs1), self._texts(inputs2)
        if len(queries) != len(docs):
            raise ValueError("inputs1 and inputs2 must be aligned pairs")
        uq, ud = sorted(set(queries)), sorted(set(docs))
        qv = dict(zip(uq, self.encoder.embed(uq, is_query=True)))
        dv = dict(zip(ud, self.encoder.embed(ud, is_query=False)))
        cos = [float(qv[q] @ dv[d]) for q, d in zip(queries, docs)]
        t0, done = time.perf_counter(), [0]

        def one(pair):
            q, d = pair
            v = verify(q, d, self.cache)
            with self._lock:
                self.stats[v.outcome] = self.stats.get(v.outcome, 0) + 1
                done[0] += 1
                if done[0] % 1000 == 0 or done[0] == len(queries):
                    el = time.perf_counter() - t0
                    log.info("rerank %d/%d pairs, %.1f min, eta %.0f min, %s", done[0], len(queries), el / 60,
                             (len(queries) - done[0]) * el / done[0] / 60, self.stats)
            return v.outcome

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            outcomes = list(ex.map(one, zip(queries, docs)))
        return np.array([stage2_score(c, o, q, self.cfg) for c, o, q in zip(cos, outcomes, queries)], dtype=np.float32)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
