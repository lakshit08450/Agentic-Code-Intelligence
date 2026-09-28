"""Stage 1 encoder: preprocessing + one or more embedders + fusion, behind mteb's AbsEncoder.

R1: only `batch["text"]` is read. IDs and any other columns in the batch are ignored.
Fusion: v = concat(sqrt(w_i) * u_i), then L2. For unit u_i this gives
cos(v_q, v_d) = sum_i w_i * cos_i (plan Section 5).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

from codeintel.common import proc
from codeintel.common.cache import EmbeddingStore
from codeintel.common.config import REPO_ROOT, ModelCfg, Stage1Cfg, load_cfg
from codeintel.stage1.doc_pre import build_doc_pre
from codeintel.stage1.query_pre import build_query_pre

log = logging.getLogger(__name__)


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Backend(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _extended_attention_mask(self: Any, attention_mask: Any, input_shape: Any = None, *args: Any, **kwargs: Any) -> Any:
    """transformers 4.x `ModuleUtilsMixin.get_extended_attention_mask` for 2-D padding masks.

    Removed in transformers 5; CodeRankEmbed's remote NomicBert code still calls it.
    """
    import torch

    if attention_mask.dim() == 3:
        ext = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        ext = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"unsupported attention_mask shape {tuple(attention_mask.shape)}")
    dtype = next(self.parameters()).dtype
    ext = ext.to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def _patch_extended_attention_mask(model: Any) -> None:
    for module in model.modules():
        cls = type(module)
        if cls.__name__.endswith("PreTrainedModel") or not hasattr(module, "config"):
            continue
        if "get_extended_attention_mask" in getattr(cls, "__dict__", {}) or hasattr(module, "get_extended_attention_mask"):
            continue
        cls.get_extended_attention_mask = _extended_attention_mask
        log.info("patched get_extended_attention_mask onto %s (transformers 5 compat)", cls.__name__)


class SentenceTransformerBackend:
    """Loads a sentence-transformers model; returns unit-norm float32 vectors."""

    def __init__(self, m: ModelCfg, device: str, batch_size: int, fp16_on_cuda: bool, attn_budget: int = 8192**2) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            m.id, revision=m.revision, device=device, trust_remote_code=m.trust_remote_code
        )
        if m.trust_remote_code:  # remote code written for transformers 4.x
            _patch_extended_attention_mask(self.model)
        self.model.max_seq_length = m.max_seq_length
        if device.startswith("cuda") and fp16_on_cuda:
            self.model.half()
        self.batch_size = batch_size
        self.attn_budget = attn_budget
        log.info("loaded %s rev=%s device=%s max_seq_length=%d", m.id, m.revision, device, m.max_seq_length)

    def _batches(self, texts: list[str]) -> list[list[int]]:
        """Length-sorted batches with batch_size * len^2 <= attn_budget (eager attention memory)."""
        tok = self.model.tokenizer
        cap = self.model.max_seq_length
        lens = [min(len(ids), cap) for ids in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: -lens[i])
        batches, cur = [], []
        for i in order:
            longest = lens[cur[0]] if cur else lens[i]
            if cur and (len(cur) + 1 > self.batch_size or (len(cur) + 1) * longest**2 > self.attn_budget):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()), np.float32)
        for b in self._batches(texts):
            out[b] = self.model.encode(
                [texts[i] for i in b],
                batch_size=len(b),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return l2(out)


def _git_rev() -> str:
    try:
        out = proc.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(
        self,
        cfg: Stage1Cfg | str | Path = "configs/stage1_final.yaml",
        *,
        device: str | None = None,
        backends: list[Backend] | None = None,
    ) -> None:
        self.cfg = cfg if isinstance(cfg, Stage1Cfg) else load_cfg(cfg, device=device)
        if device is not None:
            self.cfg.device = device
        self.weights = list(self.cfg.fusion_weights)
        self.query_pre = build_query_pre(self.cfg.query_pre)
        self.doc_pre = build_doc_pre(self.cfg.doc_pre)
        self.stores = [EmbeddingStore(self.cfg.cache_path(), m) for m in self.cfg.models]
        self._backends: list[Backend | None] = list(backends) if backends else [None] * len(self.cfg.models)
        self.stats = {"hits": 0, "misses": 0}
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(
                name=f"prism-codeintel/{self.cfg.name}",
                revision=_git_rev(),
                similarity_fn_name=ScoringFunction.COSINE,
                max_tokens=max(m.max_seq_length for m in self.cfg.models),
            )
        )

    def _backend(self, i: int) -> Backend:
        if self._backends[i] is None:
            self._backends[i] = SentenceTransformerBackend(
                self.cfg.models[i], self.cfg.device, self.cfg.batch_size, self.cfg.fp16_on_cuda, self.cfg.attn_budget
            )
        return self._backends[i]  # type: ignore[return-value]

    def model_texts(self, texts: list[str], is_query: bool, i: int) -> list[str]:
        """Exact strings fed to model i (preprocessing + that model's prompt); also the cache key text."""
        m = self.cfg.models[i]
        pre = self.query_pre if is_query else self.doc_pre
        prompt = m.query_prompt if is_query else m.doc_prompt
        return [prompt + pre(t) for t in texts]

    def fill_cache(self, texts: list[str], is_query: bool, i: int, chunk: int = 512, progress: bool = False) -> int:
        """Embed and store every text not yet cached for model i. Returns the number computed."""
        store = self.stores[i]
        todo = store.missing(self.model_texts(texts, is_query, i))
        if todo and self.cfg.on_cache_miss == "error":
            raise RuntimeError(f"{len(todo)} cache misses for {self.cfg.models[i].id} and on_cache_miss=error")
        for s in range(0, len(todo), chunk):
            part = todo[s : s + chunk]
            store.put(part, self._backend(i).encode(part))
            if progress:
                log.info("model %d: %d/%d embedded", i, min(s + chunk, len(todo)), len(todo))
        return len(todo)

    def embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        parts = []
        for i, w in enumerate(self.weights):
            mt = self.model_texts(texts, is_query, i)
            n_missing = len(self.stores[i].missing(mt))
            self.stats["misses"] += n_missing
            self.stats["hits"] += len(set(mt)) - n_missing
            self.fill_cache(texts, is_query, i)
            parts.append(np.sqrt(w) * l2(self.stores[i].get(mt)))
        return l2(np.concatenate(parts, axis=1))

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        return self.embed(texts, is_query=prompt_type == PromptType.query)
