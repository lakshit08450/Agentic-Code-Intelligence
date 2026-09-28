"""Text-keyed embedding cache.

One directory per (model id, model fingerprint); inside, append-only chunk files
`chunk_XXXXX.npz` holding `keys` (sha1 of the preprocessed text) and `vecs` (float32,
unit norm). Writes are atomic (tmp file + os.replace), so an interrupted encode
resumes from the last complete chunk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from codeintel.common.config import ModelCfg
from codeintel.common.hashing import sha1_text


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


class EmbeddingStore:
    def __init__(self, root: Path, model: ModelCfg) -> None:
        self.dir = Path(root) / f"{_slug(model.id)}__{model.fingerprint()}"
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "model.json"
        if not meta.exists():
            meta.write_text(json.dumps(model.__dict__, indent=2), encoding="utf-8")
        self._index: dict[str, int] = {}
        self._blocks: list[np.ndarray] = []
        self._n = 0
        self._next_chunk = 0
        for f in sorted(self.dir.glob("chunk_*.npz")):
            with np.load(f) as z:
                self._add(list(z["keys"]), z["vecs"])
            self._next_chunk = max(self._next_chunk, int(f.stem.split("_")[1]) + 1)
        self._matrix: np.ndarray | None = None

    def _add(self, keys: list[str], vecs: np.ndarray) -> None:
        self._blocks.append(vecs.astype(np.float32, copy=False))
        for i, k in enumerate(keys):
            self._index.setdefault(str(k), self._n + i)
        self._n += len(keys)
        self._matrix = None

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def key(text: str) -> str:
        return sha1_text(text)

    def missing(self, texts: list[str]) -> list[str]:
        """Unique texts (first occurrence order) with no cached vector."""
        seen, out = set(), []
        for t in texts:
            k = self.key(t)
            if k not in self._index and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def put(self, texts: list[str], vecs: np.ndarray) -> None:
        if len(texts) != len(vecs):
            raise ValueError("texts and vecs length mismatch")
        if not texts:
            return
        keys = np.array([self.key(t) for t in texts])
        vecs = np.asarray(vecs, dtype=np.float32)
        final = self.dir / f"chunk_{self._next_chunk:05d}.npz"
        tmp = final.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vecs=vecs)
        os.replace(tmp, final)
        self._next_chunk += 1
        self._add(list(keys), vecs)

    def get(self, texts: list[str]) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.concatenate(self._blocks) if self._blocks else np.zeros((0, 0), np.float32)
        rows = [self._index[self.key(t)] for t in texts]
        return self._matrix[rows]
