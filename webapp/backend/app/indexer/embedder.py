"""
Embedding module for CodeLens, backed by the PRISM model server.

Replaces the earlier random-vector mock: embeddings now come from PRISM's POST /embed (the same
pretrained models, prompts and fusion weighting as the PRISM pipeline: Qwen3-Embedding-0.6B +
EmbeddingGemma-300m, fused, 1,792 dimensions, L2-normalised). The PRISM server must be running
(see ../../README.md, "Start"). Interface unchanged: embed, embed_batch, embed_code_chunk, embed_query.
"""

import numpy as np

from app.agent import prism_client
from app.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

BATCH = 32  # texts per /embed request (PRISM accepts at most 64)


class Embedder:
    """Fused PRISM embeddings via HTTP (no model weights in this process)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dim = EMBEDDING_DIM

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = [prism_client.embed(texts[i:i + BATCH], kind) for i in range(0, len(texts), BATCH)]
        vecs = np.vstack(out).astype(np.float32)
        if vecs.shape[1] != self.dim:
            raise RuntimeError(f"PRISM returned {vecs.shape[1]}-d vectors, config EMBEDDING_DIM is {self.dim}")
        return vecs

    def embed(self, text: str) -> np.ndarray:
        """Embed a single document text."""
        return self._embed([text], "document")[0]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of document texts (code chunks)."""
        return self._embed(texts, "document")

    def embed_code_chunk(self, code: str, file_path: str = "", node_name: str = "") -> np.ndarray:
        """Embed a code chunk (file path and symbol name prepended as context)."""
        parts = []
        if file_path:
            parts.append(f"File: {file_path}")
        if node_name:
            parts.append(f"Name: {node_name}")
        parts.append(code)
        return self.embed("\n".join(parts))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (PRISM query prompt)."""
        return self._embed([query], "query")[0]


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
