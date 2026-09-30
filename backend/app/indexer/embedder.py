"""
ONNX-based embedding module for CodeLens.
Currently mocked using numpy for testing without heavy ML dependencies.
"""

import numpy as np
from typing import Union
from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM


class Embedder:
    """
    Mock embedder that generates random vectors for testing.
    Replaces sentence-transformers to skip huge downloads.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.dim = EMBEDDING_DIM

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a dummy vector."""
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.random(self.dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of texts."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    def embed_code_chunk(self, code: str, file_path: str = "", node_name: str = "") -> np.ndarray:
        """Embed a code chunk."""
        parts = []
        if file_path:
            parts.append(f"File: {file_path}")
        if node_name:
            parts.append(f"Name: {node_name}")
        parts.append(code)
        text = "\n".join(parts)
        return self.embed(text)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        return self.embed(f"search: {query}")


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
