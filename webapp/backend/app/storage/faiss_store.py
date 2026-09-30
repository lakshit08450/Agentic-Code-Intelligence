"""
FAISS vector store for CodeLens.
Manages semantic embeddings index on CPU.
"""

import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from app.config import INDEX_DIR, EMBEDDING_DIM, FAISS_NLIST, FAISS_NPROBE, FAISS_USE_IVF


class FAISSStore:
    """
    Manages FAISS index for semantic code search.
    Supports both flat (exact) and IVF (approximate) indices.
    """

    def __init__(self, index_name: str = "default", dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index_name = index_name
        self.index_path = INDEX_DIR / f"{index_name}.faiss"
        self.ids_path = INDEX_DIR / f"{index_name}_ids.npy"
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: list[str] = []
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create a new empty one."""
        if self.index_path.exists() and self.ids_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.chunk_ids = np.load(str(self.ids_path), allow_pickle=True).tolist()
        else:
            # Start with a flat index; will upgrade to IVF when enough vectors accumulate
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunk_ids = []

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]):
        """
        Add embeddings to the index.
        embeddings: (N, dim) float32 array, L2-normalized
        chunk_ids: list of corresponding chunk IDs
        """
        assert embeddings.shape[1] == self.dim, f"Expected dim {self.dim}, got {embeddings.shape[1]}"
        assert len(chunk_ids) == embeddings.shape[0], "Mismatch between embeddings and IDs"

        # Normalize for cosine similarity (inner product on normalized = cosine)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors.
        Returns list of (chunk_id, score) tuples, sorted by descending score.
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = FAISS_NPROBE

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def rebuild_ivf(self):
        """
        Rebuild index as IVF for faster search on large corpora.
        Should be called after bulk indexing when ntotal > FAISS_NLIST * 40.
        """
        if not FAISS_USE_IVF or self.index.ntotal < FAISS_NLIST * 40:
            return

        # Reconstruct all vectors
        vectors = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            vectors[i] = self.index.reconstruct(i)

        # Build IVF index
        quantizer = faiss.IndexFlatIP(self.dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, self.dim, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(vectors)
        ivf_index.add(vectors)
        ivf_index.nprobe = FAISS_NPROBE

        self.index = ivf_index

    def save(self):
        """Persist index and ID mapping to disk."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        np.save(str(self.ids_path), np.array(self.chunk_ids, dtype=object))

    def clear(self):
        """Reset the index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunk_ids = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.ids_path.exists():
            self.ids_path.unlink()

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0
