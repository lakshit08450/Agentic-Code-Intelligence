"""
CodeLens Configuration
All settings are centralized here with sensible defaults.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "indices"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "codelens.db"

# Ensure dirs exist
for d in [DATA_DIR, INDEX_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding Model ───────────────────────────────────
EMBEDDING_MODEL_NAME = "prism-fusion (Qwen3-Embedding-0.6B + EmbeddingGemma-300m via PRISM /embed)"
EMBEDDING_DIM = 1792  # PRISM fusion: 1,024 (Qwen3) + 768 (EmbeddingGemma)

# ── Cross-Encoder (Refiner) ───────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ──────────────────────────────────────────
CHUNK_MIN_LINES = 5
CHUNK_MAX_LINES = 100
CHUNK_OVERLAP_LINES = 10

# ── Search ────────────────────────────────────────────
SEMANTIC_TOP_K = 50
STRUCTURAL_TOP_K = 30
RERANK_TOP_K = 20
FINAL_TOP_K = 10

# ── FAISS ─────────────────────────────────────────────
FAISS_NPROBE = 10
FAISS_USE_IVF = True
FAISS_NLIST = 100  # Number of Voronoi cells (tune based on corpus size)

# ── Server ────────────────────────────────────────────
HOST = os.getenv("CODELENS_HOST", "0.0.0.0")
PORT = int(os.getenv("CODELENS_PORT", "8000"))
CORS_ORIGINS = os.getenv("CODELENS_CORS", "http://localhost:5173").split(",")

# ── Git / Versions ────────────────────────────────────
DEFAULT_BRANCH = "main"
MAX_INDEXED_VERSIONS = 10
